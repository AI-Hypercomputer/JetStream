"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""AttentionOps Module"""

import dataclasses
import jax
from jax import numpy as jnp
import jax.experimental
from jax.sharding import NamedSharding
from inference import kernel
from inference.nn.module import Module
from inference.utils import *


@register_flat_dataclass_as_pytree
@dataclasses.dataclass
class KVCache:
  k: jax.Array | NamedSharding
  v: jax.Array | NamedSharding


@register_flat_dataclass_as_pytree
@dataclasses.dataclass
class AttentionMetadata:
  prefill_length: (
      jax.Array | NamedSharding
  )  # shape: []; Prefill True length without padding
  prefill_pos: jax.Array | NamedSharding  # shape: [padded length]
  prefill_page_table: jax.Array | NamedSharding  # shape: [max_len // page_size]

  generate_pos: jax.Array | NamedSharding  # shape: [generate_batch_size]
  generate_page_table: (
      jax.Array | NamedSharding
  )  # shape: [generate_batch, max_len // page_size]


class AttentionOps(Module):

  def __init__(self, num_attn_heads, num_kv_heads, head_dim):
    super().__init__()
    self.num_attn_heads = num_attn_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim

  def _write_prefill_kv_to_kv_cache(
      self, k, v, kv_cache: KVCache, unpadded_len, page_table
  ):
    padded_prefill_len = k.shape[0]
    num_kv_heads_per_device = kv_cache.k.shape[0]
    page_size = kv_cache.k.shape[2]
    num_pages = padded_prefill_len // page_size
    num_pages = jnp.where(num_pages < 1, 1, num_pages)
    num_active_pages, reminder = jnp.divmod(unpadded_len, page_size)
    num_active_pages += jnp.where(reminder > 0, 1, 0)

    k = k.transpose((1, 0, 2))
    v = v.transpose((1, 0, 2))
    # kv shape after the change: (num_kv_heads, num_pages, page_size, head_dim)
    k = k.reshape(
        (num_kv_heads_per_device, -1, page_size, self.head_dim)
    ).astype(kv_cache.k.dtype)
    v = v.reshape(
        (num_kv_heads_per_device, -1, page_size, self.head_dim)
    ).astype(kv_cache.v.dtype)

    def update_cond(carry):
      _, idx = carry
      return idx < num_active_pages

    def per_page_update(carry):
      kv_cache, idx = carry
      page_k = k[:, idx, :, :][:, None, :, :]
      page_v = v[:, idx, :, :][:, None, :, :]
      mapped_idx = page_table[idx]
      kv_cache.k = jax.lax.dynamic_update_slice_in_dim(
          kv_cache.k,
          page_k,
          mapped_idx,
          axis=1,
      )
      kv_cache.v = jax.lax.dynamic_update_slice_in_dim(
          kv_cache.v,
          page_v,
          mapped_idx,
          axis=1,
      )
      idx += 1
      return kv_cache, idx

    idx = 0
    kv_cache, idx = jax.lax.while_loop(
        update_cond, per_page_update, (kv_cache, idx)
    )

    return kv_cache

  def _write_generate_kv_to_kv_cache(self, k, v, kv_cache, pos, page_table):
    k = k.transpose((1, 0, 2))
    v = v.transpose((1, 0, 2))

    k = k.astype(kv_cache.k.dtype)
    v = v.astype(kv_cache.v.dtype)

    num_tokens = k.shape[1]
    num_kv_heads_per_device, num_pages, page_size, head_dim = kv_cache.k.shape
    page_idx, offset = jnp.divmod(pos, page_size)
    page_to_update = page_table[jnp.arange(0, num_tokens), page_idx]

    mapped_page_to_update = page_to_update * page_size + offset
    mapped_page_to_update = jnp.tile(
        mapped_page_to_update, num_kv_heads_per_device
    )

    kv_heads_axis_stride = (
        jnp.repeat(jnp.arange(0, num_kv_heads_per_device), num_tokens)
        * num_pages
        * page_size
    )
    mapped_page_to_update = kv_heads_axis_stride + mapped_page_to_update

    k = k.reshape(-1, head_dim)
    v = v.reshape(-1, head_dim)

    kv_cache.k = kv_cache.k.reshape(-1, head_dim)
    kv_cache.v = kv_cache.v.reshape(-1, head_dim)

    kv_cache.k = kv_cache.k.at[mapped_page_to_update, :].set(k)
    kv_cache.v = kv_cache.v.at[mapped_page_to_update, :].set(v)

    kv_cache.k = kv_cache.k.reshape(
        num_kv_heads_per_device, num_pages, page_size, head_dim
    )
    kv_cache.v = kv_cache.v.reshape(
        num_kv_heads_per_device, num_pages, page_size, head_dim
    )

    return kv_cache

  def _prefill(
      self, q, k, v, kv_cache: KVCache, attn_metadata: AttentionMetadata
  ):
    kv_cache = self._write_prefill_kv_to_kv_cache(
        k,
        v,
        kv_cache,
        attn_metadata.prefill_length,
        attn_metadata.prefill_page_table,
    )
    output = kernel.chunked_prefill_attention(
        q,
        kv_cache.k,
        kv_cache.v,
        attn_metadata.prefill_length,
        attn_metadata.prefill_page_table,
    )
    #        if self.num_attn_heads == self.num_kv_heads:
    #            output = kernel.vanilla_prefill_mha(q, k, v, attn_metadata.prefill_length)
    #        elif self.num_kv_heads == 1:
    #            output = kernel.vanilla_prefill_mqa(q, k, v, attn_metadata.prefill_length)
    #        else:
    #            output = kernel.vanilla_prefill_gqa(q, k, v, attn_metadata.prefill_length)
    return output, kv_cache

  def _generate(
      self, q, k, v, kv_cache: KVCache, attn_metadata: AttentionMetadata
  ):
    kv_cache = self._write_generate_kv_to_kv_cache(
        k,
        v,
        kv_cache,
        attn_metadata.generate_pos,
        attn_metadata.generate_page_table,
    )

    batch = q.shape[0]

    output = kernel.decode_attention(
        q,
        kv_cache.k,
        kv_cache.v,
        attn_metadata.generate_pos,
        attn_metadata.generate_page_table,
    )

    output = output.reshape((batch, -1))

    return output, kv_cache

  def _mixed_prefill_generate(
      self, q, k, v, kv_cache: KVCache, attn_metadata: AttentionMetadata
  ):
    total_len, num_attn_heads_per_device, head_dim = q.shape
    output = jnp.zeros(
        shape=(total_len, num_attn_heads_per_device * head_dim),
        dtype=q.dtype,
    )
    padded_prompt_length = attn_metadata.prefill_pos.shape[0]
    prefill_output, kv_cache = self._prefill(
        q[:padded_prompt_length, :, :],
        k[:padded_prompt_length, :, :],
        v[:padded_prompt_length, :, :],
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )

    generate_output, kv_cache = self._generate(
        q[padded_prompt_length:, :, :],
        k[padded_prompt_length:, :, :],
        v[padded_prompt_length:, :, :],
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )

    output = jax.lax.dynamic_update_slice_in_dim(
        output,
        prefill_output,
        start_index=0,
        axis=0,
    )

    output = jax.lax.dynamic_update_slice_in_dim(
        output,
        generate_output,
        start_index=padded_prompt_length,
        axis=0,
    )

    return output, kv_cache

  def __call__(
      self, q, k, v, kv_cache: KVCache, attn_metadata: AttentionMetadata
  ):
    # q, k, v has shape as (tokens, num_heads/devices, head_dim)
    if len(attn_metadata.generate_pos.shape) == 0:
      return self._prefill(q, k, v, kv_cache, attn_metadata)
    elif len(attn_metadata.prefill_pos.shape) == 0:
      return self._generate(q, k, v, kv_cache, attn_metadata)
    else:
      return self._mixed_prefill_generate(q, k, v, kv_cache, attn_metadata)
