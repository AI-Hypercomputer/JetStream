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

"""Attention kernel builder.

TODO: change to builder pattern instead of direct function call.
"""

import jax
from jax import numpy as jnp
from inference.kernel.attention import tpu

K_MASK = -2.3819763e38  # Set to a large negative number.


def vanilla_prefill_mha(q, k, v):
  """multi-head attention.

  q, k, v shape is (total_len, num_kv_heads, head_dim).
  output shape is (total_len, num_kv_heads * head_dim).
  """
  total_len = q.shape[0]
  num_kv_heads = q.shape[1]
  head_dim = q.shape[2]
  causal_mask = jnp.tril(jnp.ones(shape=(total_len, total_len))).astype(
      jnp.bool
  )
  mask = causal_mask[:, None, :]

  wei = jnp.einsum("tkh,skh->tks", q, k) / jnp.sqrt(head_dim)
  wei = jnp.where(mask, wei, K_MASK).astype(jnp.float32)
  wei = jax.nn.softmax(wei, axis=-1)
  out = jnp.einsum("tks,skh->tkh", wei, v)
  out = out.astype(q.dtype).reshape(total_len, num_kv_heads * head_dim)
  return out


def vanilla_prefill_mqa(q, k, v):
  """multi-query attention.

  q is (total_len, num_attn, head_dim).
  k/v shape is (total_len, 1, head_dim).
  output shape is (total_len, num_kv_heads * head_dim).
  """
  total_len = q.shape[0]
  head_dim = q.shape[2]
  k = jnp.squeeze(k, axis=1)
  v = jnp.squeeze(v, axis=1)
  causal_mask = jnp.tril(jnp.ones(shape=(total_len, total_len))).astype(
      jnp.bool
  )
  mask = causal_mask[:, None, :]

  wei = jnp.einsum("tah,sh->tas", q, k) / jnp.sqrt(head_dim)
  wei = jnp.where(mask, wei, K_MASK)
  wei = jax.nn.softmax(wei, axis=-1)
  out = jnp.einsum("tas,sh->tah", wei, v)
  out = out.astype(q.dtype).reshape((total_len, -1))
  return out


def vanilla_prefill_gqa(q, k, v):
  """group-query attention.

  q shape is (total_len, num_attn_heads, head_dim).
  k/v shape is (total_len, num_kv_heads, head_dim).
  output shape is (total_len, num_attn_heads * head_dim).
  """
  total_len, num_attn_heads, head_dim = q.shape
  num_kv_heads = k.shape[1]
  q = q.reshape(
      (total_len, num_kv_heads, num_attn_heads // num_kv_heads, head_dim)
  )
  causal_mask = jnp.tril(jnp.ones(shape=(total_len, total_len))).astype(
      jnp.bool
  )
  # padding_mask_row = jnp.where(jnp.arange(0, total_len) < len, 1, 0).astype(jnp.bool).reshape((-1, 1))
  # padding_mask_col = jnp.reshape(padding_mask_row, (1, -1))
  # padding_mask = jnp.logical_and(padding_mask_row, padding_mask_col)
  # mask = jnp.logical_and(padding_mask, causal_mask)[:, None, None, :]
  mask = causal_mask[:, None, None, :]

  wei = jnp.einsum("tkgh,skh->tkgs", q, k) / jnp.sqrt(head_dim)
  wei = jnp.where(mask, wei, K_MASK)
  wei = jax.nn.softmax(wei, axis=-1)
  out = jnp.einsum("tkgs,skh->tkgh", wei, v)
  out = out.astype(q.dtype).reshape((total_len, -1))
  return out


def chunked_prefill_attention(
    q, cache_k, cache_v, length, page_indices, accelerator="tpu"
):
  if accelerator == "tpu":
    return tpu.chunked_prefill_attention(
        q,
        cache_k,
        cache_v,
        length,
        page_indices,
    )
  else:
    raise NotImplementedError(f"not supported accelerate {accelerator}")


def decode_attention(q, cache_k, cache_v, pos, page_table, accelerator="tpu"):
  if accelerator == "tpu":
    # Heuristically set the pages per compute block.
    # TODO: tune the setup.
    pages_per_compute_block = 8
    _, _, _, head_dim = cache_k.shape
    q = q / jnp.sqrt(head_dim)
    seq_len = pos + 1

    output = tpu.paged_attention(
        q,
        cache_k,
        cache_v,
        seq_len,
        page_table,
        pages_per_compute_block=pages_per_compute_block,
    )
    return output

  else:
    raise NotImplementedError(f"not supported accelerate {accelerator}")
