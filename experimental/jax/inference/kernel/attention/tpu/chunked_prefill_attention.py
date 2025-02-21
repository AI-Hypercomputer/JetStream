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

"""Chunked prefill TPU kernel with paged kv cache."""

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from inference.kernel.attention.tpu.paged_attention import *


DEFAULT_MASK_VALUE = -2.3819763e38  # Set to a large negative number.


def chunked_prefill_attention_impl(
    length_ref,  # shape: (1,), smem,
    page_indices_ref,  # shape: (max_seq_len // page_size), smem,
    buffer_index_ref,  # shape: (1,), smem,
    q_ref,  # shape: (group_size, chunk, head_dim), vmem,
    k_pages_hbm_ref,  # shape: (num_kv_heads, num_pages, page_size, head_dim), hbm
    v_pages_hbm_ref,  # shape: (num_kv_heads, num_pages, page_size, head_dim), hbm
    out_ref,  # shape: (group_size, chunk, head_dim), vmem,
    l_ref,  # shape: (group_size, chunk, 1), vmem,
    m_ref,  # shape: (group_size, chunk, 1), vmem,
    k_vmem_buffer,  # shape: (2, page_per_chunk, page_size, head_dim), vmem,
    v_vmem_buffer,  # shape: (2, page_per_chunk, page_size, head_dim), vmem,
    sem,
):
  h = pl.program_id(0)
  page_size = k_pages_hbm_ref.shape[2]
  head_dim = k_pages_hbm_ref.shape[3]
  group_size = q_ref.shape[0]
  num_kv_heads = k_pages_hbm_ref.shape[0]
  chunk_size = q_ref.shape[1]
  length = length_ref[0]
  q_chunk_idx = jax.lax.div(length, chunk_size)
  reminder = jax.lax.rem(length, chunk_size)
  q_chunk_idx -= jnp.where(reminder > 0, 0, 1)
  out_ref[...] = jnp.zeros_like(out_ref)

  def create_kv_async_copy_descriptors(h, i, buffer_index):
    pages_to_load = chunk_size // page_size
    page_offset = i * pages_to_load
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref,
        None,
        k_vmem_buffer.at[buffer_index],
        None,
        sem,
        page_indices_ref,
        page_offset,
        pages_to_load,
        head_index=h,
    )
    async_copy_v = MultiPageAsyncCopyDescriptor(
        v_pages_hbm_ref,
        None,
        v_vmem_buffer.at[buffer_index],
        None,
        sem,
        page_indices_ref,
        page_offset,
        pages_to_load,
        head_index=h,
    )
    return async_copy_k, async_copy_v

  def next_block_indice(h, i):
    return jax.lax.cond(
        (i + 1) * chunk_size < length, lambda: (h, i + 1), lambda: (h + 1, 0)
    )

  def per_kv_chunk_body(i, _):
    @pl.when((i * chunk_size) < length)
    def body():
      buffer_index = buffer_index_ref[0]

      @pl.when(i == 0)
      def init():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)

        @pl.when(h == 0)
        def prefetch_first_kv():
          # prefetch the first kv chunk.
          async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
              h, i, buffer_index
          )
          async_copy_k.start()
          async_copy_v.start()

      next_h, next_i = next_block_indice(h, i)

      @pl.when((next_h < num_kv_heads) & (next_i <= q_chunk_idx))
      def prefetch_next_block():
        # prefetch the kv chunk for next iteration.
        next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
        async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
            next_h, next_i, next_buffer_index
        )

        async_copy_next_k.start()
        async_copy_next_v.start()
        buffer_index_ref[0] = next_buffer_index

      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          h, i, buffer_index
      )

      k = async_copy_k.wait_and_get_loaded()
      v = async_copy_v.wait_and_get_loaded()

      mask_shape = (chunk_size, chunk_size)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      row_ids += q_chunk_idx * chunk_size
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      col_ids += i * chunk_size
      causal_mask = col_ids <= row_ids
      causal_mask_value = jnp.where(causal_mask, 0.0, DEFAULT_MASK_VALUE)

      def per_group_body(group_idx, _):
        q = q_ref[group_idx]
        s = (
            jnp.einsum("td,sd->ts", q, k, preferred_element_type=jnp.float32)
            + causal_mask_value
        )
        # mask.
        s_max = jnp.max(s, axis=1, keepdims=True)

        prev_m = m_ref[group_idx]
        prev_l = l_ref[group_idx]

        cur_m = jnp.maximum(prev_m, s_max)
        cur_m_to_attn_size = jax.lax.broadcast_in_dim(
            cur_m, (chunk_size, chunk_size), (0, 1)
        )

        p = jnp.exp(s - cur_m_to_attn_size)

        cur_l = jnp.exp(prev_m - cur_m) * prev_l + jnp.sum(
            p, axis=1, keepdims=True
        )

        out = out_ref[group_idx]

        out_ref[group_idx, :, :] = (
            out
            * jax.lax.broadcast_in_dim(
                jnp.exp(prev_m - cur_m), (chunk_size, head_dim), (0, 1)
            )
            + p @ v
        ).astype(
            out_ref.dtype
        )  # p @ v  "ts,sd->td"

        m_ref[group_idx, :, :] = cur_m
        l_ref[group_idx, :, :] = cur_l
        return ()

      jax.lax.fori_loop(0, group_size, per_group_body, ())

    @pl.when(((i + 1) * chunk_size) >= length)
    def rescale():
      l = jax.lax.broadcast_in_dim(
          l_ref[...], (group_size, chunk_size, head_dim), (0, 1, 2)
      )
      out_ref[...] = (out_ref[...] / l).astype(out_ref.dtype)

    return ()

  # loop over k, v cache chunk.
  jax.lax.fori_loop(
      0, lax.div(length + chunk_size - 1, chunk_size), per_kv_chunk_body, ()
  )


# TODO: Change to firstly attend to the current chunk kv
# and then write to the KV Cache storage to avoid redundant
# KV Cache reading.
def chunked_prefill_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    length: jax.Array,
    page_indices: jax.Array,
):
  """TPU chunked prefill attention."""
  chunk_size, num_attn_heads, head_dim = q.shape
  num_kv_heads, _, page_size, _ = k_pages.shape

  assert num_attn_heads % num_kv_heads == 0
  assert chunk_size % page_size == 0
  attn_group_size = num_attn_heads // num_kv_heads
  page_per_chunk = chunk_size // page_size

  # q shape as (num_attn_heads, chunk_size, head_dim)
  q = q.transpose((1, 0, 2))
  q = q / jnp.sqrt(head_dim)

  q_block_spec = pl.BlockSpec(
      (attn_group_size, chunk_size, head_dim), lambda i, *_: (i, 0, 0)
  )
  lm_block_spec = pl.BlockSpec(
      (attn_group_size, chunk_size, 1), lambda *_: (0, 0, 0)
  )
  lm_shape = jax.ShapeDtypeStruct(
      shape=(attn_group_size, chunk_size, 1), dtype=jnp.float32
  )
  # loop over Q chunk and num kv heads dimension.
  out, _, _ = pl.pallas_call(
      chunked_prefill_attention_impl,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              q_block_spec,
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
          ],
          out_specs=[
              q_block_spec,
              lm_block_spec,
              lm_block_spec,
          ],
          scratch_shapes=[
              pltpu.VMEM(
                  (
                      2,  # For double buffering during DMA copies.
                      page_per_chunk,
                      page_size,
                      head_dim,
                  ),
                  k_pages.dtype,
              ),  # k_pages buffer
              pltpu.VMEM(
                  (
                      2,  # For double buffering during DMA copies.
                      page_per_chunk,
                      page_size,
                      head_dim,
                  ),
                  v_pages.dtype,
              ),  # v_pages buffer
              pltpu.SemaphoreType.DMA,
          ],
          grid=(num_kv_heads,),
      ),
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q.dtype),
          lm_shape,
          lm_shape,
      ],
      # interpret=True,
      # debug=True
  )(
      jnp.reshape(length, (1,)),
      page_indices,
      jnp.asarray([0], jnp.int32),
      q,
      k_pages,
      v_pages,
  )
  out = out.transpose((1, 0, 2)).reshape(chunk_size, -1).astype(q.dtype)

  return out
