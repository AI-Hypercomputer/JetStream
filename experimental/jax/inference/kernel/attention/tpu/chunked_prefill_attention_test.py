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

from absl.testing import absltest
import numpy as np
import math
import jax
from jax import numpy as jnp
from inference import nn
from inference.kernel.attention_ops import vanilla_prefill_gqa
from inference.kernel.attention.tpu.chunked_prefill_attention import chunked_prefill_attention


class ChunkedPrefillTest(absltest.TestCase):

  def test(self):
    num_attn_heads = 4
    num_kv_heads = 2
    head_dim = 128
    total_page_num = 16
    page_size = 16

    rng = jax.random.key(0)

    attn_layer = nn.AttentionOps(num_attn_heads, num_kv_heads, head_dim)

    kv_cache = nn.KVCache(
        k=jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
        v=jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
    )

    prefill_len = 6 * page_size
    prefill_non_padding_len = 4 * page_size + 3

    q = jax.random.uniform(rng, (prefill_len, num_attn_heads, head_dim))

    rng_1, rng_2 = jax.random.split(rng)
    k_to_save, v_to_save = (
        jax.random.uniform(rng_1, (prefill_len, num_kv_heads, head_dim)),
        jax.random.uniform(rng_2, (prefill_len, num_kv_heads, head_dim)),
    )

    num_pages_with_padding = math.ceil(prefill_len / page_size)
    page_table = jnp.array(([i for i in range(num_pages_with_padding)]))
    kv_cache = attn_layer._write_prefill_kv_to_kv_cache(
        k_to_save,
        v_to_save,
        kv_cache,
        prefill_non_padding_len,
        page_table,
    )

    chunk_size = 2 * page_size

    expected_output = vanilla_prefill_gqa(q, k_to_save, v_to_save)

    num_active_pages_per_prefill = math.ceil(
        prefill_non_padding_len / page_size
    )
    compute_times = math.ceil(prefill_non_padding_len / chunk_size)

    for i in range(compute_times):
      idx = i * chunk_size
      length = idx + chunk_size
      length = min(length, prefill_non_padding_len)
      chunk_output = chunked_prefill_attention(
          q[idx : idx + chunk_size], kv_cache.k, kv_cache.v, length, page_table
      )

      if i < compute_times - 1:
        np.testing.assert_allclose(
            np.array(chunk_output),
            np.array(expected_output[idx : idx + chunk_size]),
            rtol=4e-03,
            atol=1e-03,
        )
      if i == compute_times - 1:
        offset = prefill_non_padding_len % chunk_size
        np.testing.assert_allclose(
            np.array(chunk_output[:offset]),
            np.array(expected_output[idx : idx + offset]),
            rtol=4e-03,
            atol=1e-03,
        )

    num_pages_per_prefill = prefill_len // page_size
    for i in range(num_active_pages_per_prefill):
      idx = i * page_size
      np.testing.assert_equal(
          np.array(kv_cache.k[:, page_table[i], :, :][:, :, :]),
          np.array(k_to_save[idx : idx + page_size, :, :].transpose(1, 0, 2)),
      )
      np.testing.assert_equal(
          np.array(kv_cache.v[:, page_table[i], :, :][:, :, :]),
          np.array(v_to_save[idx : idx + page_size, :, :].transpose(1, 0, 2)),
      )

    for i in range(num_active_pages_per_prefill, num_pages_per_prefill):
      idx = i * page_size
      np.testing.assert_equal(
          np.array(kv_cache.v[:, page_table[i], :, :][:, None, :, :]),
          np.array(jnp.zeros((num_kv_heads, 1, page_size, head_dim))),
      )

      np.testing.assert_equal(
          np.array(kv_cache.v[:, page_table[i], :, :][:, None, :, :]),
          np.array(jnp.zeros((num_kv_heads, 1, page_size, head_dim))),
      )


if __name__ == "__main__":
  absltest.main()
