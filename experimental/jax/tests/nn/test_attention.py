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
import jax.experimental
import jax.experimental.mesh_utils
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from inference import nn
from inference import parallel

class AttentionTest(absltest.TestCase):
    def test_kv_cache_sharding(self):
        axis = parallel.tp_axis_names()
        k_hbm = jnp.zeros((4, 8, 4, 128))
        v_hbm = jnp.copy(k_hbm)
        num_layer = 3

        mesh = parallel.create_device_mesh(jax.devices(), shape = (4,))
        kv_cache = [nn.KVCache(
            k=k_hbm,
            v=v_hbm,
        ) for _ in range(num_layer)]
        kv_cache_sharding = [nn.KVCache(
            k=NamedSharding(mesh, P(axis, None, None, None)),
            v=NamedSharding(mesh, P(axis, None, None, None)),
        ) for _ in range(num_layer)]

        def sharding(a, b):
            res = jax.device_put(a, b)
            return res

        kv_cache = jax.tree.map(sharding, kv_cache, kv_cache_sharding)
        for i in range(num_layer):
            self.assertIsInstance(kv_cache[i].k.sharding, NamedSharding)
            self.assertIsInstance(kv_cache[i].v.sharding, NamedSharding)

    def test_prefill(self):
        """Only available on TPU."""
        num_attn_heads = 8
        num_kv_heads = 4
        head_dim = 8
        total_page_num = 16
        page_size = 4
        attn_layer = nn.AttentionOps(num_attn_heads, num_kv_heads, head_dim)
        k_hbm, v_hbm = (
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim))
        )

        kv_cache = nn.KVCache(k=k_hbm ,v=v_hbm)

        prefill_len = 8
        prefill_non_padding_len = 7
        q = jnp.ones(((num_attn_heads, prefill_len, head_dim)))
        k_to_save, v_to_save = (
            jnp.ones((num_kv_heads, prefill_len, head_dim)),
            jnp.ones((num_kv_heads, prefill_len, head_dim))
        )
        num_page_to_use = prefill_len // page_size
        # using the second and third page to save the kv cache.
        page_table = jnp.array([1, 3, 0, 0, 0, 0])
        output, kv_cache = attn_layer._prefill(
            q,
            k_to_save,
            v_to_save,
            kv_cache,
            nn.AttentionMetadata(
                prefill_length=prefill_non_padding_len,
                prefill_pos=jnp.arange(0, 8),
                prefill_page_table=page_table,
                generate_pos=0,
                generate_page_table=0
            ),
        )
        kv_cache = attn_layer._write_prefill_kv_to_kv_cache(
            k_to_save,
            v_to_save,
            kv_cache,
            prefill_non_padding_len,
            page_table
        )
        np.testing.assert_allclose(
            kv_cache.k[:, page_table[:2], :, :],
            jnp.ones((num_kv_heads, num_page_to_use , page_size, head_dim))
        )
        np.testing.assert_allclose(
            kv_cache.v[:, page_table[:2], :, :],
            jnp.ones((num_kv_heads, num_page_to_use, page_size, head_dim))
        )
        zero_index = [i for i in range(page_size)]
        zero_index = (
            zero_index[0:page_table[0]] +
            zero_index[page_table[0]+1:page_table[1]] +
            zero_index[page_table[1]+1:]
        )

        np.testing.assert_allclose(
            kv_cache.k[:, zero_index, :, :],
            jnp.zeros((num_kv_heads, page_size-num_page_to_use , page_size, head_dim))
        )
        np.testing.assert_allclose(
            kv_cache.v[:, zero_index ,:,:],
            jnp.zeros((num_kv_heads, page_size-num_page_to_use , page_size, head_dim))
        )
        np.testing.assert_equal(output.shape, (prefill_len, num_attn_heads * head_dim))

    def test_generate_cache_update(self):
        num_attn_heads = 8
        num_kv_heads = 4
        head_dim = 8
        total_page_num = 128
        page_size = 4
        max_len = 32
        attn_layer = nn.AttentionOps(num_attn_heads, num_kv_heads, head_dim)
        k_hbm, v_hbm = (
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
        )

        kv_cache = nn.KVCache(k=k_hbm ,v=v_hbm)

        generate_len = 4
        k_to_save, v_to_save = (
            jnp.ones((num_kv_heads, generate_len, head_dim)),
            jnp.ones((num_kv_heads, generate_len, head_dim))
        )
        prng = jax.random.PRNGKey(99)
        page_table = jnp.asarray(np.random.choice(total_page_num, (generate_len, max_len//page_size), replace=False))
        page_pos = jax.random.randint(prng, shape=(generate_len,),
                                      minval=0, maxval=total_page_num * page_size)
        kv_cache = attn_layer._write_generate_kv_to_kv_cache(
            k_to_save,
            v_to_save,
            kv_cache,
            page_pos,
            page_table
        )

        page_idx, offset = jnp.divmod(page_pos, page_size)
        page_to_update = page_table[jnp.arange(0, generate_len), page_idx]

        np.testing.assert_allclose(
            kv_cache.k[:, page_to_update, offset,:],
            jnp.ones_like(kv_cache.k[:, page_to_update , offset,:]),
        )

        np.testing.assert_allclose(
            kv_cache.v[:, page_to_update, offset,:],
            jnp.ones_like(kv_cache.v[:, page_to_update , offset,:]),
        )

        np.testing.assert_allclose(jnp.sum(kv_cache.k), generate_len * num_kv_heads * head_dim)

    def test_generate(self):
        """Only available on TPU."""
        num_attn_heads = 8
        num_kv_heads = 4
        head_dim = 8
        total_page_num = 128
        page_size = 4
        max_len = 32
        attn_layer = nn.AttentionOps(num_attn_heads, num_kv_heads, head_dim)
        k_hbm, v_hbm = (
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
            jnp.zeros((num_kv_heads, total_page_num, page_size, head_dim)),
        )

        kv_cache = nn.KVCache(k=k_hbm ,v=v_hbm)

        num_generate_tokens = 4
        q = jnp.ones((num_attn_heads, num_generate_tokens, head_dim))
        k_to_save, v_to_save = (
            jnp.ones((num_kv_heads, num_generate_tokens, head_dim)),
            jnp.ones((num_kv_heads, num_generate_tokens, head_dim)),
        )
        prng = jax.random.PRNGKey(99)
        page_table = jnp.asarray(np.random.choice(total_page_num, (num_generate_tokens, max_len//page_size), replace=False))
        page_pos = jax.random.randint(prng, shape=(num_generate_tokens,),
                                      minval=0, maxval=total_page_num * page_size)

        page_table = page_table
        page_pos = page_pos
        output, kv_cache = attn_layer._generate(
            q,
            k_to_save,
            v_to_save,
            kv_cache,
            nn.AttentionMetadata(
                prefill_length=0,
                prefill_pos=0,
                prefill_page_table=0,
                generate_pos=page_pos,
                generate_page_table=page_table,
            ),
        )

        page_idx, offset = jnp.divmod(page_pos, page_size)
        page_to_update = page_table[jnp.arange(0, num_generate_tokens), page_idx]

        np.testing.assert_allclose(
            kv_cache.k[:, page_to_update, offset,:],
            jnp.ones_like(kv_cache.k[:, page_to_update , offset,:]),
        )

        np.testing.assert_allclose(
            kv_cache.v[:, page_to_update, offset,:],
            jnp.ones_like(kv_cache.v[:, page_to_update , offset,:]),
        )

        np.testing.assert_allclose(jnp.sum(kv_cache.k), num_generate_tokens * num_kv_heads * head_dim)
        np.testing.assert_equal(output.shape, (num_generate_tokens, num_attn_heads * head_dim))