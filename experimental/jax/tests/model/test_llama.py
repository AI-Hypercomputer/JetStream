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
import jax
from jax.experimental.shard_map import shard_map
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import torch
from transformers import AutoModelForCausalLM
from inference.model import LlamaModel, ModelRegistry
from inference import parallel
from inference import nn

class LlamaModelTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_llama_creation(self):
        # TODO: make it as an accuracy test.
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        mesh = parallel.create_device_mesh(jax.devices(), 4)
        model_registry = ModelRegistry()

        config, tokenizer = model_registry.load_model_config(model_id), model_registry.load_tokenizer(model_id)
        config.num_hidden_layers = 1
        num_prefill_tokens = 16
        input_ids = tokenizer.encode(
            "I have a dog that is",
            return_tensors="pt"
        )
        prompt_len = input_ids.shape[1]
        hg_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                config=config
        )

        outputs = hg_model(input_ids)
        expected_logits = outputs.logits.detach().numpy()[0]

        tokens = jnp.asarray(input_ids)[0]
        tokens = jax.lax.dynamic_update_index_in_dim(
            jnp.zeros((num_prefill_tokens), dtype=jnp.int32),
            tokens,
            0,
            0
        )
        pos = jnp.arange(0, num_prefill_tokens)
        kv_caches = [
            nn.KVCache(
                k=jnp.zeros((config.num_key_value_heads, 32, 16, config.head_dim)),
                v=jnp.zeros((config.num_key_value_heads, 32, 16, config.head_dim)),
            )
            for _ in range(config.num_hidden_layers)
        ]
        kv_caches_sharding = [
            nn.KVCache(
                k=NamedSharding(mesh, P(parallel.tp_axis_names(), None, None, None)),
                v=NamedSharding(mesh, P(parallel.tp_axis_names(), None, None, None)),
            )
            for _ in range(config.num_hidden_layers)
        ]
        kv_caches = jax.device_put(kv_caches, kv_caches_sharding)
        attn_metadata = nn.AttentionMetadata(
            num_prefill_tokens=jnp.asarray(num_prefill_tokens),
            prefill_pos=pos,
            prefill_len=prompt_len,
            prefill_page_table=jnp.asarray([0, 1, 2, 3]),
            num_generate_tokens=jnp.asarray(0),
            generate_pos=jnp.asarray(0),
            generate_len=jnp.asarray(0),
            generate_page_table=jnp.asarray(0),
        )

        attention_metadata_sharding = nn.AttentionMetadata(
                num_prefill_tokens=NamedSharding(mesh, P()),
                prefill_pos=NamedSharding(mesh, P(None)),
                prefill_len=NamedSharding(mesh, P()),
                prefill_page_table=NamedSharding(mesh, P(None)),
                num_generate_tokens=NamedSharding(mesh, P()),
                generate_pos=NamedSharding(mesh, P()),
                generate_len=NamedSharding(mesh, P()),
                generate_page_table=NamedSharding(mesh, P()),
        )
        attn_metadata = jax.device_put(attn_metadata, attention_metadata_sharding)

        casual_lm_weight_cpu = model_registry.load_weight_to_host(
            model_id,
            num_devices=np.prod(mesh.devices.shape),
            dtype=jnp.float32,
            custom_hf_model_config=config
        )
        model = LlamaModel(config, parallel.ModelParallelConfig(mesh=mesh))

        weight_dict = model.load_weight_dict(casual_lm_weight_cpu["model"])
        weight_dict_pspec = jax.tree_util.tree_map(
            lambda a: a.sharding.spec, weight_dict
        )
        kv_caches_pspec = jax.tree_util.tree_map(
            lambda a: a.sharding.spec, kv_caches
        )
        attn_meta_pspec = jax.tree_util.tree_map(
            lambda a: a.spec, attention_metadata_sharding 
        )

        del casual_lm_weight_cpu

        infer_func = shard_map(
            f=model.jittable_inference,
            mesh=mesh,
            in_specs= (
                weight_dict_pspec,
                P(None),
                P(None),
                kv_caches_pspec,
                attn_meta_pspec,
            ),
            out_specs= (
                P(None, None),
                kv_caches_pspec,
            ),
            check_rep=False,
        )

        executable = jax.jit(infer_func, donate_argnums=(3,)).lower(
                weight_dict,
                jnp.asarray(tokens),
                pos,
                kv_caches,
                attn_metadata,
        ).compile()
        got_logits, _ = executable(
            weight_dict,
            jnp.asarray(tokens),
            pos,
            kv_caches,
            attn_metadata,
        )

        got_logits = got_logits[:prompt_len]
        np.testing.assert_allclose(got_logits, expected_logits, atol=1e-05, rtol=1e-05)