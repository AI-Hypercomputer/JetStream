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
from inference.config.config import ModelId
from inference.model import LlamaModel, ModelRegistry
from inference import parallel
from inference import nn


class LlamaModelTest(absltest.TestCase):

  def _create_device_mesh(self):
    devices = jax.devices()
    return parallel.create_device_mesh(
        devices=devices,
        shape=(len(devices), 1),
    )

  def test_llama(self):
    # TODO: make it as an accuracy test.
    mesh = self._create_device_mesh()
    model_registry = ModelRegistry()
    model_id = ModelId.llama_2_7b_chat_hf
    config = model_registry.load_model_config(model_id)
    config.num_hidden_layers = 1
    tokenizer = model_registry.load_tokenizer(model_id)

    input_ids = tokenizer.encode("I have a dog that is", return_tensors="pt")
    prompt_len = input_ids.shape[1]
    hg_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, config=config
    )

    outputs = hg_model(input_ids)
    expected_logits = outputs.logits.detach().numpy()[0]

    num_prefill_tokens = 16
    tokens = jnp.asarray(input_ids)[0]
    tokens = jax.lax.dynamic_update_index_in_dim(
        jnp.zeros((num_prefill_tokens), dtype=jnp.int32), tokens, 0, 0
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
            k=NamedSharding(
                mesh, P(parallel.tp_axis_names(), None, None, None)
            ),
            v=NamedSharding(
                mesh, P(parallel.tp_axis_names(), None, None, None)
            ),
        )
        for _ in range(config.num_hidden_layers)
    ]
    kv_caches = jax.device_put(kv_caches, kv_caches_sharding)
    attn_metadata = nn.AttentionMetadata(
        prefill_length=prompt_len,
        prefill_pos=pos,
        prefill_page_table=jnp.asarray([0, 1, 2, 3]),
        generate_pos=jnp.asarray(0),
        generate_page_table=jnp.asarray(0),
    )

    attention_metadata_sharding = nn.AttentionMetadata(
        prefill_length=NamedSharding(mesh, P()),
        prefill_pos=NamedSharding(mesh, P(None)),
        prefill_page_table=NamedSharding(mesh, P(None)),
        generate_pos=NamedSharding(mesh, P()),
        generate_page_table=NamedSharding(mesh, P()),
    )
    attn_metadata = jax.device_put(attn_metadata, attention_metadata_sharding)

    casual_lm_weight_cpu = model_registry.load_weights_to_host(
        model_id=model_id,
        num_devices=np.prod(mesh.devices.shape),
        model_config=config,
        dtype=jnp.float32,
    )
    model = LlamaModel(config, parallel.ModelParallelConfig(mesh=mesh))

    weight_dict = model.load_weights_dict(casual_lm_weight_cpu["model"])
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
        f=model.jittable_call,
        mesh=mesh,
        in_specs=(
            weight_dict_pspec,
            P(None),
            P(None),
            kv_caches_pspec,
            attn_meta_pspec,
        ),
        out_specs=(
            P(None, None),
            kv_caches_pspec,
        ),
        check_rep=False,
    )

    executable = (
        jax.jit(infer_func, donate_argnums=(3,))
        .lower(
            weight_dict,
            jnp.asarray(tokens),
            pos,
            kv_caches,
            attn_metadata,
        )
        .compile()
    )
    got_logits, _ = executable(
        weight_dict,
        jnp.asarray(tokens),
        pos,
        kv_caches,
        attn_metadata,
    )

    got_logits = got_logits[:prompt_len]
    np.testing.assert_allclose(
        got_logits, expected_logits, atol=3e-02, rtol=1e-02
    )


if __name__ == "__main__":
  absltest.main()
