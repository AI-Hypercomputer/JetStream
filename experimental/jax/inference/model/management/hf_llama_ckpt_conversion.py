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

"""Hugging Face Llama2 ckpt conversion utility function."""

from jax import numpy as jnp
from transformers import LlamaConfig
from .util import convert_to_jax_array_on_cpu


def merge_gate_up_proj_weights(gate, up, num_devices):
  col_chunk_size = gate.shape[1] // num_devices
  weight = jnp.concatenate(
      [gate[:, :col_chunk_size], up[:, :col_chunk_size]], axis=-1
  )
  for i in range(1, num_devices):
    index = col_chunk_size * i
    weight = jnp.concatenate(
        [
            weight,
            gate[:, index : index + col_chunk_size],
            up[:, index : index + col_chunk_size],
        ],
        axis=-1,
    )

  return weight


def convert_hf_llama(torch_weight_state, num_devices, config: LlamaConfig):
  jax_weight_state = {
      "embed_tokens": {
          "weight": convert_to_jax_array_on_cpu(
              torch_weight_state["model.embed_tokens.weight"]
          ),
      },
      "layers": {},
      "norm": {
          "weight": convert_to_jax_array_on_cpu(
              torch_weight_state["model.norm.weight"]
          ),
      },
      "lm_head": {
          "weight": convert_to_jax_array_on_cpu(
              torch_weight_state["lm_head.weight"].T
          ),
      },
  }
  del torch_weight_state["model.embed_tokens.weight"]
  del torch_weight_state["model.norm.weight"]
  del torch_weight_state["lm_head.weight"]
  for i in range(config.num_hidden_layers):
    gate_up_proj_weight = merge_gate_up_proj_weights(
        gate=convert_to_jax_array_on_cpu(
            torch_weight_state[f"model.layers.{i}.mlp.gate_proj.weight"].T
        ),
        up=convert_to_jax_array_on_cpu(
            torch_weight_state[f"model.layers.{i}.mlp.up_proj.weight"].T
        ),
        num_devices=num_devices,
    )
    del torch_weight_state[f"model.layers.{i}.mlp.gate_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.mlp.up_proj.weight"]
    jax_weight_state["layers"][i] = {
        "self_attn": {
            "q_proj": {
                "weight": convert_to_jax_array_on_cpu(
                    torch_weight_state[
                        f"model.layers.{i}.self_attn.q_proj.weight"
                    ].T
                ),
            },
            "k_proj": {
                "weight": convert_to_jax_array_on_cpu(
                    torch_weight_state[
                        f"model.layers.{i}.self_attn.k_proj.weight"
                    ].T
                ),
            },
            "v_proj": {
                "weight": convert_to_jax_array_on_cpu(
                    torch_weight_state[
                        f"model.layers.{i}.self_attn.v_proj.weight"
                    ].T
                ),
            },
            "o_proj": {
                "weight": convert_to_jax_array_on_cpu(
                    torch_weight_state[
                        f"model.layers.{i}.self_attn.o_proj.weight"
                    ].T
                ),
            },
        },
        "ffw": {
            "gate_up_proj": {
                "weight": gate_up_proj_weight,
            },
            "down_proj": {
                "weight": convert_to_jax_array_on_cpu(
                    torch_weight_state[
                        f"model.layers.{i}.mlp.down_proj.weight"
                    ].T
                ),
            },
        },
        "input_layernorm": {
            "weight": convert_to_jax_array_on_cpu(
                torch_weight_state[f"model.layers.{i}.input_layernorm.weight"]
            ),
        },
        "post_attention_layernorm": {
            "weight": convert_to_jax_array_on_cpu(
                torch_weight_state[
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ]
            ),
        },
    }
    del torch_weight_state[f"model.layers.{i}.self_attn.q_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.self_attn.k_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.self_attn.v_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.self_attn.o_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.mlp.down_proj.weight"]
    del torch_weight_state[f"model.layers.{i}.input_layernorm.weight"]
    del torch_weight_state[f"model.layers.{i}.post_attention_layernorm.weight"]

  del torch_weight_state

  jax_causal_lm_weight_state = {}
  jax_causal_lm_weight_state["model"] = jax_weight_state
  return jax_causal_lm_weight_state
