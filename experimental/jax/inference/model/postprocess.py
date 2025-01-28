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

"""Post-process utilities"""

import jax
from jax import numpy as jnp
import dataclasses
from jax.sharding import NamedSharding
from inference.nn import AttentionMetadata
from inference.utils import register_flat_dataclass_as_pytree


@register_flat_dataclass_as_pytree
@dataclasses.dataclass
class ModelOutput:
  prefill_token: jax.Array | NamedSharding
  prefill_done: jax.Array | NamedSharding
  prefill_next_pos: jax.Array | NamedSharding
  generate_tokens: jax.Array | NamedSharding
  generate_done: jax.Array | NamedSharding
  generate_next_pos: jax.Array | NamedSharding


def postprocess(
    tokens: jax.Array, done: jax.Array, attn_meta: AttentionMetadata
) -> ModelOutput:
  dummy_scalar = jnp.asarray(-1, dtype=jnp.int32)
  dummy_vec = jnp.asarray([-1], dtype=jnp.int32)
  output = ModelOutput(
      prefill_token=dummy_scalar,
      prefill_done=dummy_scalar,
      prefill_next_pos=dummy_scalar,
      generate_tokens=dummy_vec,
      generate_done=dummy_vec,
      generate_next_pos=dummy_vec,
  )

  has_prefill = False
  has_generate = False
  if len(attn_meta.prefill_pos.shape) != 0:
    has_prefill = True
  if len(attn_meta.generate_pos.shape) != 0:
    has_generate = True

  if has_prefill and not has_generate:
    output.prefill_token = tokens[0]
    output.prefill_done = done[0]
    output.prefill_next_pos = attn_meta.prefill_length

  if not has_prefill and has_generate:
    output.generate_tokens = tokens
    output.generate_done = done
    output.generate_next_pos = jnp.where(
        output.generate_done, -1, attn_meta.generate_pos + 1
    )
    output.generate_next_pos = jnp.where(
        output.generate_next_pos, output.generate_next_pos, -1
    )

  if has_prefill and has_generate:
    output.prefill_token = tokens[0]
    output.prefill_done = done[0]
    output.prefill_next_pos = attn_meta.prefill_length

    output.generate_tokens = tokens[1:]
    output.generate_done = done[1:]
    output.generate_next_pos = jnp.where(
        output.generate_done, -1, attn_meta.generate_pos + 1
    )
    output.generate_next_pos = jnp.where(
        output.generate_next_pos, output.generate_next_pos, -1
    )

  return output
