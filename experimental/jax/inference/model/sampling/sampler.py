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

import dataclasses
import jax
from jax import numpy as jnp
from inference.nn import AttentionMetadata
from inference.utils import register_flat_dataclass_as_pytree


@register_flat_dataclass_as_pytree
@dataclasses.dataclass
class SamplingParams:
  temperature: jax.Array
  top_k: jax.Array
  rng: jax.Array


class Sampler:

  def __init__(
      self,
      eos: int | None = None,
      max_length: int | None = None,
  ) -> None:
    self.eos = eos
    self.max_length = max_length

  def sample(
      self,
      logits: jax.Array,  # [num_tokens, vocab_size]
      positions: jax.Array,
      attn_metadata: AttentionMetadata,
      sampling_params: SamplingParams,
  ) -> jax.Array:
    sampling_rng = sampling_params.rng
    probabilities = jax.nn.softmax(
        logits / sampling_params.temperature, axis=-1
    )
    top_k_prob, top_k_indices = jax.lax.top_k(probabilities, 1)
    selected_index = jax.random.categorical(sampling_rng, top_k_prob)

    tokens = top_k_indices[
        jnp.arange(0, top_k_indices.shape[0]), selected_index
    ]
    done = jnp.equal(tokens, self.eos)
    done = jnp.logical_or(
        done, jnp.greater_equal(positions, self.max_length - 1)
    )

    if len(attn_metadata.prefill_pos.shape) == 0:
      padded_prefill_len = 0
    else:
      padded_prefill_len = attn_metadata.prefill_pos.shape[0]

    if len(attn_metadata.generate_pos.shape) != 0 and padded_prefill_len != 0:
      prefill_token = tokens.at[attn_metadata.prefill_length - 1].get()[None]
      generate_tokens = tokens.at[padded_prefill_len:].get()

      prefill_done = done.at[attn_metadata.prefill_length - 1].get()[None]
      generate_done = done.at[padded_prefill_len:].get()

      tokens = jnp.concatenate((prefill_token, generate_tokens))
      done = jnp.concatenate((prefill_done, generate_done))
    elif padded_prefill_len != 0:
      tokens = tokens.at[attn_metadata.prefill_length - 1].get()[None]
      done = done.at[attn_metadata.prefill_length - 1].get()[None]

    return tokens, done
