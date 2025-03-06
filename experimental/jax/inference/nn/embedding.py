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

"""Embedding Module"""

from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from inference.nn import Module, Parameter
from inference.parallel import EmbeddingParallelType
from inference import parallel


class Embedding(Module):

  def __init__(
      self,
      vocab_size,
      embedding_dim,
      parallel_config: parallel.EmbeddingParallelConfig,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.parallel_config = parallel_config

    self.weight = Parameter(
        jnp.zeros((vocab_size, embedding_dim), dtype=jnp.bfloat16)
    )

    mesh = parallel_config.mesh
    if parallel_config.parallel_type == EmbeddingParallelType.COLUMN:
      weight_pspec = P(None, parallel.tp_axis_names())
    else:
      weight_pspec = P(None, None)

    self.weight.sharding = NamedSharding(mesh, weight_pspec)

  def __call__(self, input):
    return self.weight.value[input]


def apply_rope_embedding(input, position, theta=10000):
  emb_dim = input.shape[-1]
  fraction = jnp.arange(0, emb_dim, 2) / emb_dim
  timescale = theta**fraction
  position = position[:, None, None]
  sinusoid_inp = position / timescale
  sin = jnp.sin(sinusoid_inp).astype(input.dtype)
  cos = jnp.cos(sinusoid_inp).astype(input.dtype)
  first_half, second_half = jnp.split(input, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  return jnp.concatenate((first_part, second_part), axis=-1)
