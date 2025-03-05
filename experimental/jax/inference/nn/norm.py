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

"""Norm Module"""

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from inference.nn import Module, Parameter
from inference import parallel


class RMSNorm(Module):

  def __init__(self, dim, eps, parallel_config: parallel.RMSNormParallelConfig):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.parallel_config = parallel_config

    self.weight = Parameter(jnp.zeros((dim,)))
    self.variance_epsilon = eps

    mesh = parallel_config.mesh
    if parallel_config.activation_shared:
      self.weight.sharding = NamedSharding(mesh, P(parallel.tp_axis_names()))
    else:
      self.weight.sharding = NamedSharding(mesh, P(None))

  def __call__(self, input):
    input_dtype = input.dtype
    input = input.astype(jnp.float32)
    variance = jnp.mean(jax.lax.square(input), axis=-1, keepdims=True)

    if self.parallel_config.activation_shared:
      axis_names = parallel.tp_axis_names()
      variance = parallel.ops.all_reduce(
          variance, axis_names
      ) / parallel.get_num_partitions(axis_names)

    input = input * jax.lax.rsqrt(variance + self.variance_epsilon)
    return self.weight * input.astype(input_dtype)
