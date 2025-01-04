"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless reuired by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Linear Module

Consider break down the Linear layer module by the sharding strategy for
better readability.
"""

import logging
from typing import Sequence
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from inference.nn import Module, Parameter
from inference import kernel
from inference import parallel
from inference.parallel import LinearParallelConfig, LinearParallelType


class Linear(Module):

  def __init__(
      self,
      in_features: int,
      out_features: int | Sequence[int],
      parallel_config: LinearParallelConfig,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.parallel_config = parallel_config

    self._num_merged = (
        len(out_features) if isinstance(out_features, Sequence) else 1
    )
    out_features_sum = (
        sum(out_features) if self._num_merged > 1 else out_features
    )

    self.weight = Parameter(
        value=jnp.zeros((in_features, out_features_sum), dtype=jnp.bfloat16)
    )

    axis_names = parallel.tp_axis_names()
    if parallel_config.parallel_type == LinearParallelType.COLUMN:
      weight_pspec = P(None, axis_names)
    elif parallel_config.parallel_type == LinearParallelType.ROW:
      weight_pspec = P(axis_names, None)
    else:
      weight_pspec = P(None, None)

    self.weight.sharding = NamedSharding(parallel_config.mesh, weight_pspec)

    collective_matmul_type = parallel_config.collective_matmul_type
    if collective_matmul_type:
      self._collective_matmul = kernel.build_collective_matmul(
          collective_matmul_type,
          axis_names,
      )

  def __call__(self, input):
    if (
        self.parallel_config.collective_matmul_type != None
        and self._collective_matmul
    ):
      output = self._collective_matmul(input, self.weight.value)
      if self._num_merged > 1:
        return jnp.split(output, self._num_merged, 1)
      return output

    preferred_type = input.dtype

    output = jnp.matmul(
        input, self.weight.value, preferred_element_type=preferred_type
    )

    output = output.astype(input.dtype)

    parallel_config = self.parallel_config
    axis_names = parallel.tp_axis_names()
    if parallel_config.reduce_output:
      output = parallel.ops.all_reduce(output, axis_names)
    elif parallel_config.reduce_scatter_output:
      output = jax.lax.psum_scatter(
          output, axis_names, scatter_dimension=1, tiled=True
      )

    if self._num_merged > 1:
      return jnp.split(output, self._num_merged, 1)

    return output

  def load_weights_dict(self, weights_dict):
    res = {}
    for k, v in weights_dict.items():
      attr = getattr(self, k)
      if isinstance(attr, Parameter):
        param = self._parameters[k]
        if v.shape != param.shape:
          logging.warning(
              f"Not matched shape"
              + f": defined {param.shape},"
              + f"loaded {v.shape} for module {self.__class__}"
          )
        param.value = v
        if isinstance(param.sharding, NamedSharding):
          param.to_device()

        if (
            self.parallel_config.collective_matmul_type
            == parallel.CollectiveMatmulType.ALL_GATHER
        ):
          param.value = kernel.prepare_rhs_for_all_gather_collective_matmul(
              param.value, self.parallel_config.mesh
          )
        elif (
            self.parallel_config.collective_matmul_type
            == parallel.CollectiveMatmulType.REDUCE_SCATTER
        ):
          param.value = kernel.prepare_rhs_for_collective_matmul_reduce_scatter(
              param.value, self.parallel_config.mesh
          )
        res[k] = param.value
      else:
        logging.warning(
            f"Unknown checkpoint key {k} for module {self.__class__}"
        )

    return res
