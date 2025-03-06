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

"""Basic Collective Operations."""

import jax
from jax import numpy as jnp
from .mesh import get_num_partitions, get_partition_index


def reduce_scatter(operand, scatter_dimension, axis_names):
  """reduce-scatter sum operation via ppermute."""
  idx = get_partition_index(axis_names=axis_names)
  num_partitions = get_num_partitions(axis_names=axis_names)
  chunk_size = operand.shape[scatter_dimension] // num_partitions
  half_chunk_size = chunk_size // 2
  half_accum_shape = (
      operand.shape[:scatter_dimension]
      + (half_chunk_size,)
      + operand.shape[scatter_dimension + 1 :]
  )

  def step(i, carry):
    accum_fwd, accum_bwd, p_fwd_res, p_bwd_res = carry
    accum_fwd += p_fwd_res
    accum_bwd += p_bwd_res

    fwd_idx = ((idx - i - 1) % num_partitions) * chunk_size
    bwd_idx = ((idx + i + 1) % num_partitions) * chunk_size + half_chunk_size
    p_fwd_res = jax.lax.dynamic_slice_in_dim(
        operand, fwd_idx, half_chunk_size, scatter_dimension
    )
    p_bwd_res = jax.lax.dynamic_slice_in_dim(
        operand, bwd_idx, half_chunk_size, scatter_dimension
    )

    accum_fwd = jax.lax.ppermute(
        accum_fwd,
        axis_name=axis_names,
        perm=[(j, (j + 1) % num_partitions) for j in range(num_partitions)],
    )
    accum_bwd = jax.lax.ppermute(
        accum_bwd,
        axis_name=axis_names,
        perm=[(j, (j - 1) % num_partitions) for j in range(num_partitions)],
    )
    return accum_fwd, accum_bwd, p_fwd_res, p_bwd_res

  accum_fwd = jnp.zeros(half_accum_shape, dtype=operand.dtype)
  accum_bwd = jnp.zeros(half_accum_shape, dtype=operand.dtype)
  initial_fwd_idx = ((idx - 1) % num_partitions) * chunk_size
  initial_bwd_idx = ((idx + 1) % num_partitions) * chunk_size + half_chunk_size
  p_fwd_res = jax.lax.dynamic_slice_in_dim(
      operand, initial_fwd_idx, half_chunk_size, scatter_dimension
  )
  p_bwd_res = jax.lax.dynamic_slice_in_dim(
      operand, initial_bwd_idx, half_chunk_size, scatter_dimension
  )

  accum_fwd, accum_bwd, p_fwd_res, p_bwd_res = jax.lax.fori_loop(
      1, num_partitions, step, (accum_fwd, accum_bwd, p_fwd_res, p_bwd_res)
  )

  return jnp.concatenate(
      (p_fwd_res, p_bwd_res), scatter_dimension
  ) + jnp.concatenate((accum_fwd, accum_bwd), scatter_dimension)


def all_reduce(operand, axis_names):
  """all-reduce sum operation"""
  return jax.lax.psum(operand, axis_name=axis_names)


def all_gather(operand, axis, axis_names):
  """all-gather operation"""
  return jax.lax.all_gather(
      operand, axis=axis, axis_name=axis_names, tiled=True
  )
