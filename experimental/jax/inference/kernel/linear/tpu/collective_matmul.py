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

""" collective matmul for linear layer (1d ring topology).

Reference:
https://dl.acm.org/doi/pdf/10.1145/3567955.3567959

The implementation is throughput-bound. Latency-bound implementation
should also be added.

The implementation is via the JAX/XLA collective-permute API, and it doesn't
work for GPU well: https://github.com/openxla/xla/issues/10640#issuecomment-2246448416
"""

import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map
from inference.parallel import get_num_partitions, get_partition_index, tp_axis_names


def prepare_rhs_for_all_gather_collective_matmul(rhs: jax.Array, mesh: Mesh):
  """Prepare rhs for all gather collective matmul.

  For bidirectional collective matmul, the two lhs chunks
  received from the neighbor are not contiguous in the
  all-gathered lhs. Reshuffle the rhs ahead to process
  the non-contiguous lhs chunks together in the collective
  matmul.
  """
  axis_names = tp_axis_names()

  def reshuffle(rhs: jax.Array):
    idx = get_partition_index(axis_names=axis_names)
    num_partitions = get_num_partitions(axis_names=axis_names)
    rhs_chunk_row_size = rhs.shape[0] // num_partitions
    half_rhs_chunk_row_size = rhs_chunk_row_size // 2

    def swap(i, carry):
      rhs = carry
      idx_1 = ((idx + i) % num_partitions) * rhs_chunk_row_size
      idx_2 = ((idx - i) % num_partitions) * rhs_chunk_row_size
      operand_1 = jax.lax.dynamic_slice_in_dim(
          rhs, idx_1, half_rhs_chunk_row_size, axis=0
      )
      operand_2 = jax.lax.dynamic_slice_in_dim(
          rhs, idx_2, half_rhs_chunk_row_size, axis=0
      )
      rhs = jax.lax.dynamic_update_slice_in_dim(rhs, operand_1, idx_2, axis=0)
      rhs = jax.lax.dynamic_update_slice_in_dim(rhs, operand_2, idx_1, axis=0)
      return rhs

    rhs = jax.lax.fori_loop(1, num_partitions // 2, swap, rhs)
    return rhs

  return shard_map(
      f=reshuffle,
      mesh=mesh,
      in_specs=rhs.sharding.spec,
      out_specs=rhs.sharding.spec,
  )(rhs)


def all_gather_collective_matmul(lhs, rhs, axis_names):
  """All gather collective matmul.

  The function works for matmul where the lhs is partitioned at
  the contracting dimension.
  """
  idx = get_partition_index(axis_names=axis_names)
  num_partitions = get_num_partitions(axis_names=axis_names)
  rhs_chunk_row_size = rhs.shape[0] // num_partitions

  def step(i, carry):
    accum, fwd_lhs, bwd_lhs = carry
    rhs_row_idx = ((idx + i) % num_partitions) * rhs_chunk_row_size
    cur_lhs = jnp.concatenate((fwd_lhs, bwd_lhs), axis=1)
    rhs_chunk = jax.lax.dynamic_slice_in_dim(
        rhs, rhs_row_idx, rhs_chunk_row_size
    )
    output = cur_lhs @ rhs_chunk
    accum += output
    fwd_lhs = jax.lax.ppermute(
        fwd_lhs,
        axis_names,
        [(j, (j + 1) % num_partitions) for j in range(num_partitions)],
    )
    bwd_lhs = jax.lax.ppermute(
        bwd_lhs,
        axis_names,
        [(j, (j - 1) % num_partitions) for j in range(num_partitions)],
    )
    return accum, fwd_lhs, bwd_lhs

  res_shape = (lhs.shape[0], rhs.shape[1])
  accum = jnp.zeros(shape=res_shape, dtype=lhs.dtype)
  fwd_lhs, bwd_lhs = jnp.split(lhs, 2, 1)
  accum, fwd_lhs, bwd_lhs = jax.lax.fori_loop(
      0, num_partitions - 1, step, (accum, fwd_lhs, bwd_lhs)
  )

  # Last round which doesn't need collective permute.
  rhs_row_idx = (
      (idx + num_partitions - 1) % num_partitions
  ) * rhs_chunk_row_size
  cur_lhs = jnp.concatenate((fwd_lhs, bwd_lhs), axis=1)
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, rhs_row_idx, rhs_chunk_row_size)
  output = cur_lhs @ rhs_chunk
  accum += output

  return accum


def prepare_rhs_for_collective_matmul_reduce_scatter(
    rhs: jax.Array, mesh: Mesh
):
  """Prepare rhs for collective matmul with reduce scatter.

  For bidirectional collective matmul, the two accum chunks
  received from the neighbor are not contiguous in the
  final accum. Reshuffle the rhs ahead to process
  the non-contiguous accum chunks together in the collective
  matmul.
  """
  axis_names = tp_axis_names()

  def reshuffle(rhs):
    idx = get_partition_index(axis_names=axis_names)
    num_partitions = get_num_partitions(axis_names=axis_names)
    rhs_chunk_col_size = rhs.shape[1] // num_partitions
    half_rhs_chunk_col_size = rhs_chunk_col_size // 2

    def swap(i, carry):
      rhs = carry
      idx_1 = ((idx + i) % num_partitions) * rhs_chunk_col_size
      idx_2 = ((idx - i) % num_partitions) * rhs_chunk_col_size
      operand_1 = jax.lax.dynamic_slice_in_dim(
          rhs, idx_1, half_rhs_chunk_col_size, axis=1
      )
      operand_2 = jax.lax.dynamic_slice_in_dim(
          rhs, idx_2, half_rhs_chunk_col_size, axis=1
      )
      rhs = jax.lax.dynamic_update_slice_in_dim(rhs, operand_1, idx_2, axis=1)
      rhs = jax.lax.dynamic_update_slice_in_dim(rhs, operand_2, idx_1, axis=1)
      return rhs

    rhs = jax.lax.fori_loop(1, num_partitions // 2, swap, rhs)
    return rhs

  pspec = rhs.sharding.spec

  return shard_map(
      f=reshuffle,
      mesh=mesh,
      in_specs=pspec,
      out_specs=pspec,
  )(rhs)


def collective_matmul_reduce_scatter(lhs, rhs, axis_names):
  """Collective matmul with reduce scatter at the output column axis."""
  idx = get_partition_index(axis_names=axis_names)
  num_partitions = get_num_partitions(axis_names=axis_names)
  rhs_chunk_col_size = rhs.shape[1] // num_partitions
  # Compute the partial result for the chip at the last step.
  rhs_col_idx = ((idx + 1) % num_partitions) * rhs_chunk_col_size
  rhs_chunk = jax.lax.dynamic_slice_in_dim(
      rhs,
      start_index=rhs_col_idx,
      slice_size=rhs_chunk_col_size,
      axis=1,
  )
  partial_res = lhs @ rhs_chunk
  accum_to_send_shape = (lhs.shape[0], rhs_chunk_col_size // 2)
  fwd_accum = jnp.zeros(shape=accum_to_send_shape)
  bwd_accum = jnp.zeros(shape=accum_to_send_shape)

  def step(i, carry):
    fwd_accum, bwd_accum, partial_res = carry
    accum = jnp.concatenate((fwd_accum, bwd_accum), axis=1)
    accum += partial_res
    fwd_accum, bwd_accum = jnp.split(accum, 2, axis=1)

    rhs_col_idx = ((idx + 1 + i) % num_partitions) * rhs_chunk_col_size
    rhs_chunk = jax.lax.dynamic_slice_in_dim(
        rhs,
        start_index=rhs_col_idx,
        slice_size=rhs_chunk_col_size,
        axis=1,
    )
    partial_res = lhs @ rhs_chunk
    fwd_accum = jax.lax.ppermute(
        fwd_accum,
        axis_names,
        [(j, (j + 1) % num_partitions) for j in range(num_partitions)],
    )
    bwd_accum = jax.lax.ppermute(
        bwd_accum,
        axis_names,
        [(j, (j - 1) % num_partitions) for j in range(num_partitions)],
    )
    return fwd_accum, bwd_accum, partial_res

  fwd_accum, bwd_accum, partial_res = jax.lax.fori_loop(
      1, num_partitions, step, (fwd_accum, bwd_accum, partial_res)
  )
  accum = jnp.concatenate((fwd_accum, bwd_accum), axis=1)
  accum += partial_res
  return accum
