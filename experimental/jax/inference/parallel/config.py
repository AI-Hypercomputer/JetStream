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

import enum
import dataclasses
import jax


# TODO: change to enum.StrEnum after Python upgrading in Google Cloud TPU.
@enum.unique
class ParallelAxis(enum.Enum):
  # X is used for Tensor Parallelism (as major axis) and Expert Parallelism.
  X = enum.auto()
  # Y is used for Sequence, Sequence Pipeline, Decode Batch and Tensor Parallelism (as minor axis).
  Y = enum.auto()


def dp_axis_names():
  return (ParallelAxis.Y.name,)


def sp_axis_names():
  return (ParallelAxis.Y.name,)


def spp_axis_names():
  return (ParallelAxis.Y.name,)


def tp_axis_names():
  return (ParallelAxis.X.name, ParallelAxis.Y.name)


def tp_major_axis_names():
  return (ParallelAxis.X.name,)


def tp_minor_axis_names():
  return (ParallelAxis.Y.name,)


@enum.unique
class ModelParallelStrategy(enum.Enum):
  """Overall Transformer Parallel Strategy."""

  TENSOR_PARALLEL = enum.auto()


@dataclasses.dataclass
class ModelParallelConfig:
  mesh: jax.sharding.Mesh
  parallel_type: ModelParallelStrategy = ModelParallelStrategy.TENSOR_PARALLEL


@enum.unique
class FFWParallelStrategy(enum.Enum):
  """Overall Transformer FFW Layer Parallel Strategy.

  Please refer to https://arxiv.org/pdf/2211.05102"""

  ONE_D_WEIGHT_STATIONARY = enum.auto()


@dataclasses.dataclass
class FeedForwardParallelConfig:
  mesh: jax.sharding.Mesh
  parallel_type: FFWParallelStrategy | None = None
  enable_collective_matmul: bool = False


@dataclasses.dataclass
class AttentionParallelConfig:
  mesh: jax.sharding.Mesh
  gather_input: bool = False
  reduce_output: bool = False


@enum.unique
class LinearParallelType(enum.Enum):
  """Parallel Type for Linear Layer weight."""

  ROW = enum.auto()
  COLUMN = enum.auto()


@enum.unique
class CollectiveMatmulType(enum.Enum):
  ALL_GATHER = enum.auto()
  REDUCE_SCATTER = enum.auto()


@dataclasses.dataclass
class LinearParallelConfig:
  mesh: jax.sharding.Mesh
  parallel_type: LinearParallelType | None = None
  reduce_scatter_output: bool = False
  reduce_output: bool = False
  collective_matmul_type: CollectiveMatmulType | None = None


@dataclasses.dataclass
class RMSNormParallelConfig:
  mesh: jax.sharding.Mesh
  activation_sharded: bool = False


@enum.unique
class EmbeddingParallelType(enum.Enum):
  """Parallel Type for Embedding Layer weight."""

  COLUMN = enum.auto()


@dataclasses.dataclass
class EmbeddingParallelConfig:
  mesh: jax.sharding.Mesh
  parallel_type: EmbeddingParallelType | None = None


@dataclasses.dataclass
class DecoderLayerParallelConfig:
  mesh: jax.sharding.Mesh
