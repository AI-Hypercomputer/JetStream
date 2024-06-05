# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the JetStream API.

These functions are the accelerator functions which an outer sampling loop
could want to call, enabling interleaved (continuous batching) inference.
"""

import abc
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

from flax import struct
import jax
import numpy as np

from jetstream.engine import tokenizer_pb2
from jetstream.engine import token_utils


# The model parameters - their partitioning will be unique for different prefill
# and decode topoologies.
Params = Any
# The result of a prefill operation, often a batch size 1 KVCache.
Prefix = Any
# The inputs into a generation step, often a prefill and generate cache tuple.
DecodeState = Any
# Accelerator representation of tokens.
DeviceTokens = Any
# Cpus asscociated with the mesh.
CpuDevices = Any
# Tokenkizer used by the engine
Tokenizer = Any


@struct.dataclass
class SlotData:
  """Class to store slot data."""

  tokens: Union[jax.Array, np.ndarray]
  valid: Union[jax.Array, np.ndarray]
  lengths: Union[jax.Array, np.ndarray]


# pylint: disable=g-doc-args
@struct.dataclass
class ResultTokens(abc.ABC):
  """Class to store returned tokens in.

  We store everything in one array, and keep indexes - because copying
  a single array to host is much faster.
  Each tuple represents the indices of the relevant data.
  """

  # Shape: [batch, tokens.shape[1] + validity.shape[1] + lengths.shape[1]]
  data: Union[jax.Array, np.ndarray]
  # The range of indices which contain tokens.
  tokens_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the validity of
  # the tokens.
  valid_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the lengths up till now of the lengths
  # of each generated sequence.
  length_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  samples_per_slot: int = struct.field(
      pytree_node=False,
  )

  def copy_to_host_async(self: "ResultTokens") -> None:
    """Copy to host asynchronously."""
    # Do nothing for np array
    if isinstance(self.data, np.ndarray):
      return
    self.data.copy_to_host_async()

  def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
    """Converts to numpy."""
    return ResultTokens(
        np.array(self.data),
        self.tokens_idx,
        self.valid_idx,
        self.length_idx,
        self.samples_per_slot,
    )

  def get_result_at_slot(self, slot: int) -> SlotData:
    """Returns the token at a given slot.

    Args:
      slot: An integer from [0, n) representing an index into the batch.

    Note: implementations of this method must correctly handle
    microbatches, if microbatches are used.
    """
    # Potentially get multiple beams for given slot.
    start_idx = slot * self.samples_per_slot
    end_idx = (slot + 1) * self.samples_per_slot
    # Mask out any non valid tokens.
    return SlotData(
        tokens=self.data[
            start_idx:end_idx, self.tokens_idx[0] : self.tokens_idx[1]
        ],
        valid=self.data[
            start_idx:end_idx, self.valid_idx[0] : self.valid_idx[1]
        ],
        # Only get a 1D representation here
        lengths=self.data[
            start_idx:end_idx, self.length_idx[0] : self.length_idx[1]
        ][:, 0],
    )


class Engine(abc.ABC):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  # Compiled prefills
  prefill_compiled: dict[int, jax.stages.Compiled]
  # Compiled inserts
  insert_compiled: dict[int, jax.stages.Compiled]
  # Compiled generate
  generate_compiled: jax.stages.Compiled
  prefill_buckets: list[int]

  @abc.abstractmethod
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Tuple[Prefix, ResultTokens]:
    """Computes a kv-cache for a set of tokens conditional on existing cache.

    existing_prefix (if provided) represents a prefix that has already been
    processed by the underlying model. tokens is logically appended
    to the text represented by `existing_prefix`. This method returns a new
    kv_cache (typically) for the resulting text.
    """

  @abc.abstractmethod
  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel.

    Generate takes a batch of pre-computed kv-caches, and computes:
      - the predicted next token for each of the sequences
      - an updated set of kv-caches

    In the case of pipelining, this will handle N cycles (where each cycle
    consists of each microbatch progressing through every stage), in
    non-pipelined code this is a full forward pass. In both cases, this accounts
    for a full embed-layerstack-unembed-sample operation.
    """

  @abc.abstractmethod
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Adds `new_request` into `caches` at 'slot'.

    When decoding multiple requests in parallel, when one request finishes, a
    new request must be slotted into the recently vacated spot: `insert`!

    This can occur in between and async to generate calls, and takes a lock over
    that row of the cache.

    The slot may represent a tuple of positions (e.g. microbatch, pipeline stage
    and batch), but at the engine interface level all of these are exposed as
    a [0, n) range of slots and converted internally.
    """

  @abc.abstractmethod
  def load_params(self, *args, **kwargs) -> Params:
    """Loads parameters.

    May not be used in full production form, where weights are part of the saved
    model.
    """

  @abc.abstractmethod
  def get_prefix_destination_sharding(self) -> Any:
    """Returns the shardings necessary to transfer data between engines."""

  @abc.abstractmethod
  def get_tokenizer(
      self,
  ) -> tokenizer_pb2.TokenizerParameters:
    """Returns the info to construct a tokenizer in py/c++."""

  def build_tokenizer(
      self,
      metadata: tokenizer_pb2.TokenizerParameters,
  ) -> Tokenizer:
    """Builds a new tokenizer object and returns it."""
    return token_utils.SentencePieceTokenizer(metadata)

  @abc.abstractmethod
  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    """Initialises any state which a generation step transforms."""

  @property
  @abc.abstractmethod
  def max_concurrent_decodes(self) -> int:
    """Total capacity."""

  @property
  @abc.abstractmethod
  def samples_per_slot(self) -> int:
    """Total samples per slot."""

  @property
  @abc.abstractmethod
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""

  @property
  @abc.abstractmethod
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""

  @property
  @abc.abstractmethod
  def colocated_cpus(self) -> Union[list[CpuDevices], None]:
    """CPU devices colocated with the engine's accelerators."""
