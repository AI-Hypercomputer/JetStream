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

"""Simple test engine for the JetStream API described.

Contains simple functions that we can hand calculate the desired outcome of.

Prefill: Doubles the sequence by multiplying it with an integer weight.
Insert: Writes this sequence into a cache row.
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight.

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""

import functools
from dataclasses import asdict
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import uuid
from flax import struct
from jax.experimental import mesh_utils

from jetstream.engine import engine_api
from jetstream.engine import tokenizer_pb2

Params = jax.Array  # [1,].


@struct.dataclass
class Prefix:
  """Result of the prefill step.

  This structure currently matches the prefix structure returned
  by the MaxText prefill operation, but it is flexible and can vary
  depending on the engine implementation.
  """

  logits: Optional[jax.Array] = None
  cache: Optional[jax.Array] = None
  next_pos: Optional[int] = None
  num_generated_tokens: Optional[int] = None
  first_token: Optional[int] = None


@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generate_tokens: jax.Array


class TestEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(
      self,
      batch_size: int,
      cache_length: int,
      weight: float,
      vocab_size: int = 1024,
      use_chunked_prefill: bool = False,
  ):
    self.prefill_cache_batch = batch_size
    self.generate_cache_batch = batch_size
    self.cache_length = cache_length
    self.weight = weight
    self.vocab_size = vocab_size
    self._mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((1, 1, 1), jax.devices()), ("x", "y", "z")
    )
    self._prng_key = jax.random.PRNGKey(42)
    self._use_chunked_prefill = use_chunked_prefill

  def load_params(self) -> Params:
    """Loads model weights."""
    # An integer, used to multiply inputs.
    return jnp.array([self.weight], dtype=jnp.float32)

  def load_params_dict(self) -> Params:
    """Loads model weights."""
    # An integer, used to multiply inputs.
    return {"params": jnp.array([self.weight], dtype=jnp.float32)}

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("request_id",),
  )
  # pylint: disable=unused-argument
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
      request_id: Optional[uuid.UUID] = None,
      previous_chunk=None,
      complete_padded_prompt=None,
      complete_prompt_true_length=None,
      positions=None,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Computes a kv-cache for a new generate request.

    Args:
      params: Scalar multiplier.
      existing_prefix: If provided, represents a prefix that has already been
        processed by the underlying model.
      padded_tokens: Logically appended tokens to any existing prefix, this is
        what we compute prefill on.
      true_length: The real length of the tokens, pre-pad.
    Returns:
      kv_cache: For the resulting text.
    """
    if existing_prefix is not None:
      raise NotImplementedError
    assert padded_tokens.ndim == 1

    # Generate dummy prefill cache content
    if not self._use_chunked_prefill:
      prefill_cache = padded_tokens[None, :] * params
    else:
      prefill_cache = padded_tokens[None, :]

    # Create a dummy first generated token.
    first_generated_token = (prefill_cache.sum(axis=-1).astype(jnp.int32))[
        :, jnp.newaxis
    ]

    if not self._use_chunked_prefill:
      prefix = Prefix(
          logits=jax.random.normal(self._prng_key, (1, self.vocab_size)),
          cache=prefill_cache,
          next_pos=jnp.full((1, 1), true_length, dtype=jnp.int32),
          num_generated_tokens=jnp.zeros((1, 1), dtype=jnp.int32),
          first_token=first_generated_token,
      )
    else:
      prefix = {
          "logits": jax.random.normal(self._prng_key, (1, self.vocab_size)),
          "cache": prefill_cache,
          "next_pos": jnp.full((1, 1), true_length, dtype=jnp.int32),
          "generated_tokens": jnp.zeros((1, 1), dtype=jnp.int32),
          "tokens": first_generated_token,
          "first_token": first_generated_token,
      }

    speculations = first_generated_token.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=jnp.concatenate(
            (
                first_generated_token,
                jnp.ones_like(first_generated_token),
                jnp.ones_like(first_generated_token),
            ),
            axis=-1,
        ),
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=self.generate_cache_batch // self.prefill_cache_batch,
    )
    return (prefix, result_tokens)

  @functools.partial(
      jax.jit, static_argnums=(0,), static_argnames=("num_samples",)
  )
  def prefill_multisampling(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
      rng: Optional[engine_api.PRNGKeyType] = None,
      num_samples: int = 1,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Computes a kv-cache for a new generate request.

    With multi-sampling, the engine will generate multiple first tokens in the
    prefilling stage. The number of tokens is specified by num_samples.
    """
    if existing_prefix is not None:
      raise NotImplementedError
    assert padded_tokens.ndim == 1

    # Generate dummy prefill cache content
    prefill_cache = padded_tokens[None, :] * params

    # Create dummy first generated tokens.
    first_generated_tokens = []
    for _ in range(num_samples):
      first_generated_token = (prefill_cache.sum(axis=-1).astype(jnp.int32))[
          :, jnp.newaxis
      ]
      first_generated_tokens.append(first_generated_token)
    first_generated_tokens = jnp.concatenate(first_generated_tokens, axis=0)

    prefix = Prefix(
        logits=jax.random.normal(self._prng_key, (1, self.vocab_size)),
        cache=prefill_cache,
        next_pos=jnp.full((1, 1), true_length, dtype=jnp.int32),
        num_generated_tokens=jnp.zeros((num_samples, 1), dtype=jnp.int32),
        first_token=first_generated_tokens,
    )

    speculations = first_generated_token.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=jnp.concatenate(
            (
                first_generated_tokens,
                jnp.ones_like(first_generated_tokens),
                jnp.ones_like(first_generated_tokens),
            ),
            axis=-1,
        ),
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=num_samples,
    )
    return (prefix, result_tokens)

  @functools.partial(jax.jit, static_argnums=(0,))
  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel."""
    (
        prefill_cache,
        generate_cache,
        generate_cache_index,
        generate_lengths,
        previous_timestep,
    ) = (
        decode_state.prefill_cache,
        decode_state.generate_cache,
        decode_state.generate_cache_index,
        decode_state.generate_lengths,
        decode_state.generate_tokens,
    )

    # Update generate cache
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        generate_cache,
        previous_timestep.astype(jnp.float32),
        start_index=generate_cache_index,
        axis=1,
    )
    generate_cache_index = (generate_cache_index + 1) % self.cache_length

    # Sum each row of prefill cache and generate cache to produce new timestep,
    # multiply by params.
    l_iota = jax.lax.broadcasted_iota(
        jnp.int32,
        (self.generate_cache_batch, self.cache_length),
        dimension=1,
    )

    # The generate cache should be circular and right aligned.
    # TODO: Do we need a left aligned one to test spec sampling?
    # Don't need the + 1 you normally would, because we don't provide a
    # token from prefill in the dummy.
    # This iota and masking is to allow for a cicular cache.
    length_mask = (
        -(l_iota - generate_cache_index) % self.cache_length
    ) <= generate_lengths[:, None]
    length_masked_gen_cache = generate_cache * length_mask
    generated_tokens = (
        prefill_cache.sum(axis=-1)
        + (length_masked_gen_cache.sum(axis=-1) / params)
    )[:, jnp.newaxis]
    # Wait to simulate model step time.
    fake_size = 4096
    fake_work = jnp.ones((fake_size, fake_size)) @ jnp.ones(
        (fake_size, fake_size)
    )
    # Do some fake work that isn't eliminated by dead code elimination (DCE).
    generate_cache = generate_cache + fake_work.mean() - fake_work.mean()
    new_lengths = generate_lengths + 1
    speculations = generated_tokens.shape[1]
    # Concatenates the tokens, their validity and the lengths of each sequence
    # into one tensor so that copy operations are faster on Cloud TPU
    # infrastructure.
    token_data = jnp.concatenate(
        [
            generated_tokens,
            jnp.ones_like(generated_tokens),
            new_lengths[:, None],
        ],
        axis=-1,
    )
    return DecodeState(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_cache_index=generate_cache_index,
        generate_lengths=new_lengths,
        generate_tokens=generated_tokens,
    ), engine_api.ResultTokens(
        data=token_data.astype(jnp.int32),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=self.generate_cache_batch // self.prefill_cache_batch,
    )

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      donate_argnums=(2,),
      static_argnames=("request_id",),
  )
  def insert(
      self,
      prefix: Any,
      decode_state: DecodeState,
      slot: int,
      request_id: Optional[uuid.UUID] = None,
  ) -> DecodeState:
    """Adds `prefix` into `decode_state` at `slot`."""
    if not self._use_chunked_prefill:
      prefill_cache = prefix.cache
    else:
      prefill_cache = prefix["cache"]

    prefill_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.prefill_cache, prefill_cache * 1.0, slot, axis=0
    )
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_cache,
        jnp.zeros((1, self.cache_length)),
        slot,
        axis=0,
    )
    samples_per_slot = self.generate_cache_batch // self.prefill_cache_batch
    generate_lengths = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_lengths,
        jnp.ones((samples_per_slot), dtype=jnp.int32),
        slot * samples_per_slot,
        axis=0,
    )
    if not self._use_chunked_prefill:
      first_token = prefix.first_token
    else:
      first_token = prefix["first_token"]
    generate_tokens = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_tokens,
        first_token,
        slot * samples_per_slot,
        axis=0,
    )
    return decode_state.replace(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_lengths=generate_lengths,
        generate_tokens=generate_tokens,
    )

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def bulk_insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slots: list[int],
  ) -> DecodeState:
    """Insert a single computed prefill cache into multiple slots in
    KV cache.
    """
    prefill_cache = decode_state.prefill_cache
    generate_cache = decode_state.generate_cache
    generate_lengths = decode_state.generate_lengths
    generate_tokens = decode_state.generate_tokens
    for slot in slots:
      prefill_cache = jax.lax.dynamic_update_slice_in_dim(
          prefill_cache, prefix.cache, slot, axis=0
      )
      generate_cache = jax.lax.dynamic_update_slice_in_dim(
          generate_cache,
          jnp.zeros((1, self.cache_length)),
          slot,
          axis=0,
      )
      samples_per_slot = 1
      generate_lengths = jax.lax.dynamic_update_slice_in_dim(
          generate_lengths,
          jnp.ones((samples_per_slot), dtype=jnp.int32),
          slot * samples_per_slot,
          axis=0,
      )
      generate_tokens = jax.lax.dynamic_update_slice_in_dim(
          generate_tokens,
          prefix.first_token,
          slot * samples_per_slot,
          axis=0,
      )
    return decode_state.replace(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_lengths=generate_lengths,
        generate_tokens=generate_tokens,
    )

  def get_prefix_destination_sharding(self) -> Any:
    res = jax.tree_util.tree_map(
        lambda _: jax.sharding.NamedSharding(
            mesh=self.mesh, spec=jax.sharding.PartitionSpec()
        ),
        asdict(Prefix()),
        is_leaf=lambda _: True,
    )
    return res

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path="test", extra_ids=0)

  def init_decode_state(self) -> DecodeState:
    """Initialises any state which a generation step transforms."""
    return DecodeState(
        prefill_cache=jnp.zeros(
            (self.prefill_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache=jnp.zeros(
            (self.generate_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache_index=0,
        generate_lengths=jnp.zeros(
            (self.generate_cache_batch), dtype=jnp.int32
        ),
        generate_tokens=jnp.zeros(
            (self.generate_cache_batch, 1), dtype=jnp.int32
        ),
    )

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return self.prefill_cache_batch

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return self.cache_length

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return self.generate_cache_batch // self.max_concurrent_decodes

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError

  @property
  def use_chunked_prefill(self) -> bool:
    """Maximum prefill length."""
    return self._use_chunked_prefill

  @property
  def prefill_chunk_size(self) -> int:
    """Maximum prefill length."""
    return 64
