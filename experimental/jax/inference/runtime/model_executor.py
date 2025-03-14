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

import datetime
import dataclasses
import jax
from jax import numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from collections.abc import Callable
from inference.nn import AttentionMetadata, KVCache
from inference.model import ModelOutput, SamplingParams
from inference import parallel
from inference.runtime.batch_scheduler import Schedule, PrefillPagesUpdate
from inference.runtime.request_type import GenerateState, PrefillRequest

ModelForwardFunc = Callable[
    [
        dict,
        jax.Array,
        jax.Array,
        list[KVCache],
        AttentionMetadata,
        SamplingParams,
    ],
    tuple[ModelOutput, list[KVCache]],
]


@dataclasses.dataclass
class ModelForwardInput:
  input_ids: jax.Array
  positions: jax.Array
  kv_caches: list[KVCache]
  attn_metadata: AttentionMetadata
  sampling_params: SamplingParams


class Executor:

  def __init__(
      self,
      mesh: Mesh,
      weights_dict: dict,
      model_forward: ModelForwardFunc,
      num_pages_per_seq: int,
      cache_dir: str | None = "/tmp/jax_cache",
      debug_mode: bool = False,
  ):
    self.mesh = mesh
    self.weights_dict = weights_dict
    self.executables_dict: dict[str:ModelForwardFunc] = {}
    self._model_forward = model_forward
    self.num_pages_per_seq = num_pages_per_seq

    # TODO: Understand why the following doesn't work.
    # Currently, cache is saved by "export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache".
    if cache_dir:
      from jax.experimental.compilation_cache import compilation_cache as cc

      cc.set_cache_dir(cache_dir)

    self.dummy_scalar = jax.device_put(
        jnp.asarray(1e7, dtype=jnp.int32), NamedSharding(self.mesh, P())
    )
    self.jitted_prepare_model_input = jax.jit(
        self._prepare_model_input, static_argnames=("chunk_size",)
    )
    self.debug_mode = debug_mode

  def _prepare_model_input(
      self,
      attn_meta: AttentionMetadata,
      prefill_tpp: tuple[jax.Array, jax.Array, jax.Array],
      prefill_length: jax.Array,
      chunk_id: jax.Array,
      chunk_size: jax.Array,
      generate_tpp: tuple[list[jax.Array], jax.Array, jax.Array],
      update_generate_tpp: tuple[jax.Array, jax.Array, jax.Array],
      insert_slots: jax.Array,
      generate_page_updates: jax.Array,
      generate_pt_update_slots: jax.Array,
      generate_pt_update_page_idxs: jax.Array,
  ):
    p_tokens, p_positions, p_page_indices = prefill_tpp
    if len(p_tokens.shape) > 0:
      idx = chunk_id * chunk_size
      p_tokens = jax.lax.dynamic_slice_in_dim(p_tokens, idx, chunk_size)
      p_positions = jax.lax.dynamic_slice_in_dim(p_positions, idx, chunk_size)

    g_tokens, g_positions, g_page_table = generate_tpp
    if len(g_tokens.shape) > 0:
      update_g_tokens, update_g_positions, update_g_page_table = (
          update_generate_tpp
      )
      update_g_tokens = jnp.asarray(update_g_tokens)

      g_tokens = g_tokens.at[insert_slots].set(update_g_tokens)
      g_positions = g_positions.at[insert_slots].set(update_g_positions)
      # Insert new request to the slot.
      g_page_table = g_page_table.at[insert_slots, :].set(update_g_page_table)
      # Add the new page for the existing slot.
      g_page_table = g_page_table.at[
          generate_pt_update_slots, generate_pt_update_page_idxs
      ].set(generate_page_updates)

    if len(p_tokens.shape) > 0 and len(g_tokens.shape) > 0:
      input_ids = jnp.concatenate((p_tokens, g_tokens))
      positions = jnp.concatenate((p_positions, g_positions))

      attn_meta.prefill_length = prefill_length
      attn_meta.prefill_pos = p_positions
      attn_meta.prefill_page_table = p_page_indices
      attn_meta.generate_pos = g_positions
      attn_meta.generate_page_table = g_page_table

    elif len(p_tokens.shape) > 0:
      input_ids = p_tokens
      positions = p_positions

      attn_meta.prefill_length = prefill_length
      attn_meta.prefill_pos = p_positions
      attn_meta.prefill_page_table = p_page_indices

    elif len(g_tokens.shape) > 0:
      input_ids = g_tokens
      positions = g_positions

      attn_meta.generate_pos = g_positions
      attn_meta.generate_page_table = g_page_table

    else:
      raise ValueError(
          "Failed to build the input as no prefill or generate gets scheduled"
      )

    return input_ids, g_tokens, positions, attn_meta

  def prepare_input_and_update_generate_state(
      self,
      schedule: Schedule,
      generate_state: GenerateState,
      kv_caches: list[KVCache],
      sampling_params: SamplingParams,
      batch_size: int,
  ) -> ModelForwardInput:
    attn_meta = AttentionMetadata(
        prefill_length=self.dummy_scalar,
        prefill_pos=self.dummy_scalar,
        prefill_page_table=self.dummy_scalar,
        generate_pos=self.dummy_scalar,
        generate_page_table=self.dummy_scalar,
    )

    prefill_tpp = (self.dummy_scalar, self.dummy_scalar, self.dummy_scalar)
    prefill_cur_length = self.dummy_scalar
    chunk_id = self.dummy_scalar
    chunk_size = 512

    generate_tpp = (self.dummy_scalar, self.dummy_scalar, self.dummy_scalar)
    update_generate_tpp = (
        self.dummy_scalar,
        self.dummy_scalar,
        self.dummy_scalar,
    )
    insert_slots = self.dummy_scalar
    generate_page_updates = self.dummy_scalar
    generate_pt_update_slots = self.dummy_scalar
    generate_pt_update_page_idxs = self.dummy_scalar

    if schedule.schedule_prefill:
      prefill = schedule.prefill_request
      prefill_tpp = (
          prefill.device_token_ids,
          prefill.device_positions,
          np.array(prefill.page_indices),
      )
      prefill_cur_length = (prefill.chunk_idx + 1) * prefill.chunk_size
      prefill_total_len = len(prefill.unpadded_token_ids)
      if prefill_cur_length > prefill_total_len:
        prefill_cur_length = prefill_total_len

      chunk_id = prefill.chunk_idx
      chunk_size = prefill.chunk_size

    if schedule.schedule_generate:
      generate_tpp = (
          generate_state.token_ids,
          generate_state.positions,
          generate_state.page_table,
      )
      update_token_ids = []
      update_pos = np.full((batch_size,), 1e6, dtype=np.int32)
      update_page_indices = np.full(
          (batch_size, self.num_pages_per_seq), 1e6, dtype=np.int32
      )
      slots = np.full((batch_size,), 1e6, dtype=np.int32)

      for i, gr in enumerate(schedule.new_generate_requests):
        update_token_ids.append(gr.device_prefill_token_id)
        update_pos[i] = gr.pos
        update_page_indices[i] = np.array(gr.page_indices)
        slots[i] = gr.slot

      for i in range(batch_size - len(schedule.new_generate_requests)):
        update_token_ids.append(self.dummy_scalar)

      update_generate_tpp = (update_token_ids, update_pos, update_page_indices)
      insert_slots = slots

      # Handle page indices update.
      page_update_slots = np.full((batch_size,), 1e6, dtype=np.int32)
      page_update_page_idxs = np.full((batch_size,), 1e6, dtype=np.int32)
      page_update_mapped_idxs = np.full((batch_size,), 1e6, dtype=np.int32)

      for i, update in enumerate(schedule.generate_state_page_updates):
        page_update_slots[i] = update.slot
        page_update_page_idxs[i] = update.page_idx
        page_update_mapped_idxs[i] = update.mapped_idx

      generate_page_updates = page_update_mapped_idxs
      generate_pt_update_slots = page_update_slots
      generate_pt_update_page_idxs = page_update_page_idxs

    input_ids, generate_tokens, positions, attn_meta = (
        self.jitted_prepare_model_input(
            attn_meta,
            prefill_tpp,
            prefill_cur_length,
            chunk_id,
            chunk_size,
            generate_tpp,
            update_generate_tpp,
            insert_slots,
            generate_page_updates,
            generate_pt_update_slots,
            generate_pt_update_page_idxs,
        )
    )
    _, new_key = jax.random.split(sampling_params.rng)
    sampling_params.rng = new_key

    if schedule.schedule_generate:
      generate_state.token_ids = generate_tokens
      generate_state.positions = attn_meta.generate_pos
      generate_state.page_table = attn_meta.generate_page_table

    return ModelForwardInput(
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches,
        attn_metadata=attn_meta,
        sampling_params=sampling_params,
    )

  def _executable_key(self, attn_meta: AttentionMetadata) -> str:
    prefill_chunk_size = (
        attn_meta.prefill_pos.shape[0]
        if len(attn_meta.prefill_pos.shape) > 0
        else 0
    )
    generate_batch_size = (
        attn_meta.generate_pos.shape[0]
        if len(attn_meta.generate_pos.shape) > 0
        else 0
    )
    return f"prefill_chunk_size={prefill_chunk_size}, generate_batch_size={generate_batch_size}"

  def _shard_mapped_model_forward(self, input: ModelForwardInput):
    return shard_map(
        f=self._model_forward,
        mesh=self.mesh,
        in_specs=(
            parallel.get_partition_spec(self.weights_dict),
            P(None),
            P(None),
            parallel.get_partition_spec(input.kv_caches),
            parallel.get_partition_spec(input.attn_metadata),
            parallel.get_partition_spec(input.sampling_params),
        ),
        out_specs=(
            ModelOutput(
                prefill_token=P(),
                prefill_done=P(),
                prefill_next_pos=P(),
                generate_tokens=P(None),
                generate_done=P(None),
                generate_next_pos=P(None),
            ),
            parallel.get_partition_spec(input.kv_caches),
        ),
        check_rep=False,
    )

  def _compile_once(
      self,
      key: str,
      input: ModelForwardInput,
      options: jax.stages.CompilerOptions,
  ):
    print(f"Compiling for ({key}) ...", end="")
    start_time = datetime.datetime.now()
    jitted_func = jax.jit(
        self._shard_mapped_model_forward(input), donate_argnums=(3,)
    )
    self.executables_dict[key] = jitted_func.lower(
        self.weights_dict,
        input.input_ids,
        input.positions,
        input.kv_caches,
        input.attn_metadata,
        input.sampling_params,
    ).compile(options)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f" took {duration:.2f} seconds")

  def _compile_prefix_only(
      self,
      page_size: int,
      chunk_size: int,
      batch_size: int,
      max_seq_len: int,
      max_input_len: int,
      kv_caches: list[KVCache],
      sampling_params: SamplingParams,
      compiler_options: dict[str, jax.stages.CompilerOptions] | None = None,
  ):
    dummy_padded_tensor = jnp.ones(
        (max_input_len,),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None)),
    )
    dummy_page_indices_tensor = jnp.ones(
        (max_seq_len // page_size,),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None)),
    )
    dummy_page_indices = np.asarray(dummy_page_indices_tensor).tolist()
    dummy_page_update_indices = [0 for _ in range(chunk_size // page_size)]
    dummy_schedule = Schedule(
        schedule_prefill=True,
        prefill_request=PrefillRequest(
            id="0",
            unpadded_token_ids=[1],
            chunk_idx=0,
            chunk_size=chunk_size,
            page_indices=dummy_page_indices,
            device_token_ids=dummy_padded_tensor,
            device_positions=dummy_padded_tensor,
        ),
        prefill_pages_update=PrefillPagesUpdate(
            page_indices=dummy_page_update_indices
        ),
        schedule_generate=False,
        new_generate_requests=[],
        generate_state_page_updates=[],
    )
    input = self.prepare_input_and_update_generate_state(
        schedule=dummy_schedule,
        generate_state=None,
        kv_caches=kv_caches,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    key = self._executable_key(input.attn_metadata)
    options = None
    if compiler_options and key in compiler_options:
      options = compiler_options[key]
    self._compile_once(key, input, options)

  def _compile_generate_only(
      self,
      page_size: int,
      batch_size: int,
      max_seq_len: int,
      kv_caches: list[KVCache],
      sampling_params: SamplingParams,
      compiler_options: dict[str, jax.stages.CompilerOptions] | None = None,
  ):
    dummy_batch_tensor = jnp.ones(
        (batch_size),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None)),
    )
    dummy_page_table_tensor = jnp.ones(
        (batch_size, max_seq_len // page_size),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None, None)),
    )
    dummy_schedule = Schedule(
        schedule_prefill=False,
        prefill_request=None,
        prefill_pages_update=None,
        schedule_generate=True,
        new_generate_requests=[],
        generate_state_page_updates=[],
    )
    dummy_generate_state = GenerateState(
        token_ids=dummy_batch_tensor,
        positions=dummy_batch_tensor,
        page_table=dummy_page_table_tensor,
        available_slots=0,
        active_slot_req_map={},
    )
    input = self.prepare_input_and_update_generate_state(
        schedule=dummy_schedule,
        generate_state=dummy_generate_state,
        kv_caches=kv_caches,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    key = self._executable_key(input.attn_metadata)
    options = None
    if compiler_options and key in compiler_options:
      options = compiler_options[key]
    self._compile_once(key, input, options)

  def _compile_prefill_generate(
      self,
      page_size: int,
      chunk_size: int,
      batch_size: int,
      max_seq_len: int,
      max_input_len: int,
      kv_caches: list[KVCache],
      sampling_params: SamplingParams,
      compiler_options: dict[str, jax.stages.CompilerOptions] | None = None,
  ):
    dummy_batch_tensor = jnp.ones(
        (batch_size),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None)),
    )
    dummy_page_table_tensor = jnp.ones(
        (batch_size, max_seq_len // page_size),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None, None)),
    )
    dummy_padded_prompt_tensor = jnp.ones(
        (max_input_len,),
        dtype=jnp.int32,
        device=NamedSharding(self.mesh, P(None)),
    )
    dummy_prefill_page_indices = np.ones(
        (max_seq_len // page_size,),
        dtype=np.int32,
    ).tolist()
    dummy_prefill_page_update_indices = [
        0 for _ in range(chunk_size // page_size)
    ]
    dummy_schedule = Schedule(
        schedule_prefill=True,
        prefill_request=PrefillRequest(
            id="0",
            unpadded_token_ids=[1],
            chunk_idx=0,
            chunk_size=chunk_size,
            page_indices=dummy_prefill_page_indices,
            device_token_ids=dummy_padded_prompt_tensor,
            device_positions=dummy_padded_prompt_tensor,
        ),
        prefill_pages_update=PrefillPagesUpdate(
            page_indices=dummy_prefill_page_update_indices
        ),
        schedule_generate=True,
        new_generate_requests=[],
        generate_state_page_updates=[],
    )
    dummy_generate_state = GenerateState(
        token_ids=dummy_batch_tensor,
        positions=dummy_batch_tensor,
        page_table=dummy_page_table_tensor,
        available_slots=0,
        active_slot_req_map={},
    )
    input = self.prepare_input_and_update_generate_state(
        schedule=dummy_schedule,
        generate_state=dummy_generate_state,
        kv_caches=kv_caches,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    key = self._executable_key(input.attn_metadata)
    options = None
    if compiler_options and key in compiler_options:
      options = compiler_options[key]
    self._compile_once(key, input, options)

  def compile(
      self,
      prefill_chunk_sizes: list[int],
      batch_size: int,
      max_seq_len: int,
      max_input_len: int,
      kv_caches: list[KVCache],
      sampling_params: SamplingParams,
      compiler_options: dict[str, jax.stages.CompilerOptions] | None = None,
  ):
    page_size = kv_caches[0].k.shape[2]
    self._compile_generate_only(
        page_size,
        batch_size,
        max_seq_len,
        kv_caches,
        sampling_params,
        compiler_options,
    )

    for chunk_size in prefill_chunk_sizes:
      self._compile_prefix_only(
          page_size,
          chunk_size,
          batch_size,
          max_seq_len,
          max_input_len,
          kv_caches,
          sampling_params,
          compiler_options,
      )

    for chunk_size in prefill_chunk_sizes:
      self._compile_prefill_generate(
          page_size,
          chunk_size,
          batch_size,
          max_seq_len,
          max_input_len,
          kv_caches,
          sampling_params,
          compiler_options,
      )

  def execute(
      self, input: ModelForwardInput
  ) -> tuple[ModelOutput, list[KVCache]]:
    if self.debug_mode:
      func = self._shard_mapped_model_forward(input)
    else:
      key = self._executable_key(input.attn_metadata)
      func = self.executables_dict[key]

    return func(
        self.weights_dict,
        input.input_ids,
        input.positions,
        input.kv_caches,
        input.attn_metadata,
        input.sampling_params,
    )
