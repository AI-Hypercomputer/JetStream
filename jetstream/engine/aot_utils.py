# Copyright 2025 Google LLC
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

"""Model server warmup utils."""

import numpy as np
import concurrent.futures
import logging
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import layout as jax_layout

from jetstream.engine import engine_api, token_utils

DLL = jax_layout.DeviceLocalLayout
Layout = jax_layout.Layout

Executable = jax.stages.Compiled

# key: prefill length, val: compiled prefill for the length
PrefillExecutables = dict[int, Executable]

GenerateExecutables = tuple[
    # Decode state init executable.
    Executable,
    # Insert executable
    dict[int, Executable],
    # Generate step executable.
    Executable,
]


def create_aot_engines(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    prefill_params: Optional[list[engine_api.Params]] = None,
    generate_params: Optional[list[engine_api.Params]] = None,
    relayout_optimally: bool = True,
) -> tuple[
    list[engine_api.Params],
    list[engine_api.Params],
    list[engine_api.JetStreamEngine],
    list[engine_api.JetStreamEngine],
]:
  (
      prefill_params,
      generate_params,
      prefill_executables,
      generate_executables,
  ) = layout_params_and_compile_executables(
      prefill_engines,
      generate_engines,
      prefill_params,
      generate_params,
      relayout_optimally,
  )

  assert len(prefill_executables) == len(prefill_engines)
  assert len(generate_executables) == len(generate_engines)
  aot_engines = [
      AotGenerateEngine(generate_engine, generate_executable)
      for generate_engine, generate_executable in zip(
          generate_engines, generate_executables
      )
  ]
  return (
      prefill_params,
      generate_params,
      prefill_engines if prefill_engines else [],
      aot_engines,
  )


class AotGenerateEngine(engine_api.JetStreamEngine):
  """A wrapper to JetStreamEngine
  This provides generate engine optimized based on auto layout.
  """

  def __init__(
      self,
      engine: engine_api.JetStreamEngine,
      generate_executables: GenerateExecutables,
  ):
    super().__init__(engine)
    self._generate_executables = generate_executables

  def insert(
      self,
      prefix: engine_api.Prefix,
      decode_state: engine_api.DecodeState,
      slot: int,
      prefill_length: Optional[int] = None,
  ) -> engine_api.DecodeState:

    return self._generate_executables[1][prefill_length](
        prefix,
        decode_state,
        slot,
    )

  def generate(
      self, params: engine_api.Params, decode_state: engine_api.DecodeState
  ) -> Tuple[engine_api.DecodeState, engine_api.ResultTokens]:
    return self._generate_executables[2](params, decode_state)

  def init_decode_state(self, *args, **kwargs) -> engine_api.DecodeState:
    return self._generate_executables[0](*args, **kwargs)


# class GenerateExecutable:


# TODO: Refactor to re-use the function in maxtext
def _identity(x: Any) -> Any:
  """Avoids lambda that breaks JAX caching."""
  return x


# TODO: Refactor to re-use the function in maxtext
def _iterated_layout(
    arrays: Any, layouts: Any, xla_flags: dict[str, Any] | None = None
) -> Any:
  """Lays out an array tensor by tensor to prevent OOMs."""

  def _layout(x, s, l):
    if x.layout == l:
      return x
    # Somehow this can be None sometimes.
    dll = l.device_local_layout if isinstance(l, Layout) else l
    f = (
        jax.jit(_identity, out_shardings=Layout(dll, s))
        .lower(x)
        .compile(compiler_options=xla_flags)
    )
    y = f(x)
    # Achieves donation of the input argument, but allows for different memory
    # layouts and shapes.
    jax.tree.map(lambda z: z.delete(), x)
    jax.block_until_ready(y)
    return y

  shardings = jax.tree.map(lambda x: x.sharding, arrays)
  arrays = jax.tree.map(_layout, arrays, shardings, layouts)
  return arrays


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    prefill_params: Optional[list[engine_api.Params]] = None,
    generate_params: Optional[list[engine_api.Params]] = None,
    relayout_optimally: bool = True,
) -> tuple[
    list[engine_api.Params],
    list[engine_api.Params],
    list[PrefillExecutables],
    list[GenerateExecutables],
]:
  """Organizes the engines and executables.

  Args:
      prefill_engines: Prefill only engines.
      generate_engines: Generate only engines.
      prefill_params: Prefill only params.
      generate_params: Generate only params.
  """
  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []

  prefill_executables_list: list[PrefillExecutables] = []
  generate_executables_list: list[GenerateExecutables] = []

  any_prefill_engine = None
  any_prefill_params = None

  for i, pe in enumerate(prefill_engines):
    prefill_executables, prefill_params[i] = _initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
        relayout_params_optimally=relayout_optimally,
    )
    any_prefill_engine = pe
    any_prefill_params = prefill_params[i]
    prefill_executables_list.append(prefill_executables)

  for i, ge in enumerate(generate_engines):
    (generate_executables, generate_params[i]) = (
        _initialize_insert_generate_jit_cache(
            prefill_engine=any_prefill_engine,
            generate_engine=ge,
            prefill_params=any_prefill_params,
            generate_params=generate_params[i],
            generate_idx=i,
            relayout_optimally=relayout_optimally,
        )
    )
    generate_executables_list.append(generate_executables)

  return (
      prefill_params,
      generate_params,
      prefill_executables_list,
      generate_executables_list,
  )


def _get_prefill_buckets(prefill_engine: engine_api.JetStreamEngine):
  """Returns the list of prefill buckets including the max prefill length"""
  prefill_buckets = [
      bucket
      for bucket in token_utils.DEFAULT_PREFILL_BUCKETS
      if bucket <= prefill_engine.max_prefill_length
  ]
  prefill_engine.prefill_buckets = prefill_buckets
  if prefill_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(prefill_engine.max_prefill_length)
  logging.info("Prefill buckets: %s", prefill_buckets)
  return prefill_buckets


def _to_shape_dtype(
    t: Any, sharding: None | Any = None
) -> jax.ShapeDtypeStruct:
  if hasattr(t, "sharding"):
    return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=t.sharding)
  else:
    return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=sharding)


def _initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: engine_api.Params,
    prefill_idx: int,
    relayout_params_optimally: bool = False,
) -> tuple[PrefillExecutables, engine_api.Params]:
  """Precompile all prefill functions in parallel.
  If we don't do this, then when a new request triggers a new prefill bucket it
  will take a very long time for that query to come back.

  Args:
      prefill_engine: A prefill engine to be compiled for.
      prefill_params: The associated prefill parameters.
      prefill_idx: Which prefill engine it is.
      relayout_params_optimally: When set, modify the param layouts and
        re-layout the params to make them optimally laid out for prefill.
  Returns:
    A tuple, whose first element is a dictionary {prefill_length : executable}
    and second argument is relayed out params when relayout is enabled.
  """
  # Get a list of prefill buckets
  prefill_buckets = _get_prefill_buckets(prefill_engine)

  logging.info("---Prefill compilation %d began.---", prefill_idx)
  prefill_param_shapes = jax.tree.map(_to_shape_dtype, prefill_params)
  padded_tokens = jnp.ones((prefill_engine.max_prefill_length), dtype=jnp.int32)
  padded_tokens_shape = jax.ShapeDtypeStruct(
      padded_tokens.shape, padded_tokens.dtype
  )
  length_shape = jax.ShapeDtypeStruct((), jnp.int32)

  # TODO(wyzhang): Consider if this can be merged into maxtext
  def _compile_prefill(length) -> tuple[int, Executable]:
    prefill_executable = (
        jax.jit(
            prefill_engine.prefill_aot,
            in_shardings=(Layout(DLL.AUTO), None, None),
        )
        .lower(prefill_param_shapes, padded_tokens_shape, length_shape)
        .compile(compiler_options=None)
    )  # TODO(wyzhang): pass in xla flag
    logging.info(
        "---Prefill engine %d compiled for prefill length %d.---",
        prefill_idx,
        length,
    )
    return (length, prefill_executable)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    executables = dict(executor.map(_compile_prefill, prefill_buckets))
    prefill_engine.aot = True
  logging.info("---Prefill compilation %d complete.---", prefill_idx)

  # Relayout params if needed.
  if relayout_params_optimally:
    input_args, _ = executables[prefill_engine.max_prefill_length].input_layouts
    prefill_param_layouts = input_args[0]
    prefill_params = _iterated_layout(prefill_params, prefill_param_layouts)

  return (executables, prefill_params)


def _compile_generate_and_get_layouts(
    generate_engine: engine_api.JetStreamEngine,
    generate_params: engine_api.Params,
    generate_idx: int,
    decode_state: Optional[Any] = None,
    relayout_optimally: bool = False,
) -> tuple[Executable, Layout, Layout, Layout]:
  param_layout = Layout(DLL.AUTO if relayout_optimally else None)
  decode_state_layout = Layout(DLL.AUTO if relayout_optimally else None)
  # TODO(wyzhang): Pass XLA flags
  executable = (
      jax.jit(
          generate_engine.generate_aot,
          in_shardings=(param_layout, decode_state_layout),
          out_shardings=(param_layout, decode_state_layout),
          donate_argnums=(1,),
      )
      .lower(generate_params, decode_state)
      .compile(compiler_options=None)
  )
  arg_layouts, _ = executable.input_layouts
  generated_out_layouts, _ = executable.output_layouts
  logging.info(
      "---Generate engine %d compiled for generate.---",
      generate_idx,
  )
  return (executable, arg_layouts[0], arg_layouts[1], generated_out_layouts)


def replace_devices_in_sharding(
    sharding: Any,
    devices: np.ndarray,
) -> Any:
  """Util function to replace the devices in the sharding.

  This is typically used when the output of a program is transferred to another
  slice of devices to be further processed, e.g. prefill -> generate transfer.

  Args:
    sharding: PyTree of shardings. The leaves of the tree has to be of type
      jax.sharding.NamedSharding.
    devices: The devices to replace with.
  Returns:
    A tree-structured sharding object with the same structure as input @sharding
    while all the devices are replaced with @devices.
  """

  def replace_mesh_devices(
      mesh: jax.sharding.Mesh, spec: jax.sharding.PartitionSpec
  ) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(
        devices.reshape(mesh.devices.shape), mesh.axis_names
    )

  def replace_sharding_devices(sharding: Any) -> jax.sharding.NamedSharding:
    assert isinstance(sharding, jax.sharding.NamedSharding)
    if list(sharding.mesh.devices.flat) == list(devices.flat):
      return sharding
    return jax.sharding.NamedSharding(
        replace_mesh_devices(sharding.mesh, sharding.spec),
        sharding.spec,
        memory_kind=sharding.memory_kind,
        _parsed_pspec=sharding._parsed_pspec,  # pylint: disable=protected-access
        _manual_axes=sharding._manual_axes,  # pylint: disable=protected-access
    )

  return jax.tree_util.tree_map(replace_sharding_devices, sharding)


def _initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    generate_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    generate_params: Any,
    generate_idx: int,
    relayout_optimally: bool = False,
) -> tuple[GenerateExecutables, engine_api.Params]:
  """Initialiszes jit cache for insert and generate.

  Args:
      generate_engine: A generate engine to be compiled for.
      generate_params: The associated parameters.
      generate_idx: Which generate engine it is.
  """

  decode_state_shapes = jax.eval_shape(generate_engine.init_decode_state)
  generate_param_shapes = jax.tree.map(_to_shape_dtype, generate_params)
  (
      generate_executable,
      param_layouts,
      decode_state_layouts,
      generated_out_layouts,
  ) = _compile_generate_and_get_layouts(
      generate_engine,
      generate_param_shapes,
      generate_idx,
      decode_state_shapes,
      relayout_optimally,
  )
  if relayout_optimally:
    input_args, _ = generate_executable.input_layouts
    generate_param_layouts = input_args[0]
    generate_params = _iterated_layout(generate_params, generate_param_layouts)

  # Compile insert
  def _compile_insert(length) -> tuple[int, Executable]:
    def _prefill():
      prefix, _ = prefill_engine._downstream_engine.prefill(
          # pylint: disable=protected-access
          params=prefill_params,
          padded_tokens=jnp.ones((length), dtype=jnp.int32),
          true_length=length,
      )
      return prefix

    prefix_shape = jax.eval_shape(_prefill)
    print(f"prefix_shape {prefix_shape}")
    slot_shape = jax.ShapeDtypeStruct((), jnp.int32)
    # TODO(wyzhang): Pass XLA flag
    insert_executable = (
        jax.jit(
            generate_engine.insert,
            in_shardings=(None, decode_state_layouts, None),
            out_shardings=decode_state_layouts,
            donate_argnums=(1,),
        )
        .lower(prefix_shape, decode_state_shapes, slot_shape)
        .compile(compiler_options=None)
    )
    logging.info(
        "---Generate engine %d compiled for insert length %d.---",
        generate_idx,
        length,
    )
    return (length, insert_executable)

  prefill_buckets = _get_prefill_buckets(prefill_engine)
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    insert_executables = dict(executor.map(_compile_insert, prefill_buckets))

  # Compile init decode state
  def _compile_init_decode_state():
    # TODO(wyzhang): Pass in XLA flag
    decode_state_executable = (
        jax.jit(
            generate_engine.init_decode_state,
            in_shardings=(None),
            out_shardings=(decode_state_layouts),
        )
        .lower()
        .compile(compiler_options=None)
    )
    logging.info(
        "---Generate engine %d compiled for init decode state.---",
        generate_idx,
    )
    return decode_state_executable

  init_decode_state_executable = _compile_init_decode_state()

  generate_engine.aot = True
  return (
      (init_decode_state_executable, insert_executables, generate_executable),
      generate_params,
  )
