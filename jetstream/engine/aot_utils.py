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

"""Ahead-of-time compilation for engine computations."""

import concurrent.futures
import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import layout as jax_layout

from jetstream.engine import engine_api, token_utils

# Configure logging
log = logging.getLogger(__name__)  # Use __name__ for better module tracking
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
log.addHandler(console_handler)


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


# TODO: Evaluate if this should be in a common lib or reuse those in Maxtext
def _identity(x: Any) -> Any:
  """Avoids lambda that breaks JAX caching."""
  return x


# TODO: Evaluate if this should be in a common lib or reuse those in Maxtext
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
    interleaved_engines: Optional[list[
      tuple[engine_api.JetStreamEngine, engine_api.JetStreamEngine]]] = None,
    prefill_params: Optional[list[engine_api.Params]] = None,
    generate_params: Optional[list[engine_api.Params]] = None,
    shared_params: Optional[list[engine_api.Params]] = None,
    relayout_params_optimally: bool = True,
    relayout_decode_state_optimally: bool = True,
) -> tuple[
  list[engine_api.JetStreamEngine],
  list[engine_api.JetStreamEngine],
  list[PrefillExecutables],
  list[GenerateExecutables],
  list[engine_api.Params],
  list[engine_api.Params],
]:
  """Compile engine and optimize param & decode state layouts if needed.

  Args:
    prefill_engines: Prefill only engines.
    generate_engines: Generate only engines.
    interleaved_engines: A list of tuples where the first is an prefill
      engine and the second is a generate engine.
    prefill_params: Params for prefill engine.
    generate_params: Params for generate engine.
    shared_params: Params shared by generate and prefill engines.
  """
  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  interleaved_engines = interleaved_engines if interleaved_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []
  shared_params = shared_params if shared_params else []

  out_prefill_engines = []
  out_generate_engines = []
  out_prefill_executables = []
  out_generate_executables = []
  out_prefill_params = []
  out_generate_params = []

  any_prefill_engine = None
  any_prefill_params = None

  for i, pe in enumerate(prefill_engines):
    (
      prefill_executables, 
      prefill_params_i 
    ) = _initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_params_layouts_override=None,
        prefill_idx=i,
        relayout_params_optimally=relayout_params_optimally,
    )
    out_prefill_engines.append(pe)
    out_prefill_executables.append(prefill_executables)
    out_prefill_params.append(prefill_params_i)
    any_prefill_engine = pe
    any_prefill_params = prefill_params_i

  for i, ge in enumerate(generate_engines):
    (
        generate_executables,
        generate_params_i,
        _,
    ) = _initialize_insert_generate_jit_cache(
        prefill_engine=any_prefill_engine,
        generate_engine=ge,
        prefill_params=any_prefill_params,
        generate_params=generate_params[i],
        generate_idx=i,
        relayout_params_optimally=relayout_params_optimally,
        relayout_decode_state_optimally=relayout_decode_state_optimally,
    )
    out_generate_engines.append(ge)
    out_generate_executables.append(generate_executables)
    out_generate_params.append(generate_params_i)

  for i, (pe, ge) in enumerate(interleaved_engines):
    # For interleaved engines, use layout optimized for generate engine.
    (
        generate_executables,
        generate_params_i,
        generate_params_layouts_i
    ) = _initialize_insert_generate_jit_cache(
        prefill_engine=pe,
        generate_engine=ge,
        prefill_params=shared_params[i],
        generate_params=shared_params[i],
        generate_idx=i,
        relayout_params_optimally=relayout_params_optimally,
        relayout_decode_state_optimally=relayout_decode_state_optimally,
    )
    prefill_params_i = generate_params_i
    (
        prefill_executables,
        _
    ) = _initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params_i,
        prefill_params_layouts_override=generate_params_layouts_i,
        prefill_idx=i,
        # No need to relayout prefill params, since prefill and generate
        # share the same params.
        relayout_params_optimally=False
    )
    out_prefill_engines.append(pe)
    out_prefill_executables.append(prefill_executables)
    out_prefill_params.append(prefill_params_i)
    out_generate_engines.append(ge)
    out_generate_executables.append(generate_executables)
    out_generate_params.append(generate_params_i)

  return (
      out_prefill_engines,
      out_generate_engines,
      out_prefill_executables,
      out_generate_executables,
      out_prefill_params,
      out_generate_params,
  )


def _get_prefill_buckets(
    prefill_engine: engine_api.JetStreamEngine
) -> list[int]:
  """Returns the list of prefill buckets including the max prefill length"""
  prefill_buckets = [
      bucket
      for bucket in token_utils.DEFAULT_PREFILL_BUCKETS
      if bucket <= prefill_engine.max_prefill_length
  ]
  prefill_engine.prefill_buckets = prefill_buckets
  if prefill_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(prefill_engine.max_prefill_length)
  return prefill_buckets


def _to_shape_dtype(
    t: jax.Array, sharding: Optional[jax.sharding.Sharding] = None
) -> jax.ShapeDtypeStruct:
  if hasattr(t, 'sharding'):
    return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=t.sharding)
  else:
    return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=sharding)


def _initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: engine_api.Params,
    prefill_params_layouts_override: Any,
    prefill_idx: int,
    relayout_params_optimally: bool,
) -> tuple[PrefillExecutables, engine_api.Params]:
  """Precompile all prefill functions in parallel.
  If we don't do this, then when a new request triggers a new prefill bucket it
  will take a very long time for that query to come back.

  Args:
      prefill_engine: A prefill engine to be compiled.
      prefill_params: The associated prefill parameters.
      prefill_idx: Which prefill engine it is.
      relayout_params_optimally: When set, modify the param layouts and
        re-layout the params to make them optimally laid out for prefill.
  Returns:
    A tuple, whose first element is a dictionary {prefill_length : executable}
    and second argument is relayed out params, when relayout is enabled, otherwise
    the original params.
  """
  # Get a list of prefill buckets
  prefill_buckets = _get_prefill_buckets(prefill_engine)

  log.info("---Prefill engine %d compilation began.---", prefill_idx)
  prefill_param_shapes = jax.tree.map(_to_shape_dtype, prefill_params)
  padded_tokens = jnp.ones((prefill_engine.max_prefill_length), dtype=jnp.int32)
  padded_tokens_shape = jax.ShapeDtypeStruct(
      padded_tokens.shape, padded_tokens.dtype
  )
  length_shape = jax.ShapeDtypeStruct((), jnp.int32)

  def _compile_prefill(length) -> tuple[int, Executable]:
    if prefill_params_layouts_override:
      in_shardings = (prefill_params_layouts_override, None, None)
    else:
      in_shardings = (Layout(DLL.AUTO), None, None)
    prefill_executable = jax.jit(
        prefill_engine.prefill_aot,
        in_shardings=in_shardings,
        out_shardings=(Layout(DLL.AUTO), Layout(DLL.AUTO)),
    ).lower(
        prefill_param_shapes, padded_tokens_shape, length_shape
    ).compile(compiler_options=None)  # TODO(wyzhang): pass in xla flag
    log.info(
        "---Prefill engine %d compiled for prefill length %d.---",
        prefill_idx, length,
    )
    return (length, prefill_executable)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    executables = dict(executor.map(_compile_prefill, prefill_buckets))
    prefill_engine.aot = True
  log.info("---Prefill engine %d compilation completed.---", prefill_idx)

  # Relayout params if needed.
  if relayout_params_optimally:
    input_args, _ = executables[prefill_engine.max_prefill_length].input_layouts
    prefill_param_layouts = input_args[0]
    new_prefill_params = _iterated_layout(prefill_params, prefill_param_layouts)
  else:
    new_prefill_params = prefill_params

  return (executables, new_prefill_params)


def _compile_generate_and_get_layouts(
    generate_engine: engine_api.JetStreamEngine,
    generate_params: engine_api.Params,
    generate_idx: int,
    decode_state: Optional[Any] = None,
    relayout_params_optimally: bool = False,
    relayout_decode_state_optimally: bool = False,
) -> tuple[Executable, Layout, Layout, Layout]:
  """Precompile generate function.

  Args:
    generate_engine: A generate engine to be compiled.
    generate_params: The associated generate parameters.
    generate_idx: Which generate engine it is.
    relayout_params_optimally: When set to true, modify the param layouts
      and re-layout the params to make them optimally laid out.
    relayout_decode_state_optimally: When set to true, optimize 
      decode state layout.
  Returns:
    A tuple containing 
      - compiled generate function
      - layout for generate function input params
      - layout for generate function input decode state
      - layout for generate function outpu decode state
  """  
  param_layout = Layout(DLL.AUTO if relayout_params_optimally else None)
  decode_state_layout = Layout(
      DLL.AUTO if relayout_decode_state_optimally else None
  )
  # TODO(wyzhang): Pass XLA flags
  executable = jax.jit(
      generate_engine.generate_aot,
      in_shardings=(param_layout, decode_state_layout),
      out_shardings=(Layout(DLL.AUTO), Layout(DLL.AUTO)),
      donate_argnums=(1,),
  ).lower(
      generate_params, decode_state
  ).compile(compiler_options=None)
  arg_layouts, _ = executable.input_layouts
  generated_out_layouts, _ = executable.output_layouts
  log.info(
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
        _parsed_pspec=sharding._parsed_pspec,
        # pylint: disable=protected-access
        _manual_axes=sharding._manual_axes,  # pylint: disable=protected-access
    )

  return jax.tree_util.tree_map(replace_sharding_devices, sharding)


def _get_transferred_prefix_destination_sharding(
    prefill_engine: engine_api.Engine,
    generate_engine: engine_api.Engine,
) -> Any:
  """Returns the sharding of the prefix destination upon transfer."""

  if prefill_engine.mesh.devices.size == generate_engine.mesh.devices.size:
    shardings = prefill_engine.get_prefix_destination_sharding()
    print(f'shardings\n {shardings}')
    return replace_devices_in_sharding(
        shardings,
        generate_engine.mesh.devices,
    )
  # TODO(b/345685171): Remove this warning once transfer with different number
  # of devices has a verified fast path.
  log.error(
      'Prefill and generate engines have different number of devices. '
      'Resharding will be extremely slow. Please use the same number of '
      'devices for prefill and generate engines unless you are sure about what '
      'you are doing.'
  )
  return generate_engine.get_prefix_destination_sharding()


def _initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    generate_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    generate_params: Any,
    generate_idx: int,
    relayout_params_optimally: bool = False,
    relayout_decode_state_optimally: bool = False,
) -> tuple[GenerateExecutables, engine_api.Params, Any]:
  """Initializes jit cache for insert and generate.

  Args:
      generate_engine: A generate engine to be compiled for.
      generate_params: The associated parameters.
      generate_idx: Which generate engine it is.
  """

  decode_state_shapes = jax.eval_shape(generate_engine.init_decode_state)
  generate_param_shapes = jax.tree.map(_to_shape_dtype, generate_params)
  (
      generate_executable,
      generate_params_layouts,
      _,
      decode_state_layouts,
  ) = _compile_generate_and_get_layouts(
      generate_engine,
      generate_param_shapes,
      generate_idx,
      decode_state_shapes,
      relayout_params_optimally,
      relayout_decode_state_optimally,
  )
  if relayout_params_optimally:
    generate_params = _iterated_layout(generate_params, generate_params_layouts)

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
    # print(f'prefix_shape {prefix_shape}')
    # prefix_dest_sharding = _get_transferred_prefix_destination_sharding(
    #     prefill_engine=prefill_engine,
    #     generate_engine=generate_engine,
    # )
    # print(f'get transfer prefix dest sharding {prefix_dest_sharding}')
    # prefix_shape = jax.tree.map(
    #     lambda x, sharding: None if x is None else jax.ShapeDtypeStruct(  # pylint: disable=g-long-lambda
    #         x.shape, x.dtype, sharding=sharding,
    #     ),
    #     prefix_shape,
    #     prefix_dest_sharding,
    #     is_leaf=lambda x: x is None,
    # )
    # print(f'prefix_shape post {prefix_shape}')
    slot_shape = jax.ShapeDtypeStruct((), jnp.int32)
    # TODO(wyzhang): Pass XLA flag
    insert_executable = jax.jit(
        generate_engine.insert,
        in_shardings=(None, decode_state_layouts, None),
        out_shardings=decode_state_layouts,
        donate_argnums=(1,),
    ).lower(
        prefix_shape, decode_state_shapes, slot_shape
    ).compile(compiler_options=None)
    log.info(
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
    decode_state_executable = jax.jit(
        generate_engine.init_decode_state,
        in_shardings=(None),
        out_shardings=(decode_state_layouts),
    ).lower(
    ).compile(compiler_options=None)
    log.info(
        "---Generate engine %d compiled for init decode state.---",
        generate_idx,
    )
    return decode_state_executable

  init_decode_state_executable = _compile_init_decode_state()

  generate_engine.aot = True
  return (
      (init_decode_state_executable, insert_executables, generate_executable),
      generate_params, generate_params_layouts,
  )
