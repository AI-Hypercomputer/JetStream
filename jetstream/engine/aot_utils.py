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

"""AOT compilation utils."""

import jax
import jax.numpy as jnp
import concurrent.futures
from typing import Any, Optional, cast
import logging
from jetstream.engine import engine_api, token_utils


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
) -> bool:
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

  any_prefill_engine = None
  any_prefill_params = None

  prefill_executables = []
  inserts_generate_executables = []

  for i, pe in enumerate(prefill_engines):
    any_prefill_engine = pe
    any_prefill_params = prefill_params[i]
    prefill_executable = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
    )
    prefill_executables.append(prefill_executable)

  for i, ge in enumerate(generate_engines):
    insert_executable, generate_executable = (
        initialize_insert_generate_jit_cache(
            prefill_engine=any_prefill_engine,
            generate_engine=ge,
            prefill_params=any_prefill_params,
            generate_params=generate_params[i],
            generate_idx=i,
        )
    )
    inserts_generate_executables.append(
        [insert_executable, generate_executable]
    )

  if prefill_executables and inserts_generate_executables:
    return True
  return False


def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    prefill_idx: int,
):
  """Precompile all prefill functions in parallel.
  If we don't do this, then when a new request triggers a new prefill bucket it
  will take a very long time for that query to come back.

  Args:
      prefill_engine: A prefill engine to be compiled for.
      prefill_params: The associated prefill parameters.
      prefill_idx: Which prefill engine it is.
  """
  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= prefill_engine.max_prefill_length
  ]
  prefill_engine.prefill_buckets = prefill_buckets
  if prefill_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(prefill_engine.max_prefill_length)

  def compile_prefill(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    lowered = jax.jit(
        prefill_engine._downstream_engine.prefill,  # pylint: disable=protected-access
        out_shardings=prefill_engine.get_prefix_destination_sharding(),
    ).lower(
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    logging.info(
        "---------Prefill engine %d lowered for prefill length %d.---------",
        prefill_idx,
        length,
    )
    compiled = lowered.compile()
    logging.info(
        "---------Prefill engine %d compiled for prefill length %d.---------",
        prefill_idx,
        length,
    )
    return compiled

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    prefill_executable = list(executor.map(compile_prefill, prefill_buckets))

  prefill_executable = {
      k: cast(jax.stages.Compiled, e)
      for k, e in zip(prefill_buckets, prefill_executable)
  }

  prefill_engine.prefill_executable = prefill_executable
  prefill_engine.warm = True

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_executable


def initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    generate_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    generate_params: Any,
    generate_idx: int,
):
  """Initialiszes jit cache for insert and generate.

  Args:
      generate_engine: A generate engine to be compiled for.
      generate_params: The associated parameters.
      generate_idx: Which generate engine it is.
  """

  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= generate_engine.max_prefill_length
  ]
  generate_engine.prefill_buckets = prefill_buckets
  if generate_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(generate_engine.max_prefill_length)

  decode_state = generate_engine.init_decode_state()

  def compile_insert(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    prefill, _ = prefill_engine._downstream_engine.prefill(  # pylint: disable=protected-access
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    lowered = jax.jit(generate_engine._downstream_engine.insert).lower(  # pylint: disable=protected-access
        prefix=prefill, decode_state=decode_state, slot=1
    )
    logging.info(
        "---------Generate engine %d lowered for insert length %d.---------",
        generate_idx,
        length,
    )
    compiled = lowered.compile()

    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )
    return compiled

  def compile_generate():

    logging.info(
        "---------Generate compilation %d begun.---------", generate_idx
    )

    lowered = jax.jit(generate_engine._downstream_engine.generate).lower(  # pylint: disable=protected-access
        params=generate_params,
        decode_state=decode_state,
    )
    logging.info(
        "---------Generate engine %d lowered.---------",
        generate_idx,
    )

    compiled = lowered.compile()
    logging.info(
        "---------Generate engine %d compiled.---------",
        generate_idx,
    )

    logging.info(
        "---------Generate compilation %d complete.---------", generate_idx
    )

    return compiled

  logging.info(
      "---------Insertion generation compilation %d begun.---------",
      generate_idx,
  )

  generate_executable = compile_generate()
  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )
  generate_engine.generate_executable = generate_executable

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    insert_executable = list(executor.map(compile_insert, prefill_buckets))

  insert_executable = {
      k: cast(jax.stages.Compiled, e)
      for k, e in zip(prefill_buckets, insert_executable)
  }
  generate_engine.insert_executable = insert_executable
  generate_engine.warm = True

  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return insert_executable, generate_executable
