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

"""Model server warmup utils."""

import jax.numpy as jnp
import concurrent.futures
from typing import Any, Optional
import logging
from jetstream.engine import engine_api, token_utils


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    interleaved_engines: Optional[
      list[tuple[engine_api.JetStreamEngine, engine_api.JetStreamEngine]]
    ] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
    shared_params: Optional[list[Any]] = None,
) -> tuple[
  list[engine_api.JetStreamEngine],
  list[engine_api.JetStreamEngine],
  list[engine_api.Params],
  list[engine_api.Params]
  ]:
  """Organizes the engines and executables.

  Args:
      prefill_engines: Prefill only engines.
      generate_engines: Generate only engines.
      interleaved_engines: Engines to be used for both.
      prefill_params: Prefill only params.
      generate_params: Generate only params.
      shared_params: Shared params for prefill and generate.
  """
  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  interleaved_engines = interleaved_engines if interleaved_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []
  shared_params = shared_params if shared_params else []

  out_prefill_engines = []
  out_generate_engines = []
  out_prefill_params = []
  out_generate_params = []

  any_prefill_engine = None
  any_prefill_params = None

  for i, pe in enumerate(prefill_engines):
    prefill_params_i = prefill_params[i]
    any_prefill_engine = pe
    any_prefill_params = prefill_params_i
    _ = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params_i,
        prefill_idx=i,
    )
    out_prefill_engines.append(pe)
    out_prefill_params.append(prefill_params_i)

  for i, ge in enumerate(generate_engines):
    generate_params_i = generate_params[i]
    _ = initialize_insert_generate_jit_cache(
        prefill_engine=any_prefill_engine,
        generate_engine=ge,
        prefill_params=any_prefill_params,
        generate_params=generate_params_i,
        generate_idx=i,
    )
    out_generate_engines.append(ge)
    out_generate_params.append(generate_params_i)

  for i, (pe, ge) in enumerate(interleaved_engines):
    shared_params_i = shared_params[i]
    _ = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=shared_params_i,
        prefill_idx=i,
    )
    _ = initialize_insert_generate_jit_cache(
        prefill_engine=pe,
        generate_engine=ge,
        prefill_params=shared_params_i,
        generate_params=shared_params_i,
        generate_idx=i,
    )
    out_prefill_engines.append(pe)
    out_generate_engines.append(ge)
    out_prefill_params.append(shared_params_i)
    out_generate_params.append(shared_params_i)
  return (
      out_prefill_engines,
      out_generate_engines,
      out_prefill_params,
      out_generate_params,
  )

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

    _, _ = prefill_engine._downstream_engine.prefill(  # pylint: disable=protected-access
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    logging.info(
        "---------Prefill engine %d compiled for prefill length %d.---------",
        prefill_idx,
        length,
    )

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = executor.map(compile_prefill, prefill_buckets)

  prefill_engine.warm = True

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_engine.warm


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

    generate_engine.insert(prefix=prefill, decode_state=decode_state, slot=0)

    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )

  def compile_generate():

    logging.info(
        "---------Generate compilation %d begun.---------", generate_idx
    )

    generate_engine._downstream_engine.generate(  # pylint: disable=protected-access
        params=generate_params,
        decode_state=decode_state,
    )

    logging.info(
        "---------Generate engine %d compiled.---------",
        generate_idx,
    )

    logging.info(
        "---------Generate compilation %d complete.---------", generate_idx
    )

  logging.info(
      "---------Insertion generation compilation %d begun.---------",
      generate_idx,
  )

  compile_generate()

  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = executor.map(compile_insert, prefill_buckets)

  generate_engine.warm = True

  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return generate_engine.warm
