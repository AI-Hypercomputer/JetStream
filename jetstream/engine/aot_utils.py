"""AOT compilation utils."""

import jax
import concurrent.futures
from typing import Any, Optional
from absl import logging
from jetstream.engine import engine_api, token_utils


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.Engine]] = None,
    generate_engines: Optional[list[engine_api.Engine]] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
) -> bool:
  """Organizes the engines and executables.

  Args:
      prefill_engines: Prefill only engines.
      generate_engines: Generate only engines.
      prefill_params: Prefill only params.
      generate_params: Generate only params.

  Returns:
      bool:
  """
  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []

  compiled_prefills = []
  compiled_inserts_generate = []

  for i, pe in enumerate(prefill_engines):
    prefill_compiled = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
    )
    compiled_prefills.append(prefill_compiled)

  for i, ge in enumerate(generate_engines):
    insert_compiled, generate_compiled = initialize_insert_generate_jit_cache(
        generate_engine=ge,
        generate_params=generate_params[i],
        generate_idx=i,
    )
    compiled_inserts_generate.append([insert_compiled, generate_compiled])

  if compiled_prefills and compiled_inserts_generate:
    return True
  return False


def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.Engine,
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
    metadata = prefill_engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        "Example text, often referred to as lorem ipsum, is placeholder content used by designers and developers in the layout of documents and websites. It's a scrambled Latin passage that mimics the rhythm and flow of real text, allowing for accurate visualization of fonts, spacing, and formatting. This nonsensical text helps maintain focus on the visual aspects without distraction from actual content. Lorem ipsum has become a standard in the industry, appearing in countless projects as a temporary stand-in before the final text is incorporated.", # pylint: disable=line-too-long
        vocab=vocab,
        max_prefill_length=length,
    )

    lowered = jax.jit(
        prefill_engine.prefill,
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
    prefill_compiled[length] = compiled

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

  prefill_compiled = {}
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = executor.map(compile_prefill, prefill_buckets)

  prefill_engine.prefill_compiled = prefill_compiled

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_compiled


def initialize_insert_generate_jit_cache(
    *,
    generate_engine: engine_api.Engine,
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
    metadata = generate_engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

    padded_tokens, true_length = token_utils.tokenize_and_pad(
        "Example text, often referred to as lorem ipsum, is placeholder content used by designers and developers in the layout of documents and websites. It's a scrambled Latin passage that mimics the rhythm and flow of real text, allowing for accurate visualization of fonts, spacing, and formatting. This nonsensical text helps maintain focus on the visual aspects without distraction from actual content. Lorem ipsum has become a standard in the industry, appearing in countless projects as a temporary stand-in before the final text is incorporated.", # pylint: disable=line-too-long
        vocab=vocab,
        max_prefill_length=length,
    )

    prefill = generate_engine.prefill(
        params=generate_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    lowered = jax.jit(generate_engine.insert).lower(
        prefix=prefill, decode_state=decode_state, slot=1
    )
    logging.info(
        "---------Generate engine %d lowered for insert length %d.---------",
        generate_idx,
        length,
    )
    compiled = lowered.compile()
    insert_compiled[length] = compiled

    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )

  def compile_generate():

    logging.info(
        "---------Generate compilation %d begun.---------", generate_idx
    )

    lowered = jax.jit(generate_engine.generate).lower(
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

  generate_compiled = compile_generate()
  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )
  generate_engine.generate_compiled = generate_compiled

  insert_compiled = {}
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = list(executor.map(compile_insert, prefill_buckets))

  generate_engine.insert_compiled = insert_compiled
  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return insert_compiled, generate_compiled
