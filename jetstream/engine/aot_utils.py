"""AOT compilation utils."""

import jax
import threading
import jax.numpy as jnp
import concurrent.futures
from typing import Any, Optional
from collections.abc import Callable
from absl import logging
import numpy as np
from jetstream.engine import engine_api, token_utils
from seqio.vocabularies import Vocabulary

# Jax python tracing can't be parallelized since it's python execution.  Running
# many tracings parallelly leads to thread switching overhead.  So we run those
# parts serially and compile in parallel.
_JAX_TRACING_LOCK = threading.Lock()

ExecutablesMap = dict[int, jax.stages.Compiled]

PrefillExecutables = tuple[
    # Prefill for each input length without history.
    dict[int, ExecutablesMap],
    # Prefill for each input length with history.
    dict[int, ExecutablesMap],
]

GenerateExecutables = tuple[
    # Insert executables for each input length.
    ExecutablesMap,
    # Generate step executable.
    jax.stages.Compiled,
    # Decode state init executable.
    jax.stages.Compiled,
]

def _make_shaped_array(t):
  if hasattr(t, 'sharding'):
    return jax.ShapeDtypeStruct(t.shape, t.dtype, sharding=t.sharding)
  else:
    return jax.ShapeDtypeStruct(t.shape, t.dtype)

def layout_params_and_compile_executables(
    prefill_buckets: list[int],
    prefill_engines: Optional[list[engine_api.Engine]] = None,
    generate_engines: Optional[list[engine_api.Engine]] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
) -> tuple[
    list[engine_api.Engine],
    list[engine_api.Engine],
    list[engine_api.Params],
    list[engine_api.Params],
    list[PrefillExecutables],
    list[GenerateExecutables],
]:
    """Organizes the engines and executables.

    Args:
        prefill_buckets: Buckets to compile for.
        prefill_engines: Prefill only engines.
        generate_engines: Generate only engines.
        prefill_params: Prefill only params.
        generate_params: Generate only params.

    Returns:
        prefill_engines: Combined prefill + interleaved engines.
        generate_engines: Combined generate + interleaved engines.
        prefill_params: Combined prefill + shared params.
        generate_params: Combined generate + shared params.
        prefill_executables: Executables to be used in prefill threads.
        generate_executables: Executables to be used in generated threads.
        metrics: Dict of how long each stage took to happen.
    """
    prefill_engines = prefill_engines if prefill_engines else []
    generate_engines = generate_engines if generate_engines else []
    prefill_params = prefill_params if prefill_params else []
    generate_params = generate_params if generate_params else []

    

    compiled_prefills = []
    generate_executables = []

    # We expect
    # 1. all prefill engines are the same except for the device ids
    # 2. in separate slices setup, at least one prefill engine is provided
    # so picking up any one of them is equivalent.
    any_prefill_engine = None
    any_prefill_params = None

    # any_prefill = None
    # any_generate_params = None
    # any_decode_state = None

    for i, pe in enumerate(prefill_engines):
        any_prefill_engine = pe
        any_prefill_params = prefill_params[i]

        prefill_compiled = initialize_prefill_jit_cache(
            prefill_engine=pe,
            prefill_params=prefill_params[i],
            prefill_idx=i,
            prefill_buckets=prefill_buckets,
        )
        compiled_prefills.append(prefill_compiled)
    print(compiled_prefills)
    
    for i, ge in enumerate(generate_engines):
        initialize_insert_jit_cache(
            prefill_engine=any_prefill_engine,
            generate_engine=ge,
            generate_params=generate_params[i],
            prefill_params=any_prefill_params,
            generate_idx=i,
            prefill_buckets=prefill_buckets,
        )
    # return prefill_

    # for i, ge in enumerate(generate_engines):
    #     initialize_generate_jit_cache(
    #         prefill=any_prefill,
    #         generate_params=any_generate_params,
    #         decode_state=any_decode_state,
    #         generate_idx=i,
    #     )

def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.Engine,
    prefill_params: Any,
    prefill_idx: int,
    prefill_buckets: list[int] = token_utils.DEFAULT_PREFILL_BUCKETS,
):
    """Precompile all prefill functions in parallel.
    If we don't do this, then when a new request triggers a new prefill bucket it
    will take a very long time for that query to come back.

    Args:
        prefill_engine: A prefill engine to be compiled for.
        prefill_params: The associated prefill parameters.
        vocab: The associated vocabulary.
        prefill_idx: Which prefill engine it is.
        prefill_buckets: Buckets to compile for.
    """

    prefill_buckets = [
        bucket
        for bucket in prefill_buckets
        if bucket <= prefill_engine.max_prefill_length
    ]
    if prefill_engine.max_prefill_length not in prefill_buckets:
        prefill_buckets.append(prefill_engine.max_prefill_length)

    # param_shapes = jax.tree_map(_make_shaped_array, prefill_params)

    # param_shapes = jax.tree_map(make_shaped_array, prefill_params)

    def compile_prefill(length, history=None):
        # text="AB"
        # metadata = prefill_engine.get_tokenizer()
        # tokenizer = prefill_engine.build_tokenizer(metadata)
        # tokens, true_length = tokenizer.encode(text, is_bos=True)
        # # vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
        # # padded_tokens, true_length = token_utils.tokenize_and_pad(
        # #     'Test data',
        # #     vocab=vocab,
        # #     max_prefill_length=length,
        # # )
        # # token_shapes = jax.tree_map(make_shaped_array, padded_tokens)
        # prefill_result = prefill_engine.prefill(
        #     params=prefill_params,
        #     padded_tokens=tokens,
        #     true_length=3,
        # )

        # logging.info(
        #     '---------Prefill engine %d prefilled for prefill length %d.---------',
        #     prefill_idx,
        #     length,
        # )

        # return prefill_result

        metadata = prefill_engine.get_tokenizer()
        vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
        padded_tokens, true_length = token_utils.tokenize_and_pad(
            "Example text, often referred to as lorem ipsum, is placeholder content used by designers and developers in the layout of documents and websites. It's a scrambled Latin passage that mimics the rhythm and flow of real text, allowing for accurate visualization of fonts, spacing, and formatting. This nonsensical text helps maintain focus on the visual aspects without distraction from actual content. Lorem ipsum has become a standard in the industry, appearing in countless projects as a temporary stand-in before the final text is incorporated.",
            vocab=vocab,
            max_prefill_length=length,
        )

        lowered = jax.jit(prefill_engine.prefill, out_shardings=prefill_engine.get_prefix_destination_sharding()).lower(
            params=prefill_params,
            existing_prefix=history,
            padded_tokens=padded_tokens,
            true_length=true_length,
        )
        logging.info(
            '---------Prefill engine %d lowered for prefill length %d.---------',
            prefill_idx,
            length,
        )
        compiled = lowered.compile()
        logging.info(
            '---------Prefill engine %d compiled for prefill length %d.---------',
            prefill_idx,
            length,
        )

        # prefill_engine.prefill = compiled

        # prefill_engine.prefill = compiled(
        #     params=prefill_params,
        #     existing_prefix=history,
        #     padded_tokens=padded_tokens,
        #     true_length=true_length
        # )

        prefill_compiled[length] = compiled(
            params=prefill_params,
            existing_prefix=history,
            padded_tokens=padded_tokens,
            true_length=true_length
        )

        return compiled
    logging.info(
      '---------Prefill compilation %d begun.---------', prefill_idx
    )
    # First do the prefills without history, as the majority of requests will care
    # about this.

    prefill_compiled = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prefill_buckets)
    ) as executor:
        _ = executor.map(compile_prefill, prefill_buckets)

    print(prefill_compiled)

    prefill_engine.prefill_compiled = prefill_compiled

    logging.info(
      '---------Prefill compilation %d complete.---------', prefill_idx
    )

    return prefill_compiled

def initialize_insert_jit_cache(
    *,
    prefill_engine: engine_api.Engine,
    generate_engine: engine_api.Engine,
    generate_params: Any,
    prefill_params: Any,
    generate_idx: int,
    prefill_buckets: list[int] = token_utils.DEFAULT_PREFILL_BUCKETS,
):
    """Initialiszes jit cache for insert.

    Args:
        generate_engine: A generate engine to be compiled for.
        generate_params: The associated parameters.
        generate_idx: Which generate engine it is.
        prefill_buckets: Buckets to compile for. Insertion inserts prefixes computed
        from padded prefill sequences.
    """

    prefill_buckets = [
        bucket
        for bucket in prefill_buckets
        if bucket <= generate_engine.max_prefill_length
    ]

    if generate_engine.max_prefill_length not in prefill_buckets:
        prefill_buckets.append(generate_engine.max_prefill_length)

    # compile insert needs the following:
    # 1. need to prefill
    # 2. need to get the decode state
    # 3. need the slot, slot can just be 1 since we just need to generate once

    # compile generate just needs to lower and compile

    # any_prefill = None
    # any_generate_params = generate_params
    # any_decode_state = None

    def compile_insert(length):
        metadata = prefill_engine.get_tokenizer()
        vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

        padded_tokens, true_length = token_utils.tokenize_and_pad(
            "Example text, often referred to as lorem ipsum, is placeholder content used by designers and developers in the layout of documents and websites. It's a scrambled Latin passage that mimics the rhythm and flow of real text, allowing for accurate visualization of fonts, spacing, and formatting. This nonsensical text helps maintain focus on the visual aspects without distraction from actual content. Lorem ipsum has become a standard in the industry, appearing in countless projects as a temporary stand-in before the final text is incorporated.",
            vocab=vocab,
            max_prefill_length=length,
        )

        prefill = prefill_engine.prefill(
            params=prefill_params,
            existing_prefix=None,
            padded_tokens=padded_tokens,
            true_length=true_length,
        )
        # any_prefill = prefill

        decode_state = generate_engine.init_decode_state()
        # any_decode_state = decode_state
        lowered = jax.jit(generate_engine.insert).lower(
            prefix=prefill,
            decode_state=decode_state,
            slot=1
        )
        logging.info(
            '---------Generate engine %d lowered for insert length %d.---------',
            generate_idx,
            length,
        )
        compiled = lowered.compile()

        insert_compiled[length] = compiled(
            prefix=prefill,
            decode_state=decode_state,
            slot=1
        )

        logging.info(
            '---------Generate engine %d compiled for insert length %d.---------',
            generate_idx,
            length,
        )

        # compiled(
        #     prefix=prefill,
        #     decode_state=decode_state,
        #     slot=1
        # )

        decode_state_insert = generate_engine.insert(
            prefix=prefill,
            decode_state=decode_state,
            slot=1
        )

        lowered = jax.jit(generate_engine.generate).lower(
            params=generate_params,
            decode_state=decode_state_insert,
        )

        logging.info(
            '---------Generate engine %d lowered for generate length %d.---------',
            generate_idx,
            length,
        )
        compiled = lowered.compile()

        # compiled(
        #     params=generate_params,
        #     decode_state=decode_state_insert,
        # )

        logging.info(
            '---------Generate engine %d compiled for generate length %d.---------',
            generate_idx,
            length,
        )

        return compiled

    def compile_generate():

        logging.info(
        '---------Generate compilation %d begun.---------', generate_idx)

        decode_state = generate_engine.init_decode_state()
        decode_state_insert = generate_engine.insert(
            prefix=any_prefill,
            decode_state=decode_state,
            slot=1
        )
        print("reached here")

        lowered = jax.jit(generate_engine.generate).lower(
            params=generate_params,
            decode_state=decode_state_insert,
        )
        print("reached here 1")
        logging.info(
            '---------Generate engine %d lowered for generate length %d.---------',
            generate_idx,
            length,
        )

        compiled = lowered.compile()
        print("reached here 2")
        logging.info(
            '---------Generate engine %d compiled for generate length %d.---------',
            generate_idx,
            length,
        )

        logging.info(
            '---------Generate compilation %d complete. %d.---------',
            generate_idx
        )

        return compiled

    logging.info(
        '---------Insertion generation compilation %d begun.---------', generate_idx
    )

    insert_compiled = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prefill_buckets)
    ) as executor:
        _ = list(executor.map(compile_insert, prefill_buckets))

    print(insert_compiled)

    generate_engine.insert_compiled = insert_compiled
    logging.info(
        '---------Insertion generation compilation %d complete.---------', generate_idx
    )

    # return any_prefill, any_generate_params, any_decode_state


def initialize_generate_jit_cache(
    *,
    prefill: Any,
    generate_params: Any,
    decode_state: Any,
    generate_idx: int,
):

    def compile_generate():
        decode_state_insert = generate_engine.insert(
            prefix=prefill,
            decode_state=decode_state,
            slot=1
        )

        lowered = jax.jit(generate_engine.generate).lower(
            params=generate_params,
            decode_state=decode_state_insert,
        )

        logging.info(
            '---------Generate engine %d lowered for generate length %d.---------',
            generate_idx,
            length,
        )
        compiled = lowered.compile()

        logging.info(
            '---------Generate engine %d compiled for generate length %d.---------',
            generate_idx,
            length,
        )

    logging.info(
        '---------Generation compilation %d begun.---------', generate_idx
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    ) as executor:
        _ = executor.submit(compile_generate)
    logging.info(
        '---------Generation compilation %d complete.---------', generate_idx
    )