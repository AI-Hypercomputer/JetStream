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

"""Chunked prefill."""

import jax
import numpy as np
from typing import Any, Optional

from flax import struct

from jetstream.engine import engine_api
from jetstream.engine import tokenizer_api
from jetstream.engine import token_utils


@struct.dataclass
class ChunkedTokens:
  """Chunked tokens.

  Attributes:
    padded_input_tokens: Padded input tokens.
    input_true_lengths: True length of the input tokens.
    common_prefix_tokens_after_prefill: Common prefix tokens after prefill.
  """

  padded_input_tokens: jax.Array | np.ndarray
  input_true_lengths: int
  common_prefix_tokens_after_prefill: jax.Array | np.ndarray


def gen_chunked_padded_tokens(
    tokens: jax.Array,
    chunk_size: int,
    tokenizer: tokenizer_api.Tokenizer,
    existing_prefix_tokens: Optional[jax.Array] = None,
    jax_padding: bool = True,
) -> list[ChunkedTokens]:
  """Generates padded token chunks from an input token sequence.

  This function takes a sequence of tokens, optionally prepends existing prefix
  tokens, and then splits the sequence into chunks of a specified size.
  Each chunk is then padded.

  All the tokens will padding to the chunk_size.

  Args:
    tokens: The input sequence of token IDs..
    chunk_size: The target size for each chunk.
    tokenizer: A tokenizer.
    existing_prefix_tokens:
      Existing tokens prepend to the common prefix tokens.
      Length should be multiple of chunk size.
    jax_padding: Convert to JAX padded tokens if True.

  Returns:
    A list of `ChunkedTokens` objects. Each object represents a chunk of the
    original sequence, padded according to the rules described above. The
    exact structure of `ChunkedTokens` should contain at least the padded
    token array.
  """

  chunked_tokens_list: list[ChunkedTokens] = []
  for cur_chunk_start_pos in range(0, len(tokens), chunk_size):
    input_token = tokens[
        cur_chunk_start_pos : min(len(tokens), cur_chunk_start_pos + chunk_size)
    ]
    padded_input_token, input_true_length = token_utils.pad_tokens(
        input_token,
        tokenizer.bos_id,
        tokenizer.pad_id,
        is_bos=False,
        prefill_lengths=[chunk_size],
        max_prefill_length=chunk_size,
        jax_padding=jax_padding,
    )
    common_prefix_tokens = tokens[
        0 : min(len(tokens), cur_chunk_start_pos + chunk_size)
    ]
    if existing_prefix_tokens is not None:
      common_prefix_tokens = jax.numpy.concatenate(
          [
              existing_prefix_tokens,
              common_prefix_tokens,
          ],
      )
    chunked_tokens_list.append(
        ChunkedTokens(
            padded_input_tokens=padded_input_token,
            input_true_lengths=input_true_length,
            common_prefix_tokens_after_prefill=common_prefix_tokens,
        )
    )

  return chunked_tokens_list


def do_chunked_prefill(
    prefill_engine: engine_api.Engine,
    prefill_params: Any,
    chunked_tokens_list: list[ChunkedTokens],
    existing_prefix: Optional[engine_api.ExistingPrefix] = None,
) -> tuple[engine_api.Prefix, engine_api.ResultTokens]:
  """Do chunked prefill.

  This function performs prefill in chunks, processing each chunk sequentially.
  It is designed for scenarios where the input sequence is too long to be
  processed in a single prefill operation.

  Args:
    prefill_engine: The prefill engine to use for processing.
    prefill_params: The parameters to pass to the prefill engine.
    chunked_tokens_list: A list of `ChunkedTokens` objects, each representing a
      chunk of the input sequence.
    existing_prefix: An optional existing prefix to prepend to the input
      sequence.

  Returns:
    A tuple containing the final prefill result and the first token of the
    last chunk.
  """

  if not prefill_engine.use_chunked_prefill:
    raise ValueError("Chunked prefill is not enabled.")

  if not chunked_tokens_list:
    raise ValueError("No chunked tokens provided.")

  prefill_result = None
  first_token = None
  existing_prefix_now = existing_prefix
  for cur_chunk in chunked_tokens_list:
    padded_input_token = cur_chunk.padded_input_tokens
    input_true_length = cur_chunk.input_true_lengths
    prefill_result, first_token = prefill_engine.prefill(
        params=prefill_params,
        existing_prefix=existing_prefix_now,
        padded_tokens=padded_input_token,
        true_length=input_true_length,
    )
    existing_prefix_now = engine_api.ExistingPrefix(
        cache=prefill_result["cache"],
        common_prefix_tokens=cur_chunk.common_prefix_tokens_after_prefill,
    )

  # Should assign in the loop
  assert prefill_result is not None
  assert first_token is not None

  return prefill_result, first_token
