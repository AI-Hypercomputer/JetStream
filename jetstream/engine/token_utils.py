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

"""Token manipulation utilities."""

from bisect import bisect_left
import logging
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from seqio.vocabularies import SentencePieceVocabulary
from seqio.vocabularies import Vocabulary

from jetstream.engine import engine_api
from jetstream.engine import mock_utils


def mix_decode(vocab: Vocabulary, tok_id: int):
  """
  The IdToPiece and decode results differ for 344 tokens in Llama2.
  Use the decode function to generate the correct strings for these 344 tokens.
  If IdToPiece returns a hex string (e.g., '<0x0A>') for a token within these
    344, utilize IdToPiece to convert it into a string, likely with a space
    placeholder (' ') for the corresponding tokens.
  """
  p_token = vocab.tokenizer.IdToPiece(tok_id)
  # SentencePiece escapes the whitespace with a meta symbol "▁" (U+2581)
  p_token = p_token.replace("▁", " ")
  d_token = vocab.tokenizer.decode([tok_id])
  return p_token if p_token.lstrip() == d_token else d_token


def take_nearest_length(lengths: list[int], length: int) -> int:
  """Gets the nearest length to the right in a set of lengths."""
  pos = bisect_left(lengths, length)
  if pos == len(lengths):
    return lengths[-1]
  return lengths[pos]


def tokenize_and_pad(
    s: str,
    vocab: Vocabulary,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
) -> Tuple[jax.Array, int]:
  """Tokenize and pads a string.

  Args:
    s: String to tokenize.
    vocab: Vocabulary to tokenize with.
    is_bos: Whether or not this is the beginning of a sequence. Default to yes
      as prefill is typically used when beginning sequences.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """
  if prefill_lengths is None:
    prefill_lengths = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]
  if max_prefill_length is not None:
    prefill_lengths = prefill_lengths[
        : prefill_lengths.index(max_prefill_length)
    ] + [
        max_prefill_length,
    ]
  tokens = np.array(vocab.encode_tf(s))  # [Length]
  # Add a beginning of sequence token if this is the beginning.
  if is_bos:
    tokens = np.concatenate(
        [
            np.array(
                [
                    vocab.bos_id,
                ]
            ),
            tokens,
        ],
        axis=-1,
    )
  true_length = tokens.shape[-1]
  padded_length = take_nearest_length(prefill_lengths, true_length)
  padding = padded_length - true_length
  assert vocab.pad_id == 0, "Further logic required if pad_id not 0."
  if padding < 0:
    logging.warning("Provided sequence longer than available.")
    # Take the last N tokens if we have too many.
    padded_tokens = tokens[-padded_length:]
  else:
    padded_tokens = np.pad(tokens, (0, padding))
  return jnp.array(padded_tokens), true_length


def process_result_tokens(
    slot: int,
    slot_max_length: int,
    result_tokens: engine_api.ResultTokens,
    vocab: Vocabulary,
    complete: np.ndarray,
    debug: bool = False,
) -> Tuple[List[str], np.ndarray]:
  """Processes a result tokens into a list of strings, handling multiple
    samples.

  Args:
    slot: The slot at which to draw tokens from.
    slot_max_length: Max length for a sample in the slot.
    result_tokens: The tokens to access by slot.
    vocab: For the detokenizer.
    complete: Array representing the completion status of each sample in the
      slot.
    debug: Whether to log step by step detokenisation.

  Returns:
    sample_return: List of strings, one per sample.
    complete: Updated complete.
  """
  # tokens: [samples, speculations]
  slot_data = result_tokens.get_result_at_slot(slot)
  slot_tokens = slot_data.tokens
  slot_valid = slot_data.valid
  slot_lengths = slot_data.lengths
  samples, speculations = slot_tokens.shape
  stop_tokens = [vocab.eos_id, vocab.pad_id]
  # Stop anything which has reached it's max length.
  complete = complete | (slot_lengths > slot_max_length)
  if debug:
    logging.info(
        "Complete %s, slot_tokens: %s, slot_lengths: %s",
        str(complete),
        str(slot_tokens),
        str(slot_lengths),
    )
  sample_return = []
  for idx in range(samples):
    string_so_far = ""
    if not complete[idx].item():
      for spec_idx in range(speculations):
        tok_id = slot_tokens[idx, spec_idx].item()
        valid = slot_valid[idx, spec_idx].item()
        if debug:
          logging.info(
              "Sample idx: %d Speculation idx: %d Token: %d",
              idx,
              spec_idx,
              tok_id,
          )
        if tok_id in stop_tokens or not valid:
          complete[idx] = True
          break
        else:
          try:
            token = mix_decode(vocab, tok_id)  # pytype: disable=attribute-error
          except ValueError:
            # This error only occurs when using tests where the vocab range is
            # computed via addition and int->char is computed using chr(). Real
            # models have vocab logits which are at max the size of the vocab.
            logging.warning("%d exceeded vocab range", tok_id)
            token = "<sampled_outside_vocab>"
          string_so_far += token
    sample_return.append(string_so_far)
    if debug:
      logging.info("Sampled return %s", str(sample_return))
  return sample_return, complete


def load_vocab(path: str, extra_ids: int = 0) -> Vocabulary:
  """Eagerly loads a vocabulary.

  Args:
    path: Vocabulary file path.
    extra_ids: Number of extra IDs.

  Returns:
    A seqio Vocabulary.
  """
  if path == "test":
    return mock_utils.TestVocab()
  else:
    vocab = SentencePieceVocabulary(
        path,
        extra_ids=extra_ids,
    )
    # SentencePieceVocabulary uses lazy loading. Request access to a property,
    # forcing the lazy loading to happen now.
    sp_model = vocab.sp_model
    del sp_model
    return vocab
