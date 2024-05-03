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
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from seqio.vocabularies import SentencePieceVocabulary
from seqio.vocabularies import Vocabulary

from jetstream.engine import mock_utils
from jetstream.engine import tokenizer_api
from jetstream.engine import tokenizer_pb2
from jetstream.third_party.llama3 import llama3_tokenizer

# ResultToken class to store tokens ids.
ResultTokens = Any


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
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
  """Tokenize and pads a string.

  Args:
    s: String to tokenize.
    vocab: Vocabulary to tokenize with.
    is_bos: Whether or not this is the beginning of a sequence. Default to yes
      as prefill is typically used when beginning sequences.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.
    jax_padding: convert to JAX padded tokens if True.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """

  tokens = np.array(vocab.encode_tf(s))  # [Length]
  bos_id = vocab.bos_id
  pad_id = vocab.pad_id
  assert pad_id == 0, "Further logic required if pad_id not 0."

  padded_tokens, true_length = pad_tokens(
      tokens=tokens,
      bos_id=bos_id,
      pad_id=pad_id,
      is_bos=is_bos,
      prefill_lengths=prefill_lengths,
      max_prefill_length=max_prefill_length,
      jax_padding=jax_padding,
  )
  return padded_tokens, true_length


def pad_tokens(
    tokens: np.ndarray,
    bos_id: int,
    pad_id: int,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
  """Tokenize and pads a string.

  Args:
    s: String to tokenize.
    vocab: Vocabulary to tokenize with.
    is_bos: Whether or not this is the beginning of a sequence. Default to yes
      as prefill is typically used when beginning sequences.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.
    jax_padding: convert to JAX padded tokens if True.

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
  # Add a beginning of sequence token if this is the beginning.
  if is_bos:
    tokens = np.concatenate(
        [
            np.array(
                [
                    bos_id,
                ]
            ),
            tokens,
        ],
        axis=-1,
    )
  true_length = tokens.shape[-1]
  padded_length = take_nearest_length(prefill_lengths, true_length)
  padding = padded_length - true_length
  if padding < 0:
    logging.warning("Provided sequence longer than available.")
    # Take the last N tokens if we have too many.
    padded_tokens = tokens[-padded_length:]
  else:
    padded_tokens = np.pad(tokens, (0, padding), constant_values=(pad_id,))
  if jax_padding:
    padded_tokens = jnp.array(padded_tokens)
  return padded_tokens, true_length


def process_result_tokens(
    slot: int,
    slot_max_length: int,
    result_tokens: ResultTokens,
    eos_id: int,
    pad_id: int,
    complete: np.ndarray,
    debug: bool = False,
) -> Tuple[List[List[int]], np.ndarray]:
  """Processes a result tokens into a list of strings, handling multiple
    samples.

  Args:
    slot: The slot at which to draw tokens from.
    slot_max_length: Max length for a sample in the slot.
    result_tokens: The tokens to access by slot.
    eos_id: Id for EOS token.
    pad_id: Id for pad token.
    complete: Array representing the completion status of each sample in the
      slot.
    debug: Whether to log step by step detokenisation.

  Returns:
    sample_return: List of tok_id list, one list per sample.
    complete: Updated complete.
  """
  # tokens: [samples, speculations]
  slot_data = result_tokens.get_result_at_slot(slot)
  slot_tokens = slot_data.tokens
  slot_valid = slot_data.valid
  slot_lengths = slot_data.lengths
  samples, speculations = slot_tokens.shape
  stop_tokens = [eos_id, pad_id]
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
    tok_id_so_far = []
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
          tok_id_so_far.append(tok_id)
    sample_return.append(tok_id_so_far)
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


class SentencePieceTokenizer(tokenizer_api.Tokenizer):
  """Tokenizer to convert strings to token ids and vice-versa."""

  def __init__(self, metadata: tokenizer_pb2.TokenizerParameters):
    self.vocab = load_vocab(metadata.path, metadata.extra_ids)

  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """
    is_bos = kwargs.pop("is_bos", True)
    prefill_lengths = kwargs.pop("prefill_lengths", None)
    max_prefill_length = kwargs.pop("max_prefill_length", None)
    jax_padding = kwargs.pop("jax_padding", True)

    tokens = np.array(self.vocab.encode_tf(s))

    tokens, true_length = pad_tokens(
        tokens,
        self.bos_id,
        self.pad_id,
        is_bos=is_bos,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=jax_padding,
    )
    return tokens, true_length

  def decode(
      self,
      slot: int,
      slot_max_length: int,
      result_tokens: ResultTokens,
      complete: np.ndarray,
      **kwargs,
  ) -> Tuple[List[List[int]], np.ndarray]:
    """Processes a result tokens into a list of strings, handling multiple
    samples.
    Args:
      slot: The slot at which to draw tokens from.
      slot_max_length: Max length for a sample in the slot.
      result_tokens: The tokens to access by slot.
      complete: Array representing the completion status of each sample in the
        slot.
      kwargs: Additional keyword arguments.
    Returns:
      sample_return: List of strings, one per sample.
      complete: Updated complete.
    """
    debug = kwargs.pop("debug", False)
    results, complete = process_result_tokens(
        slot=slot,
        slot_max_length=slot_max_length,
        result_tokens=result_tokens,
        eos_id=self.eos_id,
        pad_id=self.pad_id,
        complete=complete,
        debug=debug,
    )
    return results, complete

  def decode_str(self, token_ids: list[int]) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
    Returns:
      str: String generated from the token ids.
    """
    return self.vocab.tokenizer.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.vocab.pad_id

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.vocab.eos_id

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.vocab.bos_id


class TikToken(tokenizer_api.Tokenizer):
  """Tokenizer to convert strings to token ids and vice-versa."""

  def __init__(self, metadata: tokenizer_pb2.TokenizerParameters):
    self.tokenizer = llama3_tokenizer.Tokenizer(metadata.path)

  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """
    is_bos = kwargs.pop("is_bos", True)
    prefill_lengths = kwargs.pop("prefill_lengths", None)
    max_prefill_length = kwargs.pop("max_prefill_length", None)
    jax_padding = kwargs.pop("jax_padding", True)

    tokens = np.array(self.tokenizer.encode(s, bos=False, eos=False))

    tokens, true_length = pad_tokens(
        tokens,
        self.bos_id,
        self.pad_id,
        is_bos=is_bos,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=jax_padding,
    )
    return tokens, true_length

  def decode(
      self,
      slot: int,
      slot_max_length: int,
      result_tokens: ResultTokens,
      complete: np.ndarray,
      **kwargs,
  ) -> Tuple[List[List[int]], np.ndarray]:
    """Processes a result tokens into a list of strings, handling multiple
    samples.
    Args:
      slot: The slot at which to draw tokens from.
      slot_max_length: Max length for a sample in the slot.
      result_tokens: The tokens to access by slot.
      complete: Array representing the completion status of each sample in the
        slot.
      kwargs: Additional keyword arguments.
    Returns:
      sample_return: List of strings, one per sample.
      complete: Updated complete.
    """
    debug = kwargs.pop("debug", False)
    results, complete = process_result_tokens(
        slot=slot,
        slot_max_length=slot_max_length,
        result_tokens=result_tokens,
        eos_id=self.eos_id,
        pad_id=self.pad_id,
        complete=complete,
        debug=debug,
    )
    return results, complete

  def decode_str(self, token_ids: list[int]) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
    Returns:
      str: String generated from the token ids.
    """
    return self.tokenizer.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.tokenizer.pad_id

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.tokenizer.eos_id

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.tokenizer.bos_id
