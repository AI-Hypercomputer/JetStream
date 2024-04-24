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

"""Tests functionality of the tokenizer with supported models."""

import os
import unittest
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from sentencepiece import SentencePieceProcessor
from jetstream.engine import tokenizer_pb2, token_utils


class SPTokenizer:
  """Tokenier used in original llama2 git"""

  def __init__(self, tokenizer_path: str):
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.Load(model_file=tokenizer_path)
    assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

  def decode(self, t: List[int]) -> str:
    token = self.tokenizer.decode(t)
    return token


class JetStreamTokenizer:
  """Tokenier used in JetStream before mix_token"""

  def __init__(self, tokenizer_path: str):
    metadata = tokenizer_pb2.TokenizerParameters(path=tokenizer_path)
    self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  def decode(self, t: int) -> str:
    token = self.vocab.tokenizer.IdToPiece(t)
    token = token.replace("▁", " ")
    return token


class TokenUtilsTest(unittest.TestCase):

  def setup(self):
    tokenizer_path = "third_party/llama2/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(current_dir, tokenizer_path)
    print(f"model_path: {tokenizer_path}")
    assert os.path.isfile(
        tokenizer_path
    ), f"file not found tokenizer_path: {tokenizer_path}"
    self.sp_tokenizer = SPTokenizer(tokenizer_path)
    self.jt_tokenizer = JetStreamTokenizer(tokenizer_path)

  def test_decode_vs_piece(self):
    self.setup()
    tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
    expeted_sp_output = []
    jt_output = []
    for t in tokens:
      expeted_sp_output.append(self.sp_tokenizer.decode([t]))
      jt_output.append(self.jt_tokenizer.decode(t))

    self.assertNotEqual(jt_output, expeted_sp_output)

  def test_sp_vs_seqio(self):
    self.setup()
    for n in range(0, self.sp_tokenizer.tokenizer.vocab_size()):
      sp_t = self.sp_tokenizer.decode([n])
      seqio_t = self.jt_tokenizer.vocab.tokenizer.decode([n])
      self.assertEqual(sp_t, seqio_t)

  def test_tokenize_and_pad_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        s=s,
        vocab=vocab,
        max_prefill_length=max_prefill_length,
    )
    expected_padded_tokens = jnp.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_tokenize_and_pad_np(self):
    self.setup()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        s=s,
        vocab=vocab,
        max_prefill_length=max_prefill_length,
        jax_padding=False,
    )
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        np.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)


if __name__ == "__main__":
  unittest.main()
