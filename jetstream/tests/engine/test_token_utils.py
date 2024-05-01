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
from jetstream.engine import engine_api


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

  def setup_sentencepiece(self):
    self.tokenizer_path = "third_party/llama2/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"
    self.sp_tokenizer = SPTokenizer(self.tokenizer_path)
    self.jt_tokenizer = JetStreamTokenizer(self.tokenizer_path)

  def setup_tiktoken(self):
    self.tokenizer_path = "third_party/llama3/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"

  def test_decode_vs_piece(self):
    self.setup_sentencepiece()
    tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
    expeted_sp_output = []
    jt_output = []
    for t in tokens:
      expeted_sp_output.append(self.sp_tokenizer.decode([t]))
      jt_output.append(self.jt_tokenizer.decode(t))

    self.assertNotEqual(jt_output, expeted_sp_output)

  def test_sp_vs_seqio(self):
    self.setup_sentencepiece()
    for n in range(0, self.sp_tokenizer.tokenizer.vocab_size()):
      sp_t = self.sp_tokenizer.decode([n])
      seqio_t = self.jt_tokenizer.vocab.tokenizer.decode([n])
      self.assertEqual(sp_t, seqio_t)

  def test_tokenize_and_pad_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
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
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
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

  def test_sentencepiece_tokenizer_encode(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s)
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_no_bos(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s, is_bos=False)
    expected_padded_tokens = np.array(
        [306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 7
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_prefill_lengths(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s, prefill_lengths=[12, 24, 36])
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    padded_tokens, true_length = tokenizer.encode(s, jax_padding=True)
    expected_padded_tokens = jnp.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_decode(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [
                [
                    306,
                    4658,
                    278,
                    6593,
                    310,
                    2834,
                    338,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    length,
                ]
            ]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    tokens, complete = tokenizer.decode(0, 16, result_tokens, complete)
    expected_tokens = np.array([[306, 4658, 278, 6593, 310, 2834, 338]])
    self.assertTrue(np.allclose(tokens, expected_tokens, atol=1e-7))
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_sentencepiece_tokenizer_decode_str(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    result = tokenizer.decode_str([306, 4658, 278, 6593, 310, 2834, 338])
    self.assertTrue(result == "I believe the meaning of life is")

  def test_tiktoken_tokenizer_encode(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s)
    expected_padded_tokens = np.array(
        [
            128000,
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_no_bos(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s, is_bos=False)
    expected_padded_tokens = np.array(
        [
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 7
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_prefill_lengths(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s, prefill_lengths=[12, 24, 36])
    expected_padded_tokens = np.array(
        [128000, 40, 4510, 279, 7438, 315, 2324, 374, -1, -1, -1, -1]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    padded_tokens, true_length = tokenizer.encode(s, jax_padding=True)
    expected_padded_tokens = jnp.array(
        [
            128000,
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_decode(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [[40, 4510, 279, 7438, 315, 2324, 374, 1, 1, 1, 1, 1, 1, 1, length]]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    tokens, complete = tokenizer.decode(0, 16, result_tokens, complete)
    expected_tokens = np.array([[40, 4510, 279, 7438, 315, 2324, 374]])
    self.assertTrue(np.allclose(tokens, expected_tokens, atol=1e-7))
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_tiktoken_decode_str(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    result = tokenizer.decode_str([40, 4510, 279, 7438, 315, 2324, 374])
    self.assertTrue(result == "I believe the meaning of life is")


if __name__ == "__main__":
  unittest.main()
