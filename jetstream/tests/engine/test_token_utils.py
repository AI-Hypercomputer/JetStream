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
  """Tokenizer used in original llama2 git"""

  def __init__(self, tokenizer_path: str):
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.Load(model_file=tokenizer_path)
    assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

  def decode(self, t: List[int]) -> str:
    token = self.tokenizer.decode(t)
    return token


class JetStreamTokenizer:
  """Tokenizer used in JetStream before mix_token"""

  def __init__(self, tokenizer_path: str):
    metadata = tokenizer_pb2.TokenizerParameters(path=tokenizer_path)
    self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  def decode(self, t: int) -> str:
    token = self.vocab.tokenizer.IdToPiece(t)
    token = token.replace("▁", " ")
    return token


class TokenUtilsTest(unittest.TestCase):

  def setup_sentencepiece(self):
    self.tokenizer_path = "external_tokenizers/llama2/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"
    self.sp_tokenizer = SPTokenizer(self.tokenizer_path)
    self.jt_tokenizer = JetStreamTokenizer(self.tokenizer_path)

  def setup_tiktoken(self):
    self.tokenizer_path = "external_tokenizers/llama3/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"

  def setup_hftoken(self):

    # Download the tokenizer.
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = (
        "external_tokenizers/gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e/"
    )
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.exists(
        self.tokenizer_path
    ), f"did not find tokenizer_path: {self.tokenizer_path}"

  def test_decode_vs_piece(self):
    self.setup_sentencepiece()
    tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
    expected_sp_output = []
    jt_output = []
    for t in tokens:
      expected_sp_output.append(self.sp_tokenizer.decode([t]))
      jt_output.append(self.jt_tokenizer.decode(t))

    self.assertNotEqual(jt_output, expected_sp_output)

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

  def test_chunk_and_pad_tokens(self):
    jax.config.update("jax_platform_name", "cpu")
    tokens = np.arange(100, 166, dtype=np.int32)
    padding_tokens, true_lengths, positions = token_utils.chunk_and_pad_tokens(
        tokens,
        bos_id=1,
        pad_id=0,
        is_bos=True,
        prefill_lengths=[2, 4, 16, 64, 128],
        chunk_size=16,
        max_prefill_length=128,
        jax_padding=True,
    )
    expected_padding_tokens = [
        jnp.concat([jnp.array([1]), jnp.arange(100, 115)]),
        jnp.arange(115, 131),
        jnp.arange(131, 147),
        jnp.arange(147, 163),
        jnp.array([163, 164, 165, 0]),  # fit bucket 4 and padding 0
    ]
    expected_positions = [
        jnp.expand_dims(jnp.arange(0, 16), 0),
        jnp.expand_dims(jnp.arange(16, 32), 0),
        jnp.expand_dims(jnp.arange(32, 48), 0),
        jnp.expand_dims(jnp.arange(48, 64), 0),
        jnp.expand_dims(jnp.arange(64, 68), 0),
    ]
    print("padding_tokens ", padding_tokens)
    print("true_lengths ", true_lengths)
    print("positions ", positions)
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, padding_tokens, expected_padding_tokens)
    )
    assert true_lengths == [16, 16, 16, 16, 3]
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, positions, expected_positions)
    )

  def test_tokenize_and_pad(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        s,
        vocab,
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

  def test_pad_token_padding_less_than_zero(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is having different experiences and "
    s += "enjoy everyday of my life."
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 16
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
        max_prefill_length=max_prefill_length,
    )
    # Take the last N tokens if we have too many.
    expected_padded_tokens = jnp.array(
        [
            278,
            6593,
            310,
            2834,
            338,
            2534,
            1422,
            27482,
            322,
            13389,
            1432,
            3250,
            310,
            590,
            2834,
            29889,
        ]
    )
    expected_true_length = 19
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
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

  def test_process_result_with_sentencepiece_tokenizer_decode(self):
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
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, False
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[306, 4658, 278, 6593, 310, 2834, 338]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    self.assertTrue(
        samples[0].text
        == [" I", " believe", " the", " meaning", " of", " life", " is"]
    )
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_process_result_with_sentencepiece_tokenizer_client_decode(self):
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
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, True
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[306, 4658, 278, 6593, 310, 2834, 338]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    # Return token ids only when in client side tokenization mode.
    self.assertTrue(samples[0].text == [])
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_sentencepiece_tokenizer_decode(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    result = tokenizer.decode([306, 4658, 278, 6593, 310, 2834, 338])
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

  def test_process_result_with_tiktoken_decode(self):
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
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, False
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[40, 4510, 279, 7438, 315, 2324, 374]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    self.assertTrue(
        samples[0].text
        == ["I", " believe", " the", " meaning", " of", " life", " is"]
    )
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_process_result_with_tiktoken_client_decode(self):
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
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, True
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[40, 4510, 279, 7438, 315, 2324, 374]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    # Return token ids only when in client side tokenization mode.
    self.assertTrue(samples[0].text == [])
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_tiktoken_decode(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    result = tokenizer.decode([40, 4510, 279, 7438, 315, 2324, 374])
    self.assertTrue(result == "I believe the meaning of life is")

  def test_text_tokens_to_str(self):
    # Start with text token
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "<0x0A>", "hello"]
        )
        == "你好吗\nhello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            [
                "你",
                "好",
                "<0xE5>",
                "<0x90>",
                "<0x97>",
                "<0x0A>",
                "<0x0A>",
                "hello",
            ]
        )
        == "你好吗\n\nhello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello", "<0x0A>"]
        )
        == "你好吗hello\n"
    )
    # Start with byte token
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "\n你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            [
                "<0x0A>",
                "<0x0A>",
                "你",
                "好",
                "<0xE5>",
                "<0x90>",
                "<0x97>",
                "hello",
            ]
        )
        == "\n\n你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(["<0xE5>", "<0x90>", "<0x97>", "hello"])
        == "吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "<0x0A>", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "\n\n吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "<0x0A>", "<0xE5>", "<0x90>", "<0x97>"]
        )
        == "\n\n吗"
    )
    # Invalid byte token sequence
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x0A>", "<0x97>", "hello"]
        )
        == "你好�\n�hello"
    )

  def test_hf_decode(self):
    self.setup_hftoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
    tokenizer_model.tokenizer.pad_token = tokenizer_model.tokenizer.eos_token
    # Check that special bos & padding tokens are not emitted.
    tokens = [50256, 43, 1039, 11241, 1096, 281, 1672, 0, 50256, 50256]
    expected_hf_output = "Lets tokenize an example!"
    hf_output = tokenizer_model.decode(tokens)
    self.assertEqual(hf_output, expected_hf_output)

  def test_hf_encode_use_chat_template(self):
    self.setup_hftoken()
    metadata = tokenizer_pb2.TokenizerParameters(
        path=self.tokenizer_path, use_chat_template=True
    )
    tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
    tokenizer_model.tokenizer.pad_token = tokenizer_model.tokenizer.eos_token
    # Make this string 80 characters per line.
    tokenizer_model.tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
    )

    s = "Lets tokenize an example!"
    tokens, true_length = tokenizer_model.encode(s)
    expected_padded_tokens = np.array(
        [
            50256,
            27,
            91,
            7220,
            91,
            29,
            198,
            43,
            1039,
            11241,
            1096,
            281,
            1672,
            0,
            198,
            27,
            91,
            562,
            10167,
            91,
            29,
            198,
        ]
    )
    expected_true_length = 22
    self.assertTrue(
        np.array_equal(tokens[:true_length], expected_padded_tokens)
    )
    self.assertEqual(true_length, expected_true_length)
    self.assertTrue(np.all(tokens[true_length:] == tokenizer_model.pad_id))

  def test_hf_encode_bos(self):
    self.setup_hftoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
    tokenizer_model.tokenizer.pad_token = tokenizer_model.tokenizer.eos_token
    s = "Lets tokenize an example!"
    tokens, true_length = tokenizer_model.encode(s, is_bos=True)
    expected_padded_tokens = np.array(
        [50256, 43, 1039, 11241, 1096, 281, 1672, 0]
    )
    self.assertTrue(
        np.array_equal(tokens[:true_length], expected_padded_tokens)
    )
    self.assertEqual(true_length, 8)
    self.assertTrue(np.all(tokens[true_length:] == tokenizer_model.pad_id))

  def test_hf_encode_no_bos(self):
    self.setup_hftoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
    tokenizer_model.tokenizer.pad_token = tokenizer_model.tokenizer.eos_token
    s = "Lets tokenize an example!"
    tokens, true_length = tokenizer_model.encode(s, is_bos=False)
    expected_padded_tokens = np.array([43, 1039, 11241, 1096, 281, 1672, 0])
    self.assertTrue(
        np.array_equal(tokens[:true_length], expected_padded_tokens)
    )
    self.assertEqual(true_length, 7)
    self.assertTrue(np.all(tokens[true_length:] == tokenizer_model.pad_id))
