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

"""Tests functionality of the token processing utils using mock engine vocab."""

import numpy as np
import unittest
from jetstream.engine import engine_api
from jetstream.engine import mock_utils
from jetstream.engine import token_utils


class UtilsTest(unittest.TestCase):

  def test_speculations_with_multi_sample_slots(self, samples_per_slot=2):
    # [4, 1]
    mock_tokens = np.array(
        [
            [0, ord("A")],
            [ord("A"), ord("D")],
            [ord("T"), ord("3")],
            [ord("A"), 1],
        ]
    ).astype(np.int32)
    mock_valid_tokens = np.ones_like(mock_tokens, dtype=np.int32)
    mock_lengths = np.ones(mock_tokens.shape[0], dtype=np.int32) * 2
    # completion is 'per slot' because we track it for a given request.
    mock_complete = np.zeros(
        (mock_tokens.shape[0] // samples_per_slot), dtype=np.int32
    )
    data = np.concatenate(
        [
            mock_tokens,
            mock_valid_tokens,
            mock_lengths[:, None],
        ],
        axis=-1,
    )
    speculations = mock_tokens.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=data.astype(np.int32),
        tokens_idx=(0, speculations),
        valid_idx=(speculations, 2 * speculations),
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=2,
    )
    vocab = mock_utils.TestVocab()
    per_channel, complete = token_utils.process_result_tokens(
        tokenizer=vocab,
        slots=0,
        slot_max_length=4,
        result_tokens=result_tokens,
        complete=mock_complete,
        is_client_side_tokenization=False,
    )
    np.testing.assert_equal(complete, np.array([1, 0]))

    text_output = [
        mock_utils.TestVocab().decode(row.token_ids) for row in per_channel
    ]
    assert not text_output[0]  # i.e. == '', because of the pad.
    assert text_output[1] == "AD"
    mock_complete = np.zeros(
        (mock_tokens.shape[0] // samples_per_slot), dtype=np.int32
    )
    per_channel, complete = token_utils.process_result_tokens(
        tokenizer=vocab,
        slots=1,
        slot_max_length=4,
        result_tokens=result_tokens,
        complete=mock_complete,
        is_client_side_tokenization=False,
    )
    text_output = [
        mock_utils.TestVocab().decode(row.token_ids) for row in per_channel
    ]
    assert text_output[0] == "T3"
    assert text_output[1] == "A"  # second token is padded.
    np.testing.assert_equal(complete, np.array([0, 1]))

  def test_mock_utils(self):
    vocab = mock_utils.TestVocab()
    # test encode()
    with self.assertRaises(NotImplementedError):
      vocab.encode("AB")
    # test encode_tf()
    token_ids = vocab.encode_tf("AB")
    np.testing.assert_equal(token_ids, np.array([65, 66]))
    # test decode()
    ids = np.array([ord("A")])
    expected = "A"
    result = vocab.decode(ids)
    self.assertEqual(result, expected)
    # test decode_tf()
    ids = np.array([[ord("A")]])
    expected = ["A"]
    result_tf = vocab.decode_tf(ids)
    self.assertEqual(result_tf, expected)
