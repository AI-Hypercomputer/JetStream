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

"""Tests for a mock version of the engine API.

What should we expect?

Prefill: Doubles the sequence by multiplying it with a weight [2].
Insert: Writes this sequence into a cache row
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jetstream.engine import mock_engine
from jetstream.engine import token_utils


class EngineTest(unittest.TestCase):

  def _setup(self):
    """Initialises a test engine."""
    engine = mock_engine.TestEngine(batch_size=32, cache_length=256, weight=2.0)
    params = engine.load_params()
    return engine, params

  def _prefill(self):
    """Performs prefill and returns a kv cache."""
    engine, params = self._setup()
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True)
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, true_length=3
    )
    return engine, params, prefill_result, true_length, first_token

  def _prefill_np(self):
    """Performs prefill and returns a kv cache."""
    engine, params = self._setup()
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True, jax_padding=False)
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, true_length=3
    )
    return engine, params, prefill_result, true_length, first_token

  def _generate(self, slot=1):
    """Performs a single generation step."""
    engine, params, prefill_result, _, _ = self._prefill()
    decode_state = engine.init_decode_state()
    decode_state = engine.insert(
        prefix=prefill_result, decode_state=decode_state, slot=slot
    )
    decode_state, sampled_tokens = engine.generate(
        params=params, decode_state=decode_state
    )
    return engine, params, decode_state, sampled_tokens

  def test_load_params(self):
    """Just loads params."""
    _, params = self._setup()
    assert params == jnp.array([2.0])

  def test_prefill(self):
    """Tests prefill with weight = 2."""
    engine, _, prefill_result, true_length, first_token = self._prefill()
    prefill_cache, _ = prefill_result
    np.testing.assert_array_equal(
        prefill_cache[:, :true_length], np.array([[4.0, 130.0, 132.0]])
    )

    # test first token
    token_data = first_token.get_result_at_slot(0)
    tok = token_data.tokens

    metadata = engine.get_tokenizer()
    tokenizer = token_utils.load_vocab(
        metadata.path, metadata.extra_ids
    ).tokenizer
    assert tokenizer.IdToPiece(int(tok.item())) == "Ċ"

  def test_prefill_np(self):
    """Tests prefill with weight = 2."""
    _, _, prefill_result, true_length, _ = self._prefill_np()
    prefill_cache, _ = prefill_result
    np.testing.assert_array_equal(
        prefill_cache[:, :true_length], np.array([[4.0, 130.0, 132.0]])
    )

  def test_generate(self, slot=1):
    """Tests multiple generation steps."""
    engine, params, decode_state, sampled_tokens = self._generate(slot=slot)
    metadata = engine.get_tokenizer()
    tokenizer = token_utils.load_vocab(
        metadata.path, metadata.extra_ids
    ).tokenizer

    # Char for 399
    token_data = sampled_tokens.get_result_at_slot(slot)
    tok = token_data.tokens
    assert tokenizer.IdToPiece(int(tok.item())) == "Ə"
    _, sampled_tokens = engine.generate(
        params=params, decode_state=decode_state
    )
    # Char for 598
    token_data = sampled_tokens.get_result_at_slot(slot)
    tok = token_data.tokens
    assert tokenizer.IdToPiece(int(tok.item())) == "ɖ"
