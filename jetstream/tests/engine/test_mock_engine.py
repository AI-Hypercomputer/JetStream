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

  def _prefill(self, engine, params):
    """Performs prefill and returns a kv cache."""
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    tokenizer = engine.build_tokenizer(engine.get_tokenizer())
    tokens, true_length = tokenizer.encode(text, is_bos=True)
    prefill_result, result_tokens = engine.prefill(
      params=params, padded_tokens=tokens, true_length=3
    )
    return prefill_result, true_length, result_tokens

  def _prefill_nopadding(self, engine, params):
    """Performs prefill and returns a kv cache."""
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True, jax_padding=False)
    prefill_result, first_token = engine.prefill(
      params=params, padded_tokens=tokens, true_length=3
    )
    return prefill_result, true_length, first_token

  def _generate(self, engine, params, slot=1):
    """Performs a single generation step."""
    prefill_result, _, _ = self._prefill(engine, params)
    decode_state = engine.init_decode_state()
    decode_state = engine.insert(
      prefix=prefill_result, decode_state=decode_state, slot=slot
    )
    decode_state, sampled_tokens = engine.generate(
      params=params, decode_state=decode_state
    )
    return decode_state, sampled_tokens

  def _get_tokenizer(self, engine):
    tokenizer_metadata = engine.get_tokenizer()
    return token_utils.load_vocab(
      tokenizer_metadata.path, tokenizer_metadata.extra_ids
    ).tokenizer

  def test_load_params(self):
    """Just loads params."""
    _, params = self._setup()
    self.assertEqual(params, jnp.array([2.0]))

  def test_prefill(self):
    """Tests prefill for input with padding."""
    engine, params = self._setup()

    prefill_result, true_length, first_token = self._prefill(engine, params)

    # Verify prefill cache.
    prefill_cache = prefill_result['cache']
    print(type(prefill_cache))
    self.assertTrue(
      jnp.array_equal(
        prefill_cache[:, :true_length],
        jnp.array([[4.0, 130.0, 132.0]])
      )
    )

    # Verify the first generated token
    token_data = first_token.get_result_at_slot(0)
    tokenizer = self._get_tokenizer(engine)
    assert tokenizer.IdToPiece(int(token_data.tokens.item())) == "Ċ"

  def test_prefill_without_padding(self):
    """Tests prefill for input without padding."""
    engine, params = self._setup()
    prefill_result, true_length, _ = self._prefill_nopadding(engine, params)
    prefill_cache = prefill_result['cache']
    self.assertTrue(
      jnp.array_equal(
        prefill_cache[:, :true_length],
        jnp.array([[4.0, 130.0, 132.0]])
      )
    )

  def test_generate(self, slot=1):
    """Tests multiple generation steps."""
    engine, params = self._setup()
    tokenizer = self._get_tokenizer(engine)

    decode_state, sampled_tokens = self._generate(engine, params, slot=slot)

    # Char for 399
    token_data = sampled_tokens.get_result_at_slot(slot)
    self.assertEqual(tokenizer.IdToPiece(int(token_data.tokens.item())), "Ə")

    _, sampled_tokens = engine.generate(
      params=params, decode_state=decode_state
    )
    # Char for 598
    token_data = sampled_tokens.get_result_at_slot(slot)
    self.assertEqual(tokenizer.IdToPiece(int(token_data.tokens.item())), "ɖ")
