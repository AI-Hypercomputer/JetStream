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

"""Tests for aot utils"""

import logging
import unittest
from parameterized import parameterized

import jax.numpy as jnp

from jetstream.engine import aot_utils
from jetstream.engine import engine_api
from jetstream.engine import mock_engine
from jetstream.engine.engine_api import JetStreamEngine
from jetstream.engine import token_utils


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AotUtilsTest(unittest.TestCase):

  def _setup(
      self, cache_length=256
  ) -> tuple[engine_api.JetStreamEngine, engine_api.Params]:
    """Initialises a test engine."""
    engine = mock_engine.TestEngine(
        batch_size=32, cache_length=cache_length, weight=2.0
    )
    params = engine.load_params()
    return JetStreamEngine(engine), params

  def _get_tokenizer(self, engine):
    tokenizer_metadata = engine.get_tokenizer()
    return token_utils.load_vocab(
        tokenizer_metadata.path, tokenizer_metadata.extra_ids
    ).tokenizer

  @parameterized.expand([True, False])
  def test_create_aot_engines_match(self, relayout_optimally):
    prefill_engine, prefill_params = self._setup(16)
    generate_engine, generate_params = self._setup(16)

    prefill_engines = [prefill_engine]
    generate_engines = [generate_engine]
    prefill_params = [prefill_params]
    generate_params = [generate_params]
    (
        layout_prefill_params_list,
        layout_generate_params_list,
        aot_prefill_engines,
        aot_generate_engines,
    ) = aot_utils.create_aot_engines(
        prefill_engines,
        generate_engines,
        prefill_params,
        generate_params,
        relayout_optimally=relayout_optimally,
    )

    self.assertEqual(len(layout_prefill_params_list), 1)
    layout_prefill_params = layout_prefill_params_list[0]
    self.assertEqual(len(aot_prefill_engines), 1)

    max_prefill_len = prefill_engine.max_prefill_length

    # Prefill step
    text = "AB"
    aot_prefill_engine = aot_prefill_engines[0]
    tokenizer = aot_prefill_engine.build_tokenizer(
        aot_prefill_engine.get_tokenizer()
    )
    tokens, true_length = tokenizer.encode(text, is_bos=True)

    prefix, prefill_result_tokens = aot_prefill_engine.prefill(
        params=layout_prefill_params,
        padded_tokens=tokens,
        true_length=true_length,
    )
    # Verify the first generated token
    token_data = prefill_result_tokens.get_result_at_slot(0)
    detokenizer = self._get_tokenizer(aot_prefill_engine)
    self.assertEqual(detokenizer.IdToPiece(int(token_data.tokens.item())), "Ċ")

    # Generate step
    self.assertEqual(len(layout_generate_params_list), 1)
    layout_generate_params = layout_generate_params_list[0]
    self.assertEqual(len(aot_generate_engines), 1)

    init_decode_state = aot_generate_engines[0].init_decode_state()
    slot = 1
    decode_state = aot_generate_engines[0].insert(
        prefix, init_decode_state, slot, prefill_length=max_prefill_len
    )
    _, decoded_tokens = aot_generate_engines[0].generate(
        params=layout_generate_params, decode_state=decode_state
    )

    # Char for 399
    token_data = decoded_tokens.get_result_at_slot(slot)
    self.assertEqual(detokenizer.IdToPiece(int(token_data.tokens.item())), "Ə")

  def test_multiple_aot_engines(self):
    prefill_engine, prefill_params = self._setup()
    generate_engine, generate_params = self._setup()
    prefill_engines = [prefill_engine, prefill_engine]
    generate_engines = [generate_engine, generate_engine]
    prefill_params = [prefill_params, prefill_params]
    generate_params = [generate_params, generate_params]
    (
        layout_prefill_params_list,
        layout_generate_params_list,
        aot_prefill_engines,
        aot_generate_engines,
    ) = aot_utils.create_aot_engines(
        prefill_engines,
        generate_engines,
        prefill_params,
        generate_params,
    )

    self.assertEqual(len(layout_prefill_params_list), 2)
    layout_prefill_params = layout_prefill_params_list[1]
    self.assertEqual(len(aot_prefill_engines), 2)

    max_prefill_len = prefill_engine.max_prefill_length

    prefix, _ = aot_prefill_engines[1].prefill(
        params=layout_prefill_params,
        padded_tokens=jnp.ones((max_prefill_len), dtype=jnp.int32),
        true_length=max_prefill_len,
    )

    self.assertEqual(len(layout_generate_params_list), 2)
    layout_generate_params = layout_generate_params_list[1]
    self.assertEqual(len(aot_generate_engines), 2)

    init_decode_state = aot_generate_engines[1].init_decode_state()
    decode_state = aot_generate_engines[1].insert(
        prefix, init_decode_state, 0, prefill_length=max_prefill_len
    )
    _, _ = aot_generate_engines[1].generate(
        params=layout_generate_params, decode_state=decode_state
    )

  def test_layout_params_and_compile_executables(self):
    prefill_engine, prefill_params = self._setup()
    generate_engine, generate_params = self._setup()
    prefill_engines = [prefill_engine]
    generate_engines = [generate_engine]
    prefill_params = [prefill_params]
    generate_params = [generate_params]
    (
        layout_prefill_params_list,
        layout_generate_params_list,
        prefill_executables_list,
        generate_executables_list,
    ) = aot_utils.layout_params_and_compile_executables(
        prefill_engines,
        generate_engines,
        prefill_params,
        generate_params,
        relayout_optimally=True,
    )

    self.assertEqual(len(layout_prefill_params_list), 1)
    layout_prefill_params = layout_prefill_params_list[0]
    self.assertEqual(len(prefill_executables_list), 1)
    prefill_executables = prefill_executables_list[0]
    max_prefill_len = prefill_engine.max_prefill_length
    self.assertIn(max_prefill_len, prefill_executables)
    prefill_executable = prefill_executables[max_prefill_len]
    prefix, _ = prefill_executable(
        layout_prefill_params,
        jnp.ones((max_prefill_len), dtype=jnp.int32),
        max_prefill_len,
    )

    self.assertEqual(len(layout_generate_params_list), 1)
    layout_generate_params = layout_generate_params_list[0]
    self.assertEqual(len(generate_executables_list), 1)
    generate_executables = generate_executables_list[0]
    (init_decode_state_executable, insert_executables, generate_executable) = (
        generate_executables
    )
    init_decode_state = init_decode_state_executable()
    decode_state = insert_executables[prefill_engine.max_prefill_length](
        prefix, init_decode_state, 0
    )
    _, _ = generate_executable(layout_generate_params, decode_state)
