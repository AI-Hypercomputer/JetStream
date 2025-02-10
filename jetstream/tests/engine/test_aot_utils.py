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

import jax.numpy as jnp

from jetstream.engine import aot_utils
from jetstream.engine import engine_api
from jetstream.engine import mock_engine
from jetstream.engine.engine_api import JetStreamEngine

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AotUtilsTest(unittest.TestCase):

  def _setup(self) -> tuple[engine_api.JetStreamEngine, engine_api.Params]:
    """Initialises a test engine."""
    engine = mock_engine.TestEngine(batch_size=32, cache_length=256, weight=2.0)
    params = engine.load_params()
    return JetStreamEngine(engine), params

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
      prefill_engines, generate_engines, prefill_params, generate_params,
      relayout_params_optimally=True,
      relayout_decode_state_optimally=True,
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
      max_prefill_len
    )

    self.assertEqual(len(layout_generate_params_list), 1)
    layout_generate_params = layout_generate_params_list[0]
    self.assertEqual(len(generate_executables_list), 1)
    generate_executables = generate_executables_list[0]
    (
      init_decode_state_executable, insert_executables, generate_executable
    ) = generate_executables
    init_decode_state = init_decode_state_executable()
    decode_state = insert_executables[prefill_engine.max_prefill_length](
      prefix, init_decode_state, 0
    )
    _, _ = generate_executable(
      layout_generate_params, decode_state
    )
