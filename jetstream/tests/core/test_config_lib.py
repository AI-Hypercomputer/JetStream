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

"""Unit test for config_lib.py."""

import unittest
from parameterized import parameterized
from jetstream.core import config_lib


class TestConfigLib(unittest.TestCase):

  @parameterized.expand([("tpu=8", 8), ("v5e-8", 8), ("v5e=4", 4), ("v4-8", 4)])
  def test_slice_to_num_chips(self, accelerator_slice, expected_num_devices):
    got = config_lib.slice_to_num_chips(accelerator_slice)
    self.assertEqual(got, expected_num_devices)

  def test_get_engines_invalid(self):
    with self.assertRaises(ValueError):
      config_lib.get_engines(config_lib.InterleavedCPUTestServer, [])
