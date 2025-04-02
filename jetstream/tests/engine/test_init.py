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

"""Tests for initializing jetstream.engine module."""

import unittest
from unittest import mock


class InitTest(unittest.TestCase):

  def test_init(self):
    orig_import = __import__
    p_mock = mock.Mock()

    def import_mock(name, *args):
      if name == "pathwaysutils":
        return p_mock
      return orig_import(name, *args)

    with mock.patch("builtins.__import__", side_effect=import_mock):
      from jetstream import engine  # pylint: disable=import-outside-toplevel,unused-import

      p_mock.initialize.assert_called_once()
