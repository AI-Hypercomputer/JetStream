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

import contextlib
import io
import unittest
from unittest import mock

import importlib
import jetstream.engine


class InitTest(unittest.TestCase):

  def test_init_with_error(self):
    def mock_find_spec(name):
      if name == "pathwaysutils":
        return None
      return "some_spec"

    with mock.patch(
        "importlib.util.find_spec", side_effect=mock_find_spec
    ), contextlib.redirect_stdout(io.StringIO()) as captured_output:

      importlib.reload(jetstream.engine)

      self.assertIn(
          "Running JetStream without Pathways.", captured_output.getvalue()
      )

  def test_init(self):
    orig_import = __import__
    p_mock = mock.Mock()

    def mock_import(name, *args):

      if name == "pathwaysutils":
        return p_mock
      return orig_import(name, *args)

    def mock_find_spec(name):
      if name == "pathwaysutils":
        return "pathwaysutils_spec"
      return "some_spec"

    with mock.patch(
        "importlib.util.find_spec", side_effect=mock_find_spec
    ), mock.patch(
        "builtins.__import__", side_effect=mock_import
    ), contextlib.redirect_stdout(
        io.StringIO()
    ) as captured_output:
      importlib.reload(jetstream.engine)

      p_mock.initialize.assert_called_once()

      self.assertNotIn(
          "Running JetStream without Pathways.", captured_output.getvalue()
      )
