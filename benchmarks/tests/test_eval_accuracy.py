"""Tests for long context accuracy measurement."""

import unittest

from benchmarks.eval_accuracy_longcontext import eval_accuracy_longcontext
import datetime
import re


class TestEvalAccuracy(unittest.TestCase):
  """Tests for long context accuracy measurement."""

  def setUp(self):
    self._request_outputs_dict = [
        {"generated_text": "abc", "original_output": "abc", "metric": "rouge"},
        {"generated_text": "abc", "original_output": "abc", "metric": "rouge"},
        {"generated_text": "abc", "original_output": "abc", "metric": "qa_em"},
        {"generated_text": "abc", "original_output": "abc", "metric": "qa_em"},
        {
            "generated_text": "abc",
            "original_output": "abc",
            "metric": "niah_em",
        },
        {
            "generated_text": "abc",
            "original_output": "abc",
            "metric": "niah_em",
        },
    ]

  def test_eval_accuracy_longcontext(self):
    self.assertEqual(
        eval_accuracy_longcontext(self._request_outputs_dict),
        {"rougeL": 100.0, "exact_match": 50.0, "gen_len": 18, "gen_num": 6},
    )
