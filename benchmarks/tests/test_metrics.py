# Copyright 2025 Google LLC
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

"""Tests for metrics."""

import unittest

import benchmarks.metrics as metrics
import datetime
import re


class TestEventMetric(unittest.TestCase):
  """ "Tests for event metric (i.e. distribution)."""

  def setUp(self):
    self._metric = metrics.EventMetric(
        "requestLatency", "Latency of requests", "ms"
    )

  def test_record_adds_a_data_event(self):
    m = self._metric
    m.record(1.0)
    data_points = m.data()
    self.assertEqual(1, len(data_points))
    self.assertEqual(1.0, data_points[0])

  def test_percentile_returns_correct_percentile(self):
    m = self._metric
    n = 11
    for i in range(0, n):
      m.record(i)
    self.assertEqual(m.percentile(50), 5)
    self.assertEqual(m.percentile(90), 9)
    self.assertEqual(m.percentile(100), 10)

  def test_mean_returns_correct_mean_value(self):
    m = self._metric
    n = 3
    for i in range(0, n):
      m.record(i)
    self.assertEqual(sum(range(0, n)) / n, m.mean())

  def test_distribution_summary_str_returns_expected_str(self):
    m = self._metric
    n = 100
    for i in range(0, n):
      m.record(i)
    summary = m.distribution_summary_str()
    self.assertTrue(re.search(r"Mean requestLatency", summary))
    self.assertTrue(re.search(r"Median requestLatency", summary))
    self.assertTrue(re.search(r"P99 requestLatency", summary))

  def test_distribution_summary_dict_returns_expected_dict(self):
    m = self._metric
    n = 100
    for i in range(0, n):
      m.record(i)
    summary = m.distribution_summary_dict()
    self.assertIn("mean_requestLatency_ms", summary)
    self.assertIn("median_requestLatency_ms", summary)
    self.assertIn("p99_requestLatency_ms", summary)


class TestCounterMetric(unittest.TestCase):
  """Tests for counter metric (i.e. monotonically increasing counter)."""

  def setUp(self):
    self._counter = metrics.CounterMetric(
        "RequestCompleteCount", "Number of completed requests"
    )

  def test_increment_increases_total_count(self):
    m = self._counter
    old_total_cnt = m.total_count()
    m.increment()
    new_total_cnt = m.total_count()
    self.assertEqual(1, new_total_cnt - old_total_cnt)

  def test_increment_within_same_second_update_counts_for_the_second(self):
    """Test to ensure one entry used to cumulate counts within a second"""
    m = self._counter
    timestamp = datetime.datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    m.increment(1, timestamp)
    m.increment(2, timestamp)
    data = m.data()
    self.assertEqual(1, len(data.keys()))
    self.assertEqual(3, data[timestamp])

  def test_increment_at_different_seconds_creates_separate_entries(self):
    """Test to ensure separate entries used for different seconds"""
    m = self._counter
    timestamp_first = datetime.datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    m.increment(1, timestamp_first)
    timestamp_second = datetime.datetime.strptime(
        "2025-01-01 00:00:01", "%Y-%m-%d %H:%M:%S"
    )
    m.increment(2, timestamp_second)
    data = m.data()
    self.assertEqual(2, len(data.keys()))
    self.assertEqual(1, data[timestamp_first])
    self.assertEqual(2, data[timestamp_second])

  def test_rate_returns_expected(self):
    m = self._counter
    n = 10
    start_time = datetime.datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    delta_time_sec = 1
    for i in range(0, n):
      m.increment(
          1, start_time + datetime.timedelta(seconds=delta_time_sec * i)
      )
    # n counts across n seconds, thus rate = 1
    self.assertEqual(1, m.rate())

  def test_rate_over_window_returns_expected(self):
    m = self._counter
    n = 10
    start_time = datetime.datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    delta_time_sec = 1
    for i in range(0, n):
      m.increment(
          1, start_time + datetime.timedelta(seconds=delta_time_sec * i)
      )

    rates_with_timestamp = m.rate_over_window(window_size_sec=5)

    rates = [rate for timestamp, rate in rates_with_timestamp]
    # 10 seconds with 1 count in each second. One rate per window_size_sec=5 sec
    # so rate = 1 for [0, 5) and rate = 1 [5, 10)
    self.assertEqual([1, 1], rates)

  def test_rate_over_window_to_csv_returns_correct(self):
    m = self._counter
    n = 10
    start_time = datetime.datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    delta_time_sec = 1
    for i in range(0, n):
      m.increment(
          1, start_time + datetime.timedelta(seconds=delta_time_sec * i)
      )

    csv_output = m.rate_over_window_to_csv(window_size_sec=5)

    rows = csv_output.split("\n")
    self.assertEqual(2, len(rows))
    expected_timestamps = "TimeStamp,2025-01-01 00:00:00,2025-01-01 00:00:05"
    got_timestamps = rows[0]
    self.assertEqual(expected_timestamps, got_timestamps)
    expected_values = "Value,1.00,1.00"
    got_values = rows[1]
    self.assertEqual(expected_values, got_values)


if __name__ == "__main__":
  unittest.main()
