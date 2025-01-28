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

"""Metrics util classes for collecting and managing metrics."""

import datetime

import numpy as np
from typing import Tuple, List, Dict


def _floor_datetime_to_sec(timestamp: datetime.datetime) -> datetime.datetime:
  """ "Floor the timestamp to the nearest most recent second"""
  return timestamp.replace(microsecond=0)


def _now_floored_to_second() -> datetime.datetime:
  """Return the current timestamp floored to the nearest most recent second a"""
  now = datetime.datetime.now()
  return _floor_datetime_to_sec(now)


class EventMetric:
  """An event metric  for distribution stats reporting. Not thread-safe."""

  def __init__(self, name: str, description: str, unit: str = ""):
    self._name = name
    self._description = description
    self._unit = unit
    self._data = []

  def data(self) -> List[float]:
    """Returns all stored data points.

    Returns:
      A list of data points in the order that was stored
    """
    return self._data

  def record(self, value: float):
    """Record a data point

    Args:
      value: The data point to be stored.
    """
    self._data.append(value)

  def percentile(self, percentile: int) -> float:
    """Computes and returns the specified percentile of the collected data.

    Args:
      percentile: The percentile to compute.

    Returns:
      The computed percentile.
    """
    if not 0 <= percentile <= 100:
      raise ValueError(f"Percentile {percentile} is not in [0, 100]")
    if not self._data:
      raise ValueError(
          f"No data points in metric {self._name} to compute percentile"
      )
    return np.percentile(self._data, percentile)

  def mean(self) -> float:
    """Calculates and returns the mean value of the collected data.

    Returns:
        The mean value of the collected data
    """
    if not self._data:
      raise ValueError(f"No data points in metric {self._name} to compute mean")
    return np.mean(self._data)

  def distribution_summary_str(self) -> str:
    """Generates a string representation of the distribution summary

    Returns:
        The string representation of the distribution summary including
        mean, p50, p90 and p99.
    """
    s = ""
    s += f"Mean {self._name}: {self.mean():.2f} {self._unit}\n"
    s += f"Median {self._name}: {self.percentile(50):.2f} {self._unit}\n"
    s += f"P99 {self._name}: {self.percentile(99):.2f} {self._unit}"
    return s

  def distribution_summary_dict(self) -> dict[str, float]:
    """Generates a dictionary representation of the distribution summary

    Returns:
      A dictionary containing of the distribution summary including mean,
      p50, p90 and p99.
    """
    return {
        f"mean_{self._name}_{self._unit}": self.mean(),
        f"median_{self._name}_{self._unit}": self.percentile(50),
        f"p99_{self._name}_{self._unit}": self.percentile(99),
    }


class CounterMetric:
  """A count metric for computing rates over time. Not thread-safe."""

  def __init__(self, name: str, description: str):
    self._name = name
    self._description = description
    self._data: dict[datetime.datetime, int] = {}

  def data(self) -> Dict[datetime.datetime, int]:
    """Returns all stored data points.

    Returns:
      A dictionary of data points where the key is the timestamp and the value
      is the aggregated counts within the second of the timestamp.
    """
    return self._data

  def total_count(self) -> int:
    """Returns aggregated counts

    Returns:
      The aggregated counts.
    """
    return sum(self._data.values())

  def total_duration_sec(self) -> int:
    """Returns the duration between the first and last count increment

    Returns:
        The duration (in seconds) between the first and last increment
        (inclusive of both ends).
    """
    start_time = min(self._data.keys())
    end_time = max(self._data.keys())
    return int((end_time - start_time).total_seconds() + 1)

  def increment(
      self, count: int = 1, timestamp: datetime.datetime | None = None
  ):
    """Increment the counter by count

    Args:
        count: The amount to increment
        timestamp: The timestamp for the increment. Default to now if none is
          provided.
    """
    if timestamp is None:
      cur_time = _now_floored_to_second()
    else:
      cur_time = _floor_datetime_to_sec(timestamp)
    # Add timestamp with default value 0 if doesn't exist
    cur_count = self._data.setdefault(cur_time, 0)
    self._data[cur_time] = cur_count + count
    return

  def rate(self) -> float:
    """Calculates the rate of change between the first and last increments.

    Returns:
      The rate of change between the first and last increments.
    """
    if len(self._data.keys()) < 2:
      raise ValueError(
          "At least 2 data points are required to compute the rate"
      )
    start_time = min(self._data.keys())
    end_time = max(self._data.keys())
    delta_time_sec = (end_time - start_time).total_seconds()
    sorted_counts = [count for timestamp, count in sorted(self._data.items())]
    delta_count = sum(sorted_counts[1:])
    return delta_count / delta_time_sec

  def rate_over_window(
      self, window_size_sec: int
  ) -> List[Tuple[datetime.datetime, float]]:
    """Calculate the rates over time."

    Args:
      window_size_sec: The size of the window in seconds for computing each
        individual rate

    Returns:
      A list of rates over time, where each element represents the rate of
      change for the specified window size.
    """
    if len(self._data.keys()) < 2:
      raise ValueError(
          f"At least 2 different timestamp values are required to calculate "
          f"the rate, but have only {len(self._data.keys())}"
      )
    rates: List[Tuple[datetime.datetime, float]] = []
    sorted_data = sorted(self._data.items())

    start_time, _ = sorted_data[0]
    end_time, _ = sorted_data[-1]
    cur_start_time = start_time
    cur_end_time = cur_start_time + datetime.timedelta(seconds=window_size_sec)
    cur_total_count = 0
    for data_point in sorted_data:
      timestamp, count = data_point
      if timestamp >= cur_end_time:
        while timestamp >= cur_end_time:
          rates.append((cur_start_time, cur_total_count / window_size_sec))
          cur_start_time = cur_end_time
          cur_end_time = cur_start_time + datetime.timedelta(
              seconds=window_size_sec
          )
          cur_total_count = 0
      cur_total_count += count
    if cur_start_time <= end_time:
      delta_time_sec = (end_time - cur_start_time).total_seconds() + 1
      rates.append((cur_start_time, cur_total_count / delta_time_sec))
    return rates

  def rate_over_window_to_csv(self, window_size_sec: int) -> str:
    """Compute and return the rates over time and return them in csv string

    Args:
      window_size_sec: The size of the window in seconds for computing each
        individual rate

    Returns:
      A CSV string representation of the rates over time, with two rows:
      the first row contains timestamps, and the second row contains rate
      values.
    """
    rates = self.rate_over_window(window_size_sec)
    # Generate CSV string with two rows
    timestamps = "TimeStamp," + ",".join([str(e[0]) for e in rates])
    values = "Value," + ",".join([f"{e[1]:.2f}" for e in rates])
    csv_output = timestamps + "\n" + values
    return csv_output
