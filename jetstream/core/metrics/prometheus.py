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

"""Contains common functions for configuring Jetstream server metrics"""

import os
import shortuuid
from prometheus_client import Gauge, Histogram


class JetstreamMetricsCollector:
  """Wrapper class should be used to assure all metrics have proper tags"""

  _id: str = os.getenv("HOSTNAME", shortuuid.uuid())

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(JetstreamMetricsCollector, cls).__new__(cls)
    return cls.instance

  # Metric definitions
  _prefill_backlog = Gauge(
      name="jetstream_prefill_backlog_size",
      documentation="Size of prefill queue",
      labelnames=["id"],
  )

  _transfer_backlog = Gauge(
      name="jetstream_transfer_backlog_size",
      documentation="Size of transfer queue",
      labelnames=["id", "idx"],
  )

  _generate_backlog = Gauge(
      name="jetstream_generate_backlog_size",
      documentation="Size of generate queue",
      labelnames=["id", "idx"],
  )

  _queue_duration = Histogram(
      name="jetstream_queue_duration",
      documentation="The total time each request spends enqueued in seconds",
      labelnames=["id"],
      buckets=[
          0.01,
          0.02,
          0.05,
          0.1,
          0.2,
          0.5,
          1.0,
          2.0,
          5.0,
          10.0,
          20.0,
          50.0,
          100.0,
      ],
  )

  _slots_used_percentage = Gauge(
      name="jetstream_slots_used_percentage",
      documentation="The percentage of decode slots currently being used",
      labelnames=["id", "idx"],
  )

  _server_startup_latency = Gauge(
      name="jetstream_server_startup_latency",
      documentation="Total time taken to start the Jetstream server",
      labelnames=["id"],
  )

  _time_to_first_token = Histogram(
      name="jetstream_time_to_first_token",
      documentation="Time to first token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.001,
          0.005,
          0.01,
          0.02,
          0.04,
          0.06,
          0.08,
          0.1,
          0.25,
          0.5,
          0.75,
          1.0,
          2.5,
          5.0,
          7.5,
          10.0,
      ],
  )

  _time_per_output_token = Histogram(
      name="jetstream_time_per_output_token",
      documentation="Average time per output token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.01,
          0.025,
          0.05,
          0.075,
          0.1,
          0.15,
          0.2,
          0.3,
          0.4,
          0.5,
          0.75,
          1.0,
          2.5,
      ],
  )

  _time_per_prefill_token = Histogram(
      name="jetstream_time_per_prefill_token",
      documentation="Prefill time per token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.00001,
          0.00002,
          0.00005,
          0.0001,
          0.0002,
          0.0005,
          0.001,
          0.002,
          0.005,
          0.01,
          0.02,
          0.05,
          0.1,
      ],
  )

  _time_per_request = Histogram(
      name="jetstream_time_per_request",
      documentation="End to end request latency in seconds",
      labelnames=["id"],
      buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
  )

  _wait_time_per_request = Histogram(
      name="jetstream_wait_time_per_request",
      documentation="The total time each request is not actively being prefilled or decoded",
      labelnames=["id"],
      buckets=[
          0.01,
          0.02,
          0.05,
          0.1,
          0.2,
          0.5,
          1.0,
          2.0,
          5.0,
          10.0,
          20.0,
          50.0,
          100.0,
      ],
  )

  def get_prefill_backlog_metric(self):
    return self._prefill_backlog.labels(id=self._id)

  def get_transfer_backlog_metric(self, idx: int):
    return self._transfer_backlog.labels(id=self._id, idx=idx)

  def get_generate_backlog_metric(self, idx: int):
    return self._generate_backlog.labels(id=self._id, idx=idx)

  def get_queue_duration(self):
    return self._queue_duration.labels(id=self._id)

  def get_slots_used_percentage_metric(self, idx: int):
    return self._slots_used_percentage.labels(id=self._id, idx=idx)

  def get_server_startup_latency_metric(self):
    return self._server_startup_latency.labels(id=self._id)

  def get_time_to_first_token(self):
    return self._time_to_first_token.labels(id=self._id)

  def get_time_per_output_token(self):
    return self._time_per_output_token.labels(id=self._id)

  def get_time_per_prefill_token(self):
    return self._time_per_prefill_token.labels(id=self._id)

  def get_time_per_request(self):
    return self._time_per_request.labels(id=self._id)

  def get_wait_time_per_request(self):
    return self._wait_time_per_request.labels(id=self._id)
