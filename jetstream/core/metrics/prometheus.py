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
import re
from typing import Optional
import shortuuid
from prometheus_client import Counter, Gauge, Histogram
from jetstream.engine.token_utils import DEFAULT_PREFILL_BUCKETS


class JetstreamMetricsCollector:
  """Wrapper class should be used to assure all metrics have proper tags"""

  _initialized: bool = False
  _model_name: str
  universal_labels = {"id": os.getenv("HOSTNAME", shortuuid.uuid())}

  def __new__(cls, model_name: Optional[str] = None):
    if not hasattr(cls, "instance"):
      cls.instance = super(JetstreamMetricsCollector, cls).__new__(cls)
    return cls.instance

  def __init__(self, model_name: Optional[str] = None):
    if hasattr(self, "_initialized") and self._initialized:
      return
    self._initialized = True

    # '-'s are common in model names but invalid in prometheus labels
    # these are replaced with '_'s
    if model_name is not None:
      sanitized_model_name = model_name.replace("-", "_")
      if sanitized_model_name == "":
        print("No model name provided, omitting from metrics labels")
      elif not bool(
          re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$", sanitized_model_name)
      ):
        print(
            "Provided model name cannot be used to label prometheus metrics",
            "(does not match ^[a-zA-Z_:][a-zA-Z0-9_:]*$)",
            "omitting from metrics labels",
        )
      else:
        self.universal_labels["model_name"] = sanitized_model_name
    universal_label_names = list(self.universal_labels.keys())

    # Metric definitions
    self._prefill_backlog = Gauge(
        name="jetstream_prefill_backlog_size",
        documentation="Size of prefill queue",
        labelnames=universal_label_names,
    )

    self._transfer_backlog = Gauge(
        name="jetstream_transfer_backlog_size",
        documentation="Size of transfer queue",
        labelnames=universal_label_names + ["idx"],
    )

    self._generate_backlog = Gauge(
        name="jetstream_generate_backlog_size",
        documentation="Size of generate queue",
        labelnames=universal_label_names + ["idx"],
    )

    self._queue_duration = Histogram(
        name="jetstream_queue_duration",
        documentation="The total time each request spends enqueued in seconds",
        labelnames=universal_label_names,
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

    self._slots_used_percentage = Gauge(
        name="jetstream_slots_used_percentage",
        documentation="The percentage of decode slots currently being used",
        labelnames=universal_label_names + ["idx"],
    )
    self._model_load_time = Gauge(
        name="jetstream_model_load_time",
        documentation="Total time taken to load the model",
        labelnames=universal_label_names,
    )
    self._server_startup_latency = Gauge(
        name="jetstream_server_startup_latency",
        documentation="Total time taken to start the Jetstream server",
        labelnames=universal_label_names,
    )
    self._request_input_length = Histogram(
        name="jetstream_request_input_length",
        documentation="Number of input tokens per request",
        labelnames=universal_label_names,
        buckets=DEFAULT_PREFILL_BUCKETS,
    )
    self._request_output_length = Histogram(
        name="jetstream_request_output_length",
        documentation="Number of output tokens per request",
        labelnames=universal_label_names,
        buckets=[
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
            100000,
            200000,
            500000,
            1000000,
            2000000,
        ],
    )
    self._request_success_count = Counter(
        name="jetstream_request_success_count",
        documentation="Number of requests successfully completed",
        labelnames=universal_label_names,
    )

    self._time_to_first_token = Histogram(
        name="jetstream_time_to_first_token",
        documentation="Time to first token per request in seconds",
        labelnames=universal_label_names,
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

    self._time_per_output_token = Histogram(
        name="jetstream_time_per_output_token",
        documentation="Average time per output token per request in seconds",
        labelnames=universal_label_names,
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

    self._time_per_prefill_token = Histogram(
        name="jetstream_time_per_prefill_token",
        documentation="Prefill time per token per request in seconds",
        labelnames=universal_label_names,
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

    self._time_per_request = Histogram(
        name="jetstream_time_per_request",
        documentation="End to end request latency in seconds",
        labelnames=universal_label_names,
        buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )

    self._wait_time_per_request = Histogram(
        name="jetstream_wait_time_per_request",
        documentation="Time each request is not being prefilled or decoded",
        labelnames=universal_label_names,
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
    return self._prefill_backlog.labels(**self.universal_labels)

  def get_transfer_backlog_metric(self, idx: int):
    return self._transfer_backlog.labels(**self.universal_labels, idx=idx)

  def get_generate_backlog_metric(self, idx: int):
    return self._generate_backlog.labels(**self.universal_labels, idx=idx)

  def get_queue_duration(self):
    return self._queue_duration.labels(**self.universal_labels)

  def get_slots_used_percentage_metric(self, idx: int):
    return self._slots_used_percentage.labels(**self.universal_labels, idx=idx)

  def get_server_startup_latency_metric(self):
    return self._server_startup_latency.labels(**self.universal_labels)

  def get_model_load_time_metric(self):
    return self._model_load_time.labels(**self.universal_labels)

  def get_time_to_first_token(self):
    return self._time_to_first_token.labels(**self.universal_labels)

  def get_time_per_output_token(self):
    return self._time_per_output_token.labels(**self.universal_labels)

  def get_time_per_prefill_token(self):
    return self._time_per_prefill_token.labels(**self.universal_labels)

  def get_time_per_request(self):
    return self._time_per_request.labels(**self.universal_labels)

  def get_wait_time_per_request(self):
    return self._wait_time_per_request.labels(**self.universal_labels)

  def get_request_input_length(self):
    return self._request_input_length.labels(**self.universal_labels)

  def get_request_output_length(self):
    return self._request_output_length.labels(**self.universal_labels)

  def get_request_success_count_metric(self):
    return self._request_success_count.labels(**self.universal_labels)
