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
from prometheus_client import Counter, Gauge, Histogram
from jetstream.engine.token_utils import DEFAULT_PREFILL_BUCKETS

# Initialize the unique ID for labeling metrics
_id = os.getenv("HOSTNAME", shortuuid.uuid())

# Registry for storing metric objects
_metrics_registry = {
    "jetstream_prefill_backlog_size": Gauge(
        name="jetstream_prefill_backlog_size",
        documentation="Size of prefill queue",
        labelnames=["id"],
    ),
    "jetstream_transfer_backlog_size": Gauge(
        name="jetstream_transfer_backlog_size",
        documentation="Size of transfer queue",
        labelnames=["id", "idx"],
    ),
    "jetstream_generate_backlog_size": Gauge(
        name="jetstream_generate_backlog_size",
        documentation="Size of generate queue",
        labelnames=["id", "idx"],
    ),
    "jetstream_queue_duration": Histogram(
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
    ),
    "jetstream_slots_used_percentage": Gauge(
        name="jetstream_slots_used_percentage",
        documentation="The percentage of decode slots currently being used",
        labelnames=["id", "idx"],
    ),
    "jetstream_server_startup_latency": Gauge(
        name="jetstream_server_startup_latency",
        documentation="Total time taken to start the Jetstream server",
        labelnames=["id"],
    ),
    "jetstream_request_input_length": Histogram(
        name="jetstream_request_input_length",
        documentation="Number of input tokens per request",
        labelnames=["id"],
        buckets=DEFAULT_PREFILL_BUCKETS,
    ),
    "jetstream_request_output_length": Histogram(
        name="jetstream_request_output_length",
        documentation="Number of output tokens per request",
        labelnames=["id"],
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
    ),
    "jetstream_request_success_count": Counter(
        name="jetstream_request_success_count",
        documentation="Number of requests successfully completed",
        labelnames=["id"],
    ),
    "jetstream_time_to_first_token": Histogram(
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
    ),
    "jetstream_time_per_output_token": Histogram(
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
    ),
    "jetstream_time_per_prefill_token": Histogram(
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
    ),
    "jetstream_time_per_request": Histogram(
        name="jetstream_time_per_request",
        documentation="End to end request latency in seconds",
        labelnames=["id"],
        buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ),
    "jetstream_wait_time_per_request": Histogram(
        name="jetstream_wait_time_per_request",
        documentation="Time each request is not being prefilled or decoded",
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
    ),
    "jetstream_total_tokens_in_current_batch": Gauge(
        name="jetstream_total_tokens_in_current_batch",
        documentation="Total number of tokens in the decode batch",
        labelnames=["id", "idx"],
    ),
}


# Function to retrieve a metric with specified labels
def get_metric(metric_name, **labels):
  if metric_name not in _metrics_registry:
    raise ValueError(f"Metric {metric_name} not found in registry.")

  metric = _metrics_registry[metric_name]

  # Automatically add the 'id' label if it's required by the metric
  if "id" in metric._labelnames:  # pylint: disable=protected-access
    labels["id"] = _id

  # Check for any missing labels
  missing_labels = set(metric._labelnames) - labels.keys()  # pylint: disable=protected-access
  if missing_labels:
    raise ValueError(
        f"Missing labels for metric {metric_name}: {', '.join(missing_labels)}"
    )

  return metric.labels(**labels)
