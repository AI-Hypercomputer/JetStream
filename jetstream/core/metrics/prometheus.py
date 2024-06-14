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
from prometheus_client import Gauge


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
  _slots_used_percentage = Gauge(
      name="jetstream_slots_used_percentage",
      documentation="The percentage of decode slots currently being used",
      labelnames=["id", "idx"],
  )

  def get_prefill_backlog_metric(self):
    return self._prefill_backlog.labels(id=self._id)

  def get_transfer_backlog_metric(self, idx: int):
    return self._transfer_backlog.labels(id=self._id, idx=idx)

  def get_generate_backlog_metric(self, idx: int):
    return self._generate_backlog.labels(id=self._id, idx=idx)

  def get_slots_used_percentage_metric(self, idx: int):
    return self._slots_used_percentage.labels(id=self._id, idx=idx)
