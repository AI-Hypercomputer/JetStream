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

""" All logic around adding metrics to the Prometheus client registry """

import prometheus_client
import shortuuid

instance_uuid = shortuuid.uuid()

# Metrics we care to observe
prefill_backlog = prometheus_client.Gauge(
    name="jetstream_prefill_backlog_size",
    documentation="Size of prefill queue",
    labelnames=["uuid"],
)

slots_available_percentage = prometheus_client.Gauge(
    name="jetstream_slots_available_percentage",
    documentation="The percentage of available slots in decode batch",
    labelnames=["uuid", "idx"],
)
