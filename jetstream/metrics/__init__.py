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

import prometheus_client
import shortuuid

instance_uuid = shortuuid.uuid()

# Metrics we care to observe
prefill_backlog : prometheus_client.Gague
jetstream_slots_available_percentage: prometheus_client.Gauge

def register_metrics(): 
    prefill_backlog_metric = prometheus_client.Gauge(
        "jetstream_prefill_backlog_size",
        "Size of prefill queue",
        labelnames=["uuid"],
    )

    jetstream_slots_available_percentage_metric = prometheus_client.Gauge(
        name="jetstream_slots_available_percentage",
        documentation="The percentage of available slots in decode batch",
        labelnames=["uuid", "idx"],
    )