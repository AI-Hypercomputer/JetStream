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

"""Contains helper functions for configuring Jetstream server metrics"""

def get_time_per_prefill_token(request, true_length: int):
  return (
      request.metadata.transfer_enqueue_time
      - request.metadata.prefill_dequeue_time
  ) / true_length


def get_queue_duration(request):
  return (
      # Time in prefill queue
      request.metadata.prefill_dequeue_time
      - request.metadata.prefill_enqueue_time
      # Time in transfer queue
      + request.metadata.transfer_dequeue_time
      - request.metadata.transfer_enqueue_time
      # Time in generate queue
      + request.metadata.generate_dequeue_time
      - request.metadata.generate_enqueue_time
  )


def get_tpot(request, result_tokens, slot):
  return (
      request.metadata.complete_time - request.metadata.transfer_enqueue_time
  ) / result_tokens.get_result_at_slot(slot).lengths


def get_wait_time(request):
  total_time = request.metadata.complete_time - request.metadata.start_time
  prefill_time = (
      request.metadata.transfer_enqueue_time
      - request.metadata.prefill_dequeue_time
  )
  generate_time = (
      request.metadata.complete_time - request.metadata.generate_dequeue_time
  )
  return total_time - prefill_time - generate_time
