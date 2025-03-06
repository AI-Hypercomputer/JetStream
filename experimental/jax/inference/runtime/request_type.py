"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Request and response classes"""

# TODO: Clean up and merge the request to improve the readability.

import asyncio
from dataclasses import dataclass, field
import numpy as np
import jax
import threading
import queue


@dataclass
class Response:
  generated_text: str
  generated_tokens: list[int] | int
  input_tokens: list[int] | None = None


@dataclass
class OnlineRequest:
  prompt: str
  res_queue: asyncio.Queue[Response]


@dataclass
class OfflineRequest:
  prompt: str


@dataclass
class Request:
  """Request for holding the input and output information"""

  id: str
  prompt: str
  prompt_token_ids: list[int] = field(default_factory=lambda: [])
  generated_text: str = ""
  generated_token_ids: list[int] = field(default_factory=lambda: [])
  aio_response_queue: asyncio.Queue[Response] | None = None
  completed: bool = False


@dataclass
class PrefillRequest:
  """class for new request need to be processed in the prefill phase"""

  id: str
  unpadded_token_ids: list[int]
  chunk_idx: int
  chunk_size: int
  page_indices: list[int]
  device_token_ids: jax.Array
  device_positions: jax.Array


@dataclass
class GenerateRequest:
  """class for new request need to be processed in the generate phase"""

  id: str
  slot: int
  pos: int
  page_indices: list[int]
  device_prefill_token_id: jax.Array


@dataclass
class GenerateState:
  """generate phase state"""

  token_ids: jax.Array  # batch_size
  positions: jax.Array  # batch_size
  page_table: jax.Array  # batch_size, num_pages_per_seq
  available_slots: queue.SimpleQueue
  active_slot_req_map: dict[int, GenerateRequest]
  map_mutex: threading.Lock = threading.Lock()


@dataclass
class PostProcessRequest:
  """Post process request"""

  prefill_request_id: str | None
  prefill_token_id: jax.Array | np.ndarray
  prefill_done: jax.Array | np.ndarray

  generate_active_slots: list[int]
  generate_active_request_ids: list[str]
  generate_token_ids: jax.Array | np.ndarray
  generate_done: jax.Array | np.ndarray
