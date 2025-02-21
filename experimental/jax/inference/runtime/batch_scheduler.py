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

"""batch scheduler"""

import enum
import queue
import dataclasses
from inference.runtime.kv_cache import KVCacheManager
from inference.runtime.request_type import PrefillRequest, GenerateRequest, GenerateState


@dataclasses.dataclass
class PrefillPagesUpdate:
  page_indices: list[int]  # length: chunk_size // page_size


@dataclasses.dataclass
class GenerateStatePageUpdate:
  slot: int
  page_idx: int
  mapped_idx: int


@dataclasses.dataclass
class Schedule:
  schedule_prefill: bool
  prefill_request: PrefillRequest
  prefill_pages_update: PrefillPagesUpdate
  schedule_generate: bool
  new_generate_requests: list[GenerateRequest]
  generate_state_page_updates: list[GenerateStatePageUpdate]


@enum.unique
class SchedulePolicy(enum.Enum):
  OFFLINE = enum.auto()
  ONLINE = enum.auto()


class BatchScheduler:

  def __init__(
      self,
      kv_cache_manager: KVCacheManager,
      batch_size: int,
      max_seq_len: int,
      schedule_policy: SchedulePolicy = SchedulePolicy.OFFLINE,
  ):
    self.prefill_queue: queue.Queue[PrefillRequest] = queue.Queue()
    self.generate_queue: queue.Queue[GenerateRequest] = queue.Queue()
    self.kv_manager = kv_cache_manager
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.schedule_policy = schedule_policy

  def enqueue_prefill_req(self, req: PrefillRequest):
    self.prefill_queue.put(req)

  def enqueue_generate_req(self, req: GenerateRequest):
    self.generate_queue.put(req)

  def schedule(
      self,
      active_prefill: PrefillRequest | None,
      generate_state: GenerateState,
  ) -> Schedule | None:
    """Schedule the workload for next iteration. Only host state is
    updated in the schedule function.
    """
    avail_slot_size = generate_state.available_slots.qsize()
    next_prefill_req = active_prefill
    prefill_pages_update = None
    next_generate_reqs = []
    generate_state_page_updates = []

    schedule_prefill = False
    schedule_generate = False

    # Schedule new prefill req, if no active prefill request.
    if not next_prefill_req:
      if avail_slot_size > 0:
        try:
          next_prefill_req = self.prefill_queue.get_nowait()
          if not next_prefill_req:
            return None
        except queue.Empty:
          pass

    if next_prefill_req:
      cur_prompt_chunk_len = next_prefill_req.chunk_size
      total_len = len(next_prefill_req.unpadded_token_ids)
      if (
          total_len
          <= (next_prefill_req.chunk_idx + 1) * next_prefill_req.chunk_size
      ):
        cur_prompt_chunk_len = (
            total_len - next_prefill_req.chunk_idx * next_prefill_req.chunk_size
        )
      alloced_pages = self.kv_manager.alloc_prefill_hbm_pages(
          cur_prompt_chunk_len
      )
      if len(alloced_pages) == 0:
        # TODO: introduce priority for the request and better
        # eviction algorithm.
        raise NotImplementedError("Eviction is not supported yet")
      else:
        start_idx = (
            next_prefill_req.chunk_idx * next_prefill_req.chunk_size
        ) // self.kv_manager.page_size
        for i, page in enumerate(alloced_pages):
          next_prefill_req.page_indices[start_idx + i] = page
        prefill_pages_update = PrefillPagesUpdate(alloced_pages)

    # Schedule new generate reqs and allocate memory for all reqs.
    with generate_state.map_mutex:
      if (
          self.schedule_policy == SchedulePolicy.ONLINE
          or not next_prefill_req
          or (
              len(generate_state.active_slot_req_map)
              + self.generate_queue.qsize()
              > 0.95 * self.batch_size
          )
      ):
        # Add new generate request to the slots.
        while (
            generate_state.available_slots.qsize() > 0
            and self.generate_queue.qsize() > 0
        ):
          gr = self.generate_queue.get_nowait()
          if not gr:
            return None
          slot = generate_state.available_slots.get_nowait()
          gr.slot = slot
          generate_state.active_slot_req_map[slot] = gr
          next_generate_reqs.append(gr)

        # Check and alloc memory for generate.
        alloced_pages = self.kv_manager.alloc_hbm_pages(
            len(generate_state.active_slot_req_map)
        )
        if (
            len(generate_state.active_slot_req_map) != 0
            and len(alloced_pages) == 0
        ):
          raise NotImplementedError(
              "Eviction isn't supported yet, please set a lower value for batch_size"
          )

        page_to_use = 0
        for slot, req in generate_state.active_slot_req_map.items():
          idx = req.pos // self.kv_manager.page_size
          if req.pos % self.kv_manager.page_size != 0:
            continue
          if idx >= len(req.page_indices):
            continue

          req.page_indices[idx] = alloced_pages[page_to_use]
          generate_state_page_updates.append(
              GenerateStatePageUpdate(
                  slot=slot,
                  page_idx=idx,
                  mapped_idx=alloced_pages[page_to_use],
              )
          )
          page_to_use += 1

        self.kv_manager.free_hbm_pages(alloced_pages[page_to_use:])

        if len(generate_state.active_slot_req_map) == 0:
          schedule_generate = False
        else:
          schedule_generate = True

    if next_prefill_req:
      schedule_prefill = True
    else:
      schedule_prefill = False

    if not schedule_prefill and not schedule_generate:
      # Nothing got scheduled, busy waiting for either prefill
      # or generate queue to have pending request.
      while True:
        if self.prefill_queue.qsize() > 0 or self.generate_queue.qsize() > 0:
          return self.schedule(active_prefill, generate_state)

    return Schedule(
        schedule_prefill=schedule_prefill,
        prefill_request=next_prefill_req,
        prefill_pages_update=prefill_pages_update,
        schedule_generate=schedule_generate,
        new_generate_requests=next_generate_reqs,
        generate_state_page_updates=generate_state_page_updates,
    )
