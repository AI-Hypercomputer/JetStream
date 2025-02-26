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

"""engine module"""

import enum
import dataclasses
import datetime
import math
import queue
import threading
from typing import Any
import uuid
import jax
import jax.profiler
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from inference import nn
from inference import parallel
from inference.config.config import InferenceParams
from inference.model import ModelRegistry, SamplingParams
from inference.model.llama import LlamaForCausalLM
from inference.runtime.kv_cache import *
from inference.runtime.request_type import *
from inference.runtime.kv_cache import KVCacheStorage, KVCacheManager
from inference.runtime.batch_scheduler import BatchScheduler, Schedule, SchedulePolicy
from inference.runtime.model_executor import Executor


@enum.unique
class EngineMode(enum.Enum):
  OFFLINE = enum.auto()
  ONLINE = enum.auto()


@dataclasses.dataclass
class OfflineChannel:
  req_queue: queue.Queue[OfflineRequest]
  res_queue: queue.Queue[Response]


@dataclasses.dataclass
class OnlineChannel:
  req_queue: asyncio.Queue[OnlineRequest]
  aio_loop: asyncio.AbstractEventLoop


class Engine:
  """Engine is a wrapper of the model for inference"""

  def __init__(
      self,
      mesh: Mesh,
      inference_params: InferenceParams,
      mode: EngineMode,
      channel: OfflineChannel | OnlineChannel,
      debug_mode: bool,
  ):
    print("Initializing engine")
    self.mesh = mesh
    self.inference_params = inference_params
    model_registry = ModelRegistry()

    print("-" * 40)
    print("Loading tokenizer")
    self.tokenizer = model_registry.load_tokenizer(inference_params.model_id)

    print("-" * 40)
    print("Loading model config")
    model_config = model_registry.load_model_config(inference_params.model_id)
    if debug_mode:
      model_config.num_hidden_layers = 1

    self.model: nn.CausalLM = LlamaForCausalLM(
        model_config,
        parallel.ModelParallelConfig(mesh=self.mesh),
        self.tokenizer.eos_token_id,
        self.inference_params.max_seq_length,
    )

    print("-" * 40)
    if debug_mode:
      print("Initializing random model weights to devices")
      self.weights_dict = self.model.init_weights()
    else:
      print("Loading model weights to host")
      weights_on_host = model_registry.load_weights_to_host(
          model_id=inference_params.model_id,
          num_devices=self.mesh.devices.size,
          model_config=model_config,
          dtype=jnp.bfloat16,
      )
      print("Loading model weights to devices")
      self.weights_dict = self.model.load_weights_dict(weights_on_host)

    print("-" * 40)
    print("Initializing KV cache storage/manager")
    # init kv cache
    self.kv_storage = KVCacheStorage(
        mesh=self.mesh,
        model_config=model_config,
        page_size=self.inference_params.page_size,
        hbm_utilization=self.inference_params.hbm_utilization,
    )
    self.kv_manager = KVCacheManager(
        self.kv_storage.num_hbm_pages, self.inference_params.page_size
    )

    self.mode = mode
    self.channel = channel

    if self.mode == EngineMode.OFFLINE:
      mode = SchedulePolicy.OFFLINE
    else:
      assert self.mode == EngineMode.ONLINE
      mode = SchedulePolicy.ONLINE

    print("-" * 40)
    print("Initializing batch scheduler")
    self.scheduler = BatchScheduler(
        self.kv_manager,
        self.inference_params.batch_size,
        self.inference_params.max_seq_length,
        mode,
    )

    print("-" * 40)
    print("Initializing GenerateState")
    self.active_prefill_request = None
    self.num_pages_per_seq = math.ceil(
        self.inference_params.max_seq_length / self.inference_params.page_size
    )
    slots = queue.SimpleQueue()
    for i in range(self.inference_params.batch_size):
      slots.put(i, block=True)
    self.generate_state = GenerateState(
        token_ids=jnp.zeros(
            shape=(self.inference_params.batch_size,),
            dtype=jnp.int32,
            device=NamedSharding(self.mesh, P(None)),
        ),
        positions=jnp.full(
            shape=(self.inference_params.batch_size,),
            fill_value=-1,
            dtype=jnp.int32,
            device=NamedSharding(self.mesh, P(None)),
        ),
        page_table=jnp.full(
            shape=(
                self.inference_params.batch_size,
                self.num_pages_per_seq,
            ),
            fill_value=self.kv_manager.dummy_page_idx,
            dtype=jnp.int32,
            device=NamedSharding(self.mesh, P(None, None)),
        ),
        available_slots=slots,
        active_slot_req_map={},
    )
    self.sample_params = SamplingParams(
        temperature=jax.device_put(
            jnp.asarray((1.0), dtype=jnp.float32), NamedSharding(self.mesh, P())
        ),
        top_k=jax.device_put(
            jnp.asarray((1), dtype=jnp.int32), NamedSharding(self.mesh, P())
        ),
        rng=jax.device_put(jax.random.key(0), NamedSharding(self.mesh, P())),
    )

    self.model_executor = Executor(
        self.mesh,
        self.weights_dict,
        self.model.jittable_call,
        self.num_pages_per_seq,
        debug_mode=debug_mode,
    )

    print("-" * 40)
    print("Compiling engine")
    self.model_executor.compile(
        self.inference_params.prefill_chunk_sizes,
        self.inference_params.batch_size,
        self.inference_params.max_seq_length,
        self.inference_params.max_input_length,
        self.kv_storage.hbm_kv_caches,
        self.sample_params,
    )

    # running loop
    self.requests_dict: dict[str, Request] = {}

    print("-" * 40)
    print("Starting threads:", end="")
    if self.mode == EngineMode.OFFLINE:
      print(" offline dequeue,", end="")
      self._dequeue_offline_req_thread = threading.Thread(
          name="dequeue_offline_request", target=self._dequeue_offline_request
      )
    else:
      print(" online dequeue,", end="")
      self._dequeue_online_req_thread = threading.Thread(
          name="_dequeue_online_request", target=self._dequeue_online_request
      )

    # TODO: Assign the max_device_requests_sem number by the
    # device spec and cost model.
    self._max_device_requests_sem = threading.Semaphore(
        self.inference_params.batch_size * 3 // 2
    )
    print(" preprocess,", end="")
    self._preprocess_queue: queue.Queue[Request] = queue.Queue()
    # TODO: Seperate the running loop with the static inference model.
    self._preprocess_thread = threading.Thread(
        name="preprocess", target=self._preprocess
    )
    # Add backpressure to prevent that the inference thread never releases
    # the GIL and keeps dispatching the device program.
    print(" postprocess,", end="")
    self._postprocess_queue: queue.Queue[PostProcessRequest] = queue.Queue(8)
    self._postprocess_thread = threading.Thread(
        name="postprocess", target=self._postprocess
    )

    print(" inference")
    self._inference_thread = threading.Thread(
        name="inference", target=self._inference
    )
    self.total_reqs = 0
    self.complete_reqs = 0
    self.start_time = None

  def start(self):
    jax.profiler.start_server(9999)

    if self.mode == EngineMode.OFFLINE:
      self._dequeue_offline_req_thread.start()
    else:
      self._dequeue_online_req_thread.start()

    self._preprocess_thread.start()
    self._postprocess_thread.start()
    self._inference_thread.start()

    self.start_time = datetime.datetime.now()
    print("-" * 40)
    print("Engine starts: ", self.start_time)

  def stop(self):
    jax.profiler.stop_server()
    # Stop listen to the queue when item is None.
    self.channel.req_queue.put(None, block=True)
    self._preprocess_queue.put(None, block=True)
    self.scheduler.enqueue_prefill_req(None)
    self.scheduler.enqueue_generate_req(None)
    self._postprocess_queue.put(None, block=True)

    if self.mode == EngineMode.OFFLINE:
      self._dequeue_offline_req_thread.join()
    else:
      self._dequeue_online_req_thread.join()

    self._preprocess_thread.join()
    self._inference_thread.join()
    self._postprocess_thread.join()

    stop_time = datetime.datetime.now()
    duration = (stop_time - self.start_time).total_seconds()
    print(f"Engine stops: {stop_time}")
    print("-" * 40)
    print(f"Engine total run time: {duration:.2f} seconds")

  def _dequeue_online_request(self):
    """ "Dequeues online requests and put them in the preprocess queue."""
    while True:
      r = self.channel.req_queue.get(block=True)
      if r is None:
        return
      assert isinstance(r, OnlineRequest)
      req = Request(
          id=uuid.uuid4().hex, prompt=r.prompt, aio_response_queue=r.res_queue
      )
      self._preprocess_queue.put(req, block=True)
      self.requests_dict[req.id] = req

  def _dequeue_offline_request(self):
    """ "Dequeues offline requests and put them in the preprocess queue."""
    while True:
      r = self.channel.req_queue.get(block=True)
      if r is None:
        return
      assert isinstance(r, OfflineRequest)
      req = Request(id=uuid.uuid4().hex, prompt=r.prompt)
      self._preprocess_queue.put(req, block=True)
      self.requests_dict[req.id] = req

  def _encode(self, req: Request) -> tuple[np.ndarray, int]:
    """Encodes the request prompt and trims/pads it to the max input length."""
    token_id_list = self.tokenizer.encode(req.prompt)
    req.prompt_token_ids = token_id_list
    tokens = np.asarray(token_id_list)
    token_len, bound = tokens.size, self.inference_params.max_input_length
    if token_len == bound:
      return tokens, token_len
    elif token_len < bound:
      return np.pad(tokens, (0, bound - token_len)), token_len
    else:
      assert token_len > bound
      req.prompt_token_ids = token_id_list[-bound:]
      return tokens[-bound:], bound

  def _select_chunk_size(self, token_len: int) -> int:
    """Selects a prefill chunk size that is big enough."""
    for size in self.inference_params.prefill_chunk_sizes:
      if token_len <= size:
        return size
    return self.inference_params.prefill_chunk_sizes[-1]

  def _build_prefill_request(self, req: Request) -> PrefillRequest:
    """Builds a prefill request from the original prompt request."""
    padded_tokens, token_len = self._encode(req)
    chunk_size = self._select_chunk_size(token_len)
    pi = self.kv_manager.dummy_page_idx
    dummy_page_indices = [pi] * self.num_pages_per_seq
    device_tokens = jax.device_put(
        padded_tokens, NamedSharding(self.mesh, P(None))
    )
    positions = np.arange(0, device_tokens.shape[0])
    device_positions = jax.device_put(
        positions, NamedSharding(self.mesh, P(None))
    )
    return PrefillRequest(
        id=req.id,
        unpadded_token_ids=req.prompt_token_ids,
        chunk_idx=0,
        chunk_size=chunk_size,
        page_indices=dummy_page_indices,
        device_token_ids=device_tokens,
        device_positions=device_positions,
    )

  def _preprocess(self) -> jax.Array:
    """Converts prompts to prefill requests and give them to the scheduler."""
    while True:
      req = self._preprocess_queue.get(block=True)
      if req is None:
        return
      assert isinstance(req, Request)
      # Don't put too many pending requests to the HBM.
      self._max_device_requests_sem.acquire()
      pr = self._build_prefill_request(req)
      self.scheduler.enqueue_prefill_req(pr)

  def _inference(self) -> jax.Array:
    while True:
      schedule = self.scheduler.schedule(
          self.active_prefill_request, self.generate_state
      )
      if schedule is None:
        return

      assert isinstance(schedule, Schedule)
      input = self.model_executor.prepare_input_and_update_generate_state(
          schedule,
          self.generate_state,
          self.kv_storage.hbm_kv_caches,
          self.sample_params,
          self.inference_params.batch_size,
      )

      output, self.kv_storage.hbm_kv_caches = self.model_executor.execute(input)

      # Prepare for next iteration and post-processed request.
      post_req = PostProcessRequest(
          prefill_request_id=None,
          prefill_token_id=output.prefill_token,
          prefill_done=output.prefill_done,
          generate_active_slots=[],
          generate_active_request_ids=[],
          generate_token_ids=output.generate_tokens,
          generate_done=output.generate_done,
      )

      if schedule.schedule_prefill:
        prefill_req = schedule.prefill_request
        prefill_req.chunk_idx += 1
        start_idx = prefill_req.chunk_idx * prefill_req.chunk_size
        prefill_length = len(prefill_req.unpadded_token_ids)

        if start_idx < prefill_length:
          self.active_prefill_request = prefill_req
        else:
          self.active_prefill_request = None
          post_req.prefill_request_id = schedule.prefill_request.id

          generate_req = GenerateRequest(
              id=prefill_req.id,
              slot=-1,
              pos=prefill_length,
              page_indices=prefill_req.page_indices,
              device_prefill_token_id=output.prefill_token,
          )
          self.scheduler.enqueue_generate_req(generate_req)

      if schedule.schedule_generate:
        self.generate_state.token_ids = output.generate_tokens
        self.generate_state.positions = output.generate_next_pos

        with self.generate_state.map_mutex:
          for (
              slot,
              processed_gr,
          ) in self.generate_state.active_slot_req_map.items():
            processed_gr.pos += 1
            post_req.generate_active_slots.append(slot)
            post_req.generate_active_request_ids.append(processed_gr.id)

      self._postprocess_queue.put(post_req, block=True)

  def _postprocess(self) -> str:
    while True:
      p_req = self._postprocess_queue.get(block=True)
      if p_req is None:
        return

      assert isinstance(p_req, PostProcessRequest)
      p_req.prefill_token_id = np.asarray(p_req.prefill_token_id).item()
      p_req.prefill_done = np.asarray(p_req.prefill_done).item()
      p_req.generate_token_ids = np.asarray(p_req.generate_token_ids).tolist()
      p_req.generate_done = np.asarray(p_req.generate_done).tolist()

      # Free finished slot.
      if len(p_req.generate_active_request_ids) > 0:
        with self.generate_state.map_mutex:
          slot_to_del = []
          for slot in self.generate_state.active_slot_req_map.keys():
            if p_req.generate_done[slot]:
              self.generate_state.available_slots.put(slot, block=True)
              pages_to_free = self.generate_state.active_slot_req_map[
                  slot
              ].page_indices
              self.kv_manager.free_hbm_pages(pages_to_free)
              slot_to_del.append(slot)
          for slot in slot_to_del:
            del self.generate_state.active_slot_req_map[slot]

      # Return generated tokens to the client.
      if p_req.prefill_request_id:
        req = self.requests_dict[p_req.prefill_request_id]
        req.generated_token_ids.append(p_req.prefill_token_id)
        generated_text = self.tokenizer._convert_id_to_token(
            p_req.prefill_token_id
        ).replace("▁", " ")
        req.generated_text += generated_text

        if self.mode == EngineMode.ONLINE:
          self.channel.aio_loop.call_soon_threadsafe(
              req.aio_response_queue.put_nowait,
              Response(generated_text, p_req.prefill_token_id),
          )

        if p_req.prefill_done:
          req.completed = True
          if self.mode == EngineMode.ONLINE:
            self.channel.aio_loop.call_soon_threadsafe(
                req.aio_response_queue.put_nowait,
                Response(generated_text, p_req.prefill_token_id),
            )

          else:
            self.channel.res_queue.put_nowait(
                Response(
                    generated_text=req.generated_text,
                    generated_tokens=req.generated_token_ids,
                    input_tokens=req.prompt_token_ids,
                )
            )

          del self.requests_dict[p_req.prefill_request_id]
          self._max_device_requests_sem.release()

      for slot, req_id in zip(
          p_req.generate_active_slots, p_req.generate_active_request_ids
      ):
        if req_id not in self.requests_dict:
          continue
        req = self.requests_dict[req_id]

        req.generated_token_ids.append(p_req.generate_token_ids[slot])
        generated_text = self.tokenizer._convert_id_to_token(
            p_req.generate_token_ids[slot]
        ).replace("▁", " ")
        req.generated_text += generated_text

        if self.mode == EngineMode.ONLINE:
          self.channel.aio_loop.call_soon_threadsafe(
              req.aio_response_queue.put_nowait,
              Response(
                  generated_text=generated_text,
                  generated_tokens=p_req.generate_token_ids[slot],
              ),
          )

        if p_req.generate_done[slot]:
          req.completed = True
          if self.mode == EngineMode.ONLINE:
            self.channel.aio_loop.call_soon_threadsafe(
                req.aio_response_queue.put_nowait,
                None,
            )
          else:
            self.channel.res_queue.put_nowait(
                Response(
                    generated_text=req.generated_text,
                    generated_tokens=req.generated_token_ids,
                    input_tokens=req.prompt_token_ids,
                )
            )

          self._max_device_requests_sem.release()
          del self.requests_dict[req_id]
