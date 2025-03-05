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

"""offline driver module"""


import datetime
import multiprocessing as mp
import threading
import queue
import math
from typing import Sequence
import jax
from inference import parallel
from inference.runtime.engine import Engine, ModelLoadParams, InferenceParams, EngineMode, OfflineChannel
from inference.runtime.request_type import *


class OfflineInference:

  def __init__(
      self,
      model_id: str = "meta-llama/Llama-2-7b-chat-hf",
      num_engines: int = 1,
      enable_multiprocessing: bool = False,
  ):
    self.num_engines = num_engines
    self.req_queues: list[mp.Queue | queue.Queue] = []
    self.res_queues: list[mp.Queue | queue.Queue] = []
    self._next_pick_engine_index = 0
    self._running_pool: list[mp.Process | threading.Thread] = []
    self._engine_started_events: list = []
    self._completion_events: list = []
    for i in range(num_engines):
      if enable_multiprocessing:
        self.req_queue.append(mp.Queue())
        self.res_queue.append(mp.Queue())
        self._engine_started_events.append(mp.Event())
        self._completion_events.append(mp.Event())
        execution = mp.Process(
            target=self.launch_engine,
            args=(
                self.req_queue[i],
                self.res_queue[i],
                ModelLoadParams(model_id=model_id),
                self._engine_started_events[i],
                self._completion_events[i],
            ),
        )
      else:
        self.req_queues.append(queue.Queue())
        self.res_queues.append(queue.Queue())
        self._engine_started_events.append(threading.Event())
        self._completion_events.append(threading.Event())

        execution = threading.Thread(
            target=self.launch_engine,
            args=(
                self.req_queues[i],
                self.res_queues[i],
                ModelLoadParams(model_id=model_id),
                self._engine_started_events[i],
                self._completion_events[i],
            ),
        )
      self._running_pool.append(execution)

    for e in self._running_pool:
      e.start()

    for started_event in self._engine_started_events:
      while not started_event.is_set():
        started_event.wait()

  def launch_engine(
      self,
      req_queue,
      res_queue,
      model_load_params,
      started_event,
      completion_event,
  ):
    devices = jax.devices()
    mesh = parallel.create_device_mesh(
        devices,
        (len(devices),),
    )
    engine = Engine(
        mesh=mesh,
        model_load_params=model_load_params,
        inference_params=InferenceParams(),
        mode=EngineMode.OFFLINE,
        channel=OfflineChannel(
            req_queue=req_queue,
            res_queue=res_queue,
        ),
    )
    engine.start()
    started_event.set()
    while not completion_event.is_set():
      completion_event.wait()
    engine.stop()
    return

  def __call__(self, prompts: Sequence[str]) -> tuple[list[Response], float]:
    for i in range(self.num_engines):
      while not self._engine_started_events[i].is_set():
        self._engine_started_events[i].wait()

    print(
        f"All the engines started: {datetime.datetime.now()}, processing requests..."
    )

    num_reqs_per_engine = math.ceil(len(prompts) / self.num_engines)
    res = []

    for i in range(self.num_engines):
      idx = i * num_reqs_per_engine
      prompts_slice = prompts[idx : idx + num_reqs_per_engine]
      for p in prompts_slice:
        self.req_queues[i].put(OfflineRequest(prompt=p))

    for i in range(self.num_engines):
      if i != self.num_engines - 1:
        num_reqs = num_reqs_per_engine
      else:
        num_reqs = len(prompts) - (i * num_reqs_per_engine)
      for _ in range(num_reqs):
        res.append(self.res_queues[i].get())

    print("Offline inference ends:", datetime.datetime.now())

    for i in range(self.num_engines):
      self._completion_events[i].set()

    for e in self._running_pool:
      e.join()

    return res
