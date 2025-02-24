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
from inference.config.config import Config
from inference.runtime.engine import Engine, EngineMode, OfflineChannel
from inference.runtime.request_type import *


class OfflineInference:

  def __init__(
      self,
      model_id: str,
      num_engines: int,
      enable_multiprocessing: bool,
  ):
    self.num_engines = num_engines
    self.req_queues: list[mp.Queue | queue.Queue] = []
    self.res_queues: list[mp.Queue | queue.Queue] = []
    self._running_pool: list[mp.Process | threading.Thread] = []
    self._engine_started_events: list = []
    self._engine_completed_events: list = []
    for i in range(num_engines):
      # Create queues, events for a Process/Thread runner
      if enable_multiprocessing:
        req_q, res_q = mp.Queue(), mp.Queue()
        started, completed = mp.Event(), mp.Event()
        runner = mp.Process
      else:
        req_q, res_q = queue.Queue(), queue.Queue()
        started, completed = threading.Event(), threading.Event()
        runner = threading.Thread
      # Config the runner
      self.req_queues.append(req_q)
      self.res_queues.append(res_q)
      self._engine_started_events.append(started)
      self._engine_completed_events.append(completed)
      execution = runner(
          target=self.launch_engine,
          args=(
              model_id,
              self.req_queues[i],
              self.res_queues[i],
              self._engine_started_events[i],
              self._engine_completed_events[i],
          ),
      )
      self._running_pool.append(execution)

    # Launch the runners
    for e in self._running_pool:
      e.start()

    for started_event in self._engine_started_events:
      while not started_event.is_set():
        started_event.wait()

  def launch_engine(
      self,
      model_id,
      req_queue,
      res_queue,
      started_event,
      completion_event,
  ):
    devices = jax.devices()
    mesh = parallel.create_device_mesh(
        devices=devices,
        shape=(len(devices), 1),
    )
    engine = Engine(
        mesh=mesh,
        inference_params=Config.get(model_id),
        mode=EngineMode.OFFLINE,
        channel=OfflineChannel(
            req_queue=req_queue,
            res_queue=res_queue,
        ),
        debug_mode=False,
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

    print(f"Offline inference begins: {datetime.datetime.now()} ...")

    num_prompts = len(prompts)
    for p in range(num_prompts):
      i = p % self.num_engines
      self.req_queues[i].put(OfflineRequest(prompt=prompts[p]))

    res = []
    for p in range(num_prompts):
      i = p % self.num_engines
      res.append(self.res_queues[i].get())

    print("Offline inference ends:", datetime.datetime.now())

    for i in range(self.num_engines):
      self._engine_completed_events[i].set()

    for e in self._running_pool:
      e.join()

    return res
