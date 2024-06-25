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
"""Proxy util functions."""

import dataclasses
import logging
import jax
import time
from fastapi import FastAPI
import uvicorn


# TODO: add a manner way to terminate.
def start_profiling_server(port: int):

  logging.info("Starting JAX profiler server on port %s", port)
  app = FastAPI()

  @dataclasses.dataclass
  class ProfilingConfig:
    seconds: int
    output_dir: str

  @app.post("/profiling")
  async def profiling(pc: ProfilingConfig):
    jax.profiler.start_trace(pc.output_dir)
    logging.info("Capturing the profiling data for next %s seconds", pc.seconds)
    time.sleep(pc.seconds)
    logging.info("Writing profiling data to %s", pc.output_dir)
    jax.profiler.stop_trace()
    return {"response": "profiling completed"}

  @app.get("/")
  async def root():
    return {"message": "Hello from proxy profiling server"}

  uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
