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

"""Contains common functions for a JetStream core server.

See implementations/*/sever.py for examples.
"""

import asyncio
from concurrent import futures
import logging
import threading
from typing import Any, Type

import grpc
import jax
from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core.proto import jetstream_pb2_grpc


_HOST = "[::]"


class JetStreamServer:
  """JetStream grpc server."""

  def __init__(
      self, driver: orchestrator.Driver, threads: int, port, credentials
  ):
    self._executor = futures.ThreadPoolExecutor(max_workers=threads)

    self._loop = asyncio.new_event_loop()
    self._loop.set_default_executor(self._executor)
    self._loop_thread = threading.Thread(target=self._loop.run_forever)
    self._loop_thread.start()

    async def do_init():
      self._grpc_server = grpc.aio.server(
          self._executor,
      )

    asyncio.run_coroutine_threadsafe(do_init(), loop=self._loop).result()
    self._driver = driver
    jetstream_pb2_grpc.add_OrchestratorServicer_to_server(
        orchestrator.LLMOrchestrator(driver=self._driver), self._grpc_server
    )
    self._grpc_server.add_secure_port(f"{_HOST}:{port}", credentials)

  async def _async_start(self) -> None:
    await self._grpc_server.start()

  def start(self) -> None:
    asyncio.run_coroutine_threadsafe(
        self._async_start(), loop=self._loop
    ).result()

  async def _async_stop(self) -> None:
    await self._grpc_server.stop(grace=10)

  def stop(self) -> None:
    # Gracefully clean up threads in the orchestrator.
    self._driver.stop()
    asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result()
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._loop_thread.join()

  def wait_for_termination(self) -> None:
    try:
      asyncio.run_coroutine_threadsafe(
          self._grpc_server.wait_for_termination(), self._loop
      ).result()
    finally:
      self.stop()


def run(
    port: int,
    config: Type[config_lib.ServerConfig],
    devices: Any,
    credentials: Any = grpc.insecure_server_credentials(),
    threads: int | None = None,
) -> JetStreamServer:
  """Runs a server with a specified config.

  Args:
    port: Port on which the server will be made available.
    config: A ServerConfig to config engine, model, device slices, etc.
    devices: Device objects, will be used to get engine with proper slicing.
    credentials: Should use grpc credentials by default.
    threads: Number of RPC handlers worker threads. This should be at least
      equal to the decoding batch size to fully saturate the decoding queue.

  Returns:
    JetStreamServer that wraps the grpc server and orchestrator driver.
  """
  logging.info("Kicking off gRPC server.")
  engines = config_lib.get_engines(config, devices=devices)
  prefill_params = [pe.load_params() for pe in engines.prefill_engines]
  generate_params = [ge.load_params() for ge in engines.generate_engines]
  shared_params = [ie.load_params() for ie in engines.interleaved_engines]
  logging.info("Loaded all weights.")
  driver = orchestrator.Driver(
      prefill_engines=engines.prefill_engines + engines.interleaved_engines,
      generate_engines=engines.generate_engines + engines.interleaved_engines,
      prefill_params=prefill_params + shared_params,
      generate_params=generate_params + shared_params,
  )
  # We default threads to the total number of concurrent allowed decodes,
  # to make sure we can fully saturate the model. Set default minimum to 64.
  threads = threads or max(driver.get_total_concurrent_requests(), 64)
  jetstream_server = JetStreamServer(driver, threads, port, credentials)
  logging.info("Starting server on port %d with %d threads", port, threads)

  jetstream_server.start()
  return jetstream_server


def get_devices() -> Any:
  """Gets devices locally."""
  # Run interleaved engine on local device.
  devices = jax.devices()
  logging.info("Using local devices for interleaved serving: %d", len(devices))
  return devices
