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
import os
import signal
import threading
import traceback
from typing import Any, Type


import grpc
import jax
from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import aot_utils, engine_api

from prometheus_client import start_http_server

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
    jax_padding: bool = True,
    metrics_server_config: config_lib.MetricsServerConfig | None = None,
    enable_jax_profiler: bool = False,
    jax_profiler_port: int = 9999,
    enable_model_warmup: bool = False,
) -> JetStreamServer:
  """Runs a server with a specified config.

  Args:
    port: Port on which the server will be made available.
    config: A ServerConfig to config engine, model, device slices, etc.
    devices: Device objects, will be used to get engine with proper slicing.
    credentials: Should use grpc credentials by default.
    threads: Number of RPC handlers worker threads. This should be at least
      equal to the decoding batch size to fully saturate the decoding queue.
    jax_padding: The flag to enable JAX padding during tokenization.
    metrics_server_config: The config to enable Promethus metric server.
    enable_jax_profiler: The flag to enable JAX profiler server.
    jax_profiler_port: The port JAX profiler server (default to 9999).
    enable_model_warmup: The flag to enable model server warmup with AOT.

  Returns:
    JetStreamServer that wraps the grpc server and orchestrator driver.
  """
  logging.info("Kicking off gRPC server.")
  engines = config_lib.get_engines(config, devices=devices)
  prefill_params = [pe.load_params() for pe in engines.prefill_engines]
  generate_params = [ge.load_params() for ge in engines.generate_engines]
  shared_params = [ie.load_params() for ie in engines.interleaved_engines]
  logging.info("Loaded all weights.")
  interleaved_mode = (
      len(config.prefill_slices) + len(config.generate_slices) == 0
  )

  # Setup Prometheus server
  metrics_collector: JetstreamMetricsCollector = None
  if metrics_server_config and metrics_server_config.port:
    logging.info(
        "Starting Prometheus server on port %d", metrics_server_config.port
    )
    start_http_server(metrics_server_config.port)
    metrics_collector = JetstreamMetricsCollector()
  else:
    logging.info(
        "Not starting Prometheus server: --prometheus_port flag not set"
    )

  prefill_engines = engines.prefill_engines + engines.interleaved_engines
  generate_engines = engines.generate_engines + engines.interleaved_engines
  prefill_params = prefill_params + shared_params
  generate_params = generate_params + shared_params

  if prefill_engines is None:
    prefill_engines = []
  if generate_engines is None:
    generate_engines = []
  if prefill_params is None:
    prefill_params = []
  if generate_params is None:
    generate_params = []

  if enable_model_warmup:
    prefill_engines = [engine_api.JetStreamEngine(pe) for pe in prefill_engines]
    generate_engines = [
        engine_api.JetStreamEngine(ge) for ge in generate_engines
    ]

    try:
      _ = aot_utils.layout_params_and_compile_executables(
          prefill_engines,  # pylint: disable=protected-access
          generate_engines,  # pylint: disable=protected-access
          prefill_params,  # pylint: disable=protected-access
          generate_params,  # pylint: disable=protected-access
      )

    except ValueError as e:
      print(f"Model warmup encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)

  driver = orchestrator.Driver(
      prefill_engines=prefill_engines,
      generate_engines=generate_engines,
      prefill_params=prefill_params,
      generate_params=generate_params,
      interleaved_mode=interleaved_mode,
      jax_padding=jax_padding,
      metrics_collector=metrics_collector,
      is_ray_backend=config.is_ray_backend,
  )
  # We default threads to the total number of concurrent allowed decodes,
  # to make sure we can fully saturate the model. Set default minimum to 64.
  threads = threads or max(driver.get_total_concurrent_requests(), 64)
  jetstream_server = JetStreamServer(driver, threads, port, credentials)
  logging.info("Starting server on port %d with %d threads", port, threads)

  jetstream_server.start()

  # Setup Jax Profiler
  if enable_jax_profiler:
    logging.info("Starting JAX profiler server on port %s", jax_profiler_port)
    jax.profiler.start_server(jax_profiler_port)
  else:
    logging.info("Not starting JAX profiler server: %s", enable_jax_profiler)

  # Start profiling server by default for proxy backend.
  if jax.config.jax_platforms and "proxy" in jax.config.jax_platforms:
    from jetstream.core.utils import proxy_util  # pylint: disable=import-outside-toplevel

    thread = threading.Thread(
        target=proxy_util.start_profiling_server, args=(jax_profiler_port,)
    )
    thread.run()

  return jetstream_server


def get_devices() -> Any:
  """Gets devices."""
  # TODO: Add more logs for the devices.
  devices = jax.devices()
  logging.info("Using devices: %d", len(devices))
  return devices
