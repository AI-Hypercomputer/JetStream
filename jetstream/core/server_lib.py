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
import gc
import logging
import os
import signal
import sys
import threading
import time
import traceback
import importlib
from typing import Any, Type


import grpc
import jax
from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core import prefix_cache
from jetstream.core.lora import adapter_tensorstore as adapterstore
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import warmup_utils, engine_api

from prometheus_client import start_http_server

_HOST = "[::]"

# Create seperate logger to log all INFO message for this module. These show
# stages of server startup and inform user if server is ready to take requests.
# The default logger created in orchestrator.py only logs WARNINGs and above
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

info_handler = logging.StreamHandler(sys.stdout)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)


class JetStreamServer:
  """JetStream grpc server."""

  def __init__(
      self,
      driver: orchestrator.Driver,
      threads: int,
      port,
      credentials,
      enable_llm_inference_pool=False,
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

    if enable_llm_inference_pool:
      module_name = "jetstream.core.lora.multi_lora_inference_api"
      multi_lora_inference = importlib.import_module(module_name)

      module_name = "jetstream.core.proto.multi_lora_decoding_pb2_grpc"
      multi_lora_decoding_pb2_grpc = importlib.import_module(module_name)

      multi_lora_decoding_pb2_grpc.add_v1Servicer_to_server(
          multi_lora_inference.MultiLoraManager(driver=self._driver),
          self._grpc_server,
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


def create_driver(
    config: Type[config_lib.ServerConfig],
    devices: Any,
    jax_padding: bool = True,
    metrics_collector: JetstreamMetricsCollector | None = None,
    enable_model_warmup: bool = False,
    multi_sampling: bool = False,
    lora_input_adapters_path: str | None = None,
    prefix_caching_config: config_lib.PrefixCachingConfig | None = None,
):
  """Creates a driver with a specified config.

  Args:
    config: A ServerConfig to config engine, model, device slices, etc.
    devices: Device objects, will be used to get engine with proper slicing.
    jax_padding: The flag to enable JAX padding during tokenization.
    metrics_collector: The JetStream Prometheus metric collector.
    enable_model_warmup: The flag to enable model server warmup.
    multi_sampling: The flag to enable multi-sampling.
    prefix_caching_config: Config to prefix caching. Disable if None.

  Returns:
    An orchestrator driver.
  """
  engines = config_lib.get_engines(config, devices=devices)
  model_load_start_time = time.time()
  prefill_params = [pe.load_params() for pe in engines.prefill_engines]
  generate_params = [ge.load_params() for ge in engines.generate_engines]
  shared_params = [ie.load_params() for ie in engines.interleaved_engines]
  logger.info("Loaded all weights.")
  if metrics_collector:
    metrics_collector.get_model_load_time_metric().set(
        time.time() - model_load_start_time
    )
  interleaved_mode = (
      len(config.prefill_slices) + len(config.generate_slices) == 0
  )

  prefill_adapterstore = []
  generate_adapterstore = []
  shared_adapterstore = []

  if lora_input_adapters_path:
    # TODO: Make hbm_memory_budget and cpu_memory_budget configurable
    for pe in engines.prefill_engines:
      prefill_adapterstore.append(
          adapterstore.AdapterTensorStore(
              engine=pe,
              adapters_dir_path=lora_input_adapters_path,
              hbm_memory_budget=20 * (1024**3),  # 20 GB HBM
              cpu_memory_budget=100 * (1024**3),  # 100 GB RAM
              total_slots=pe.max_concurrent_decodes,
          )
      )

    for ge in engines.generate_engines:
      generate_adapterstore.append(
          adapterstore.AdapterTensorStore(
              engine=ge,
              adapters_dir_path=lora_input_adapters_path,
              hbm_memory_budget=20 * (1024**3),  # 20 GB HBM
              cpu_memory_budget=100 * (1024**3),  # 100 GB RAM
              total_slots=ge.max_concurrent_decodes,
          )
      )

    for ie in engines.interleaved_engines:
      shared_adapterstore.append(
          adapterstore.AdapterTensorStore(
              engine=ie,
              adapters_dir_path=lora_input_adapters_path,
              hbm_memory_budget=20 * (1024**3),  # 20 GB HBM
              cpu_memory_budget=100 * (1024**3),  # 100 GB RAM
              total_slots=ie.max_concurrent_decodes,
          )
      )

  prefill_engines = engines.prefill_engines + engines.interleaved_engines
  generate_engines = engines.generate_engines + engines.interleaved_engines
  prefill_params = prefill_params + shared_params
  generate_params = generate_params + shared_params
  prefill_adapterstore += shared_adapterstore
  generate_adapterstore += shared_adapterstore

  if prefill_engines is None:
    prefill_engines = []  # pragma: no branch
  if generate_engines is None:
    generate_engines = []  # pragma: no branch
  if prefill_params is None:
    prefill_params = []  # pragma: no branch
  if generate_params is None:
    generate_params = []  # pragma: no branch

  if enable_model_warmup:
    prefill_engines = [engine_api.JetStreamEngine(pe) for pe in prefill_engines]
    generate_engines = [
        engine_api.JetStreamEngine(ge) for ge in generate_engines
    ]

    try:
      _ = warmup_utils.layout_params_and_compile_executables(
          prefill_engines,  # pylint: disable=protected-access
          generate_engines,  # pylint: disable=protected-access
          prefill_params,  # pylint: disable=protected-access
          generate_params,  # pylint: disable=protected-access
      )

    except ValueError as e:
      print(f"Model warmup encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)

  prefix_cache_inst = None
  if prefix_caching_config is not None:
    prefix_cache_inst = prefix_cache.PrefixCache(
        hbm_bytes=prefix_caching_config.max_hbm_byte,
        dram_bytes=prefix_caching_config.max_dram_byte,
    )

  return orchestrator.Driver(
      prefill_engines=prefill_engines,
      generate_engines=generate_engines,
      prefill_params=prefill_params,
      generate_params=generate_params,
      prefill_adapterstore=prefill_adapterstore,
      generate_adapterstore=generate_adapterstore,
      interleaved_mode=interleaved_mode,
      jax_padding=jax_padding,
      metrics_collector=metrics_collector,
      is_ray_backend=config.is_ray_backend,
      multi_sampling=multi_sampling,
      prefix_cache_inst=prefix_cache_inst,
  )


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
    multi_sampling: bool = False,
    lora_input_adapters_path: str | None = None,
    prefix_caching_config: config_lib.PrefixCachingConfig | None = None,
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
    metrics_server_config: The config to enable Prometheus metric server.
    enable_jax_profiler: The flag to enable JAX profiler server.
    jax_profiler_port: The port JAX profiler server (default to 9999).
    enable_model_warmup: The flag to enable model server warmup.
    multi_sampling: The flag to enable multi-sampling.
    lora_input_adapters_path: Input path for all lora adapters.
    prefix_caching_config: Config to prefix caching. Disable if None.

  Returns:
    JetStreamServer that wraps the grpc server and orchestrator driver.
  """
  server_start_time = time.time()
  logger.info("Kicking off gRPC server.")
  # Setup Prometheus server
  metrics_collector: JetstreamMetricsCollector = None
  if metrics_server_config and metrics_server_config.port:
    logger.info(
        "Starting Prometheus server on port %d", metrics_server_config.port
    )
    start_http_server(metrics_server_config.port)
    metrics_collector = JetstreamMetricsCollector(
        model_name=metrics_server_config.model_name
    )
  else:
    logger.info(
        "Not starting Prometheus server: --prometheus_port flag not set"
    )

  if multi_sampling and lora_input_adapters_path:
    raise ValueError("LoRA adapters is not enabled for multi_sampling mode.")

  driver = create_driver(
      config,
      devices,
      jax_padding,
      metrics_collector,
      enable_model_warmup,
      multi_sampling,
      lora_input_adapters_path,
      prefix_caching_config=prefix_caching_config,
  )
  # We default threads to the total number of concurrent allowed decodes,
  # to make sure we can fully saturate the model. Set default minimum to 64.
  threads = threads or max(driver.get_total_concurrent_requests(), 64)
  enable_llm_inference_pool = False
  if lora_input_adapters_path:
    enable_llm_inference_pool = True
  jetstream_server = JetStreamServer(
      driver, threads, port, credentials, enable_llm_inference_pool
  )
  logging.info("Starting server on port %d with %d threads", port, threads)

  # Tweak gc config.
  # Force a gen 2 collection here.
  gc.collect(generation=2)
  # Freeze objects currently tracked and ignore them in future gc runs.
  gc.freeze()
  allocs, gen1, gen2 = gc.get_threshold()
  allocs = config.gc_gen0_allocs
  gen1 = gen1 * config.gc_gen1_multipler
  gen2 = gen2 * config.gc_gen2_multipler
  gc.set_threshold(allocs, gen1, gen2)
  print("GC tweaked (allocs, gen1, gen2): ", allocs, gen1, gen2)

  logger.info("Starting server on port %d with %d threads", port, threads)
  jetstream_server.start()

  if metrics_collector:
    metrics_collector.get_server_startup_latency_metric().set(
        time.time() - server_start_time
    )

  # Setup Jax Profiler
  if enable_jax_profiler:
    logger.info("Starting JAX profiler server on port %s", jax_profiler_port)
    jax.profiler.start_server(jax_profiler_port)
  else:
    logger.info("Not starting JAX profiler server: %s", enable_jax_profiler)

  # Start profiling server by default for proxy backend.
  if jax.config.jax_platforms and "proxy" in jax.config.jax_platforms:
    from jetstream.core.utils import proxy_util  # pylint: disable=import-outside-toplevel

    thread = threading.Thread(
        target=proxy_util.start_profiling_server, args=(jax_profiler_port,)
    )
    thread.run()
  logger.info("Server up and ready to process requests on port %s", port)

  return jetstream_server


def get_devices() -> Any:
  """Gets devices."""
  # TODO: Add more logs for the devices.
  devices = jax.devices()
  logger.info("Using devices: %d", len(devices))
  return devices
