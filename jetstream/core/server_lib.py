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

from concurrent import futures
import logging
from typing import Any, Type

import grpc
import jax

from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core.proto import jetstream_pb2_grpc


_HOST = '[::]'


def run(
    threads: int,
    port: int,
    config: Type[config_lib.ServerConfig],
    devices: Any,
    credentials: Any = grpc.insecure_server_credentials(),
) -> grpc.Server:
  """Runs a server with a specified config."""
  logging.info('Kicking off gRPC server.')
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=threads)
  )  # pytype: disable=wrong-keyword-args
  engines = config_lib.get_engines(config, devices=devices)
  prefill_params = [pe.load_params() for pe in engines.prefill_engines]
  generate_params = [ge.load_params() for ge in engines.generate_engines]
  shared_params = [ie.load_params() for ie in engines.interleaved_engines]
  logging.info('Loaded all weights.')
  driver = orchestrator.Driver(
      prefill_engines=engines.prefill_engines + engines.interleaved_engines,
      generate_engines=engines.generate_engines + engines.interleaved_engines,
      prefill_params=prefill_params + shared_params,
      generate_params=generate_params + shared_params,
  )
  jetstream_pb2_grpc.add_OrchestratorServicer_to_server(
      orchestrator.LLMOrchestrator(driver=driver), server
  )
  logging.info('Starting server on port %d', port)

  server.add_secure_port(f'{_HOST}:{port}', credentials)
  server.start()
  return server


def get_devices() -> Any:
  """Gets devices locally."""
  # Run interleaved engine on local device.
  devices = jax.devices()
  logging.info('Using local devices for interleaved serving: %d', len(devices))
  return devices
