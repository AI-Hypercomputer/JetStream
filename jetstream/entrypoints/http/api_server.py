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

"""JetStream Http API server."""

import json
import logging
from typing import Sequence
from absl import app as abslapp
from absl import flags
from fastapi import APIRouter, Response
import fastapi
from fastapi.responses import StreamingResponse
from prometheus_client import start_http_server
import uvicorn
from google.protobuf.json_format import Parse

from jetstream.core import config_lib, orchestrator, server_lib
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
from jetstream.core.proto import jetstream_pb2
from jetstream.entrypoints.config import get_server_config
from jetstream.entrypoints.http.protocol import DecodeRequest
from jetstream.entrypoints.http.utils import proto_to_json_generator

flags.DEFINE_string("host", "0.0.0.0", "server host address")
flags.DEFINE_integer("port", 8080, "http server port")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)
flags.DEFINE_integer(
    "prometheus_port",
    9988,
    "prometheus_port",
)

llm_orchestrator: orchestrator.LLMOrchestrator

# Define Fast API endpoints (use llm_orchestrator to handle).
router = APIRouter()


@router.get("/")
def root():
  """Root path for Jetstream HTTP Server."""
  return Response(
      content=json.dumps({"message": "JetStream HTTP Server"}, indent=4),
      media_type="application/json",
  )


@router.post("/v1/generate")
async def generate(request: DecodeRequest):
  proto_request = Parse(request.json(), jetstream_pb2.DecodeRequest())
  generator = llm_orchestrator.Decode(proto_request)
  return StreamingResponse(
      content=proto_to_json_generator(generator), media_type="text/event-stream"
  )


@router.get("/v1/health")
async def health() -> Response:
  """Health check."""
  response = await llm_orchestrator.HealthCheck(
      jetstream_pb2.HealthCheckRequest()
  )
  return Response(
      content=json.dumps({"is_live": str(response.is_live)}, indent=4),
      media_type="application/json",
      status_code=200,
  )


def server(argv: Sequence[str]):
  # Init Fast API.
  app = fastapi.FastAPI()
  app.include_router(router)

  # Init LLMOrchestrator which would be the main handler in the api endpoints.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  server_config = get_server_config(flags.FLAGS.config)
  print(f"server_config: {server_config}")
  del argv

  metrics_server_config: config_lib.MetricsServerConfig | None = None
  # Setup Prometheus server
  metrics_collector: JetstreamMetricsCollector = None
  if flags.FLAGS.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(
        port=flags.FLAGS.prometheus_port
    )
    logging.info(
        "Starting Prometheus server on port %d", metrics_server_config.port
    )
    start_http_server(metrics_server_config.port)
    metrics_collector = JetstreamMetricsCollector()
  else:
    logging.info(
        "Not starting Prometheus server: --prometheus_port flag not set"
    )

  global llm_orchestrator
  llm_orchestrator = orchestrator.LLMOrchestrator(
      driver=server_lib.create_driver(
          config=server_config,
          devices=devices,
          metrics_collector=metrics_collector,
      )
  )

  # Start uvicorn http server.
  uvicorn.run(
      app, host=flags.FLAGS.host, port=flags.FLAGS.port, log_level="info"
  )


if __name__ == "__main__":
  # Run Abseil app w flags parser.
  abslapp.run(server)
