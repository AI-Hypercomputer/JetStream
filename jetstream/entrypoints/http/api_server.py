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

import json
from typing import Sequence
from absl import app
from absl import flags
from fastapi import APIRouter, Response
import fastapi
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from jetstream.core import config_lib, orchestrator, server_lib
from jetstream.entrypoints.config import get_server_config

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

driver: orchestrator.Driver

# Define Fast API endpoints (use driver to handle).
router = APIRouter()


@router.get("/")
def root():
  """Root path for Jetstream HTTP Server."""
  return Response(
      content=json.dumps({"message": "JetStream HTTP Server"}, indent=4),
      media_type="application/json",
  )


@router.get("/v1/health")
async def health() -> Response:
  """Health check."""
  return Response(
      content=json.dumps({"is_live": str(driver.live)}, indent=4),
      media_type="application/json",
      status_code=200,
  )


def server(argv: Sequence[str]):
  # Init Fast API.
  app = fastapi.FastAPI()
  app.include_router(router)

  # Init driver which would be the main handler in the api endpoints.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  server_config = get_server_config(flags.FLAGS.config, argv)
  print(f"server_config: {server_config}")
  del argv

  metrics_server_config: config_lib.MetricsServerConfig | None = None
  if flags.FLAGS.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(
        port=flags.FLAGS.prometheus_port
    )

  global driver
  driver = server_lib.create_driver(
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
  )

  # Start uvicorn http server.
  uvicorn.run(
      app, host=flags.FLAGS.host, port=flags.FLAGS.port, log_level="info"
  )


if __name__ == "__main__":
  # Run Abseil app w flags parser.
  app.run(server)
