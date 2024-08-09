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

"""Runs a JetStream Server."""

from typing import Sequence

from absl import app
from absl import flags

from jetstream.entrypoints import config
from jetstream.core import config_lib, server_lib


flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)
flags.DEFINE_integer("prometheus_port", 0, "")


def main(argv: Sequence[str]):
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  server_config = config.get_server_config(flags.FLAGS.config, argv)
  print(f"server_config: {server_config}")
  del argv

  metrics_server_config: config_lib.MetricsServerConfig | None = None
  if flags.FLAGS.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(
        port=flags.FLAGS.prometheus_port
    )
  # We separate credential from run so that we can unit test it with local
  # credentials.
  # TODO: Add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=flags.FLAGS.threads,
      port=flags.FLAGS.port,
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
  )
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)
