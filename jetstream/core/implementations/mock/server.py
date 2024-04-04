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

"""Runs a mock server."""

from typing import Sequence

from absl import app
from absl import flags

from jetstream.core.implementations.mock import config as mock_config
from jetstream.core import server_lib


_PORT = flags.DEFINE_integer("port", 9000, "port to listen on")
_CONFIG = flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)


def main(argv: Sequence[str]):
  del argv
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = mock_config.get_server_config(_CONFIG.value)
  # We separate credential from run so that we can unit test it with local
  # credentials.
  # TODO: Add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      port=_PORT.value,
      config=server_config,
      devices=devices,
  )
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)
