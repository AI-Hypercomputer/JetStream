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

"""Tests gRPC server end-to-end.

See orchestrator test for why these characters specifically will be the
response.
"""

from typing import Any, Type

from absl.testing import absltest, parameterized
import grpc
from jetstream.core import config_lib
from jetstream.core import server_lib
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
import portpicker


class ServerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Uses weight 2 for prefill, 4 for decode.
      (
          config_lib.CPUTestServer,
          ["Ċ", "Ō", "Ɵ", ""],
          [None, None],
      ),
      # Uses the same prefill / generate weights (2).
      (
          config_lib.InterleavedCPUTestServer,
          ["Ċ", "Ə", "ɖ", ""],
          [None],
      ),
  )
  def test_server(
      self,
      config: Type[config_lib.ServerConfig],
      expected_tokens: list[str],
      devices: list[Any],
  ):
    """Sets up a server and requests token responses."""
    ######################### Server side ######################################
    port = portpicker.pick_unused_port()
    print("port: " + str(port))
    credentials = grpc.local_server_credentials()

    server = server_lib.run(
        port=port,
        config=config,
        devices=devices,
        credentials=credentials,
    )
    ###################### Requester side ######################################
    channel = grpc.secure_channel(
        f"localhost:{port}", grpc.local_channel_credentials()
    )
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)

    # The string representation of np.array([[65, 66]]), [2] will be prependd
    # as BOS
    text = "AB"
    request = jetstream_pb2.DecodeRequest(
        session_cache="",
        additional_text=text,
        priority=1,
        max_tokens=3,
    )
    iterator = stub.Decode(request)
    counter = 0
    for token in iterator:
      # Tokens come through as bytes
      print(
          "actual output: "
          + bytes(token.response[0], encoding="utf-8").decode()
      )
      assert (
          bytes(token.response[0], encoding="utf-8").decode()
          == expected_tokens[counter]
      )
      counter += 1
    server.stop()


if __name__ == "__main__":
  absltest.main()
