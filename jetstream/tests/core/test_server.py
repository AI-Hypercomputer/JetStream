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

import unittest
from typing import Any, Type

import grpc
import portpicker
import requests
from parameterized import parameterized

from jetstream.core import config_lib
from jetstream.core import server_lib
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import engine_api


class ServerTest(unittest.TestCase):

  @parameterized.expand(
      [
          # Uses weight 2 for prefill, 4 for decode.
          (
              config_lib.CPUTestServer,
              True,
              ["Ċ", "Ō", "Ɵ", ""],
              [266, 332, 415, None],
              [None, None],
          ),
          # Uses the same prefill / generate weights (2).
          (
              config_lib.InterleavedCPUTestServer,
              True,
              ["Ċ", "Ə", "ɖ", ""],
              [266, 399, 598, None],
              [None],
          ),
          # Disable the metrics server.
          (
              config_lib.InterleavedCPUTestServer,
              False,
              ["Ċ", "Ə", "ɖ", ""],
              [266, 399, 598, None],
              [None],
          ),
      ]
  )
  def test_server(
      self,
      config: Type[config_lib.ServerConfig],
      metrics_enabled: bool,
      expected_text: list[str],
      expected_token_ids: list[int | None],
      devices: list[Any],
  ):
    """Sets up a server and requests token responses."""
    ######################### Server side ######################################
    port = portpicker.pick_unused_port()
    metrics_port = portpicker.pick_unused_port()

    print("port: " + str(port))
    credentials = grpc.local_server_credentials()

    server = server_lib.run(
        port=port,
        config=config,
        devices=devices,
        credentials=credentials,
        metrics_server_config=config_lib.MetricsServerConfig(
            port=metrics_port, model_name="some_model_name"
        )
        if metrics_enabled is True
        else None,
    )
    ###################### Requester side ######################################

    # if prometheus not configured, assert no metrics collector on Driver
    if metrics_enabled is not True:
      self.assertIsNone(server._driver._metrics_collector)  # pylint: disable=protected-access

    with grpc.secure_channel(
        f"localhost:{port}", grpc.local_channel_credentials()
    ) as channel:
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)

      healthcheck_request = jetstream_pb2.HealthCheckRequest()
      healthcheck_response = stub.HealthCheck(healthcheck_request)
      self.assertTrue(healthcheck_response.is_live)

      # The string representation of np.array([[65, 66]]), [2] will be prepended
      # as BOS
      text = "AB"
      request = jetstream_pb2.DecodeRequest(
          text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
          max_tokens=3,
      )
      iterator = stub.Decode(request)
      counter = 0
      for resp in iterator:
        output_text = resp.stream_content.samples[0].text
        token_ids = resp.stream_content.samples[0].token_ids
        output_token_id = token_ids[0] if len(token_ids) > 0 else None
        print(f"actual output: {output_text=} {output_token_id=}")
        self.assertEqual(output_text, expected_text[counter])
        self.assertEqual(output_token_id, expected_token_ids[counter])
        counter += 1
      # assert prometheus server is running and responding
      if metrics_enabled is True:
        self.assertIsNotNone(server._driver._metrics_collector)  # pylint: disable=protected-access
        self.assertEqual(
            requests.get(
                f"http://localhost:{metrics_port}", timeout=5
            ).status_code,
            requests.status_codes.codes["ok"],
        )
      server.stop()
      pass

  def test_jax_profiler_server(self):
    port = portpicker.pick_unused_port()
    print("port: " + str(port))
    credentials = grpc.local_server_credentials()
    # Now test server with prometheus config
    server = server_lib.run(
        port=port,
        config=config_lib.InterleavedCPUTestServer,
        devices=[None],
        credentials=credentials,
        enable_jax_profiler=True,
    )
    self.assertIsNotNone(server)
    server.stop()

  def test_get_devices(self):
    self.assertEqual(len(server_lib.get_devices()), 1)

  def test_model_warmup(self):
    port = portpicker.pick_unused_port()

    print("port: " + str(port))
    credentials = grpc.local_server_credentials()

    server = server_lib.run(
        port=port,
        config=config_lib.InterleavedCPUTestServer,
        devices=[None],
        credentials=credentials,
        enable_model_warmup=True,
    )
    # await _wait_for_server_ready(port, timeout_sec=5)
    with grpc.secure_channel(
        f"localhost:{port}", grpc.local_channel_credentials()
    ) as channel:
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)

      healthcheck_request = jetstream_pb2.HealthCheckRequest()
      healthcheck_response = stub.HealthCheck(healthcheck_request)
      self.assertTrue(healthcheck_response.is_live)

      for pe in server._driver._prefill_engines:  # pylint: disable=protected-access
        self.assertIsInstance(pe, engine_api.JetStreamEngine)
        self.assertTrue(pe.warm)

      for ge in server._driver._generate_engines:  # pylint: disable=protected-access
        self.assertIsInstance(ge, engine_api.JetStreamEngine)
        self.assertTrue(ge.warm)

    server.stop()
