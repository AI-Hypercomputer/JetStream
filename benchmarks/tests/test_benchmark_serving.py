# Copyright 2025 Google LLC
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

"""Tests for benchmarks."""

import asyncio
import unittest
from unittest import mock
from benchmarks import benchmark_serving
from jetstream.core.proto import jetstream_pb2


class TestBenchmarkServing(unittest.IsolatedAsyncioTestCase):
  """ "Tests for benchmark_serving.py."""

  async def test_benchmark(self):
    api_url = "test_url"
    tokenizer = mock.MagicMock()
    tokenizer.encode = mock.MagicMock(return_value=[1, 2, 3])
    tokenizer.decode = mock.MagicMock(return_value="test_decode")
    input_requests = [
        benchmark_serving.InputRequest(
            prompt="test_prompt", prompt_len=3, output_len=5, sample_idx=0
        ),
        benchmark_serving.InputRequest(
            prompt="another_prompt", prompt_len=3, output_len=5, sample_idx=0
        ),
    ]
    request_rate = 0.0
    prefill_quota = benchmark_serving.AsyncCounter(1)
    active_req_quota = benchmark_serving.AsyncCounter(10)
    disable_tqdm = True

    async def mocked_decode_response():
      """Mocks decode reponse as an async generator."""
      responses = [
          jetstream_pb2.DecodeResponse(
              stream_content=jetstream_pb2.DecodeResponse.StreamContent(
                  samples=[
                      jetstream_pb2.DecodeResponse.StreamContent.Sample(
                          token_ids=[1]
                      ),
                  ]
              )
          ),
      ]

      for response in responses:
        await asyncio.sleep(10)  # Introduce a 10-second delay
        yield response

    def mock_orchestrator_factory(*args, **kwargs):
      """Mocks generation of an orchestrator stub."""
      del args, kwargs  # Unused.
      mock_stub = mock.MagicMock()
      mock_stub.Decode.return_value = mocked_decode_response()
      return mock_stub

    with mock.patch(
        "grpc.aio.insecure_channel", new_callable=mock.MagicMock
    ) as _, mock.patch(
        "jetstream.core.proto.jetstream_pb2_grpc.OrchestratorStub",
        new_callable=mock.MagicMock,
    ) as mock_stub:
      mock_stub.side_effect = mock_orchestrator_factory

      metrics, outputs = await benchmark_serving.benchmark(
          api_url,
          tokenizer,
          input_requests,
          request_rate,
          disable_tqdm,
          prefill_quota,
          active_req_quota,
      )

    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].generated_text, "test_decode")
    self.assertTrue(outputs[0].success)
    self.assertEqual(metrics["completed"], 2)

  def test_calculate_metrics(self):
    input_requests = [
        benchmark_serving.InputRequest(
            prompt="test_prompt", prompt_len=5, output="test", output_len=4
        )
    ]
    outputs = [
        benchmark_serving.RequestFuncOutput(
            input_request=input_requests[0],
            generated_text="test",
            generated_token_list=[1, 2, 3, 4],
            success=True,
            latency_sec=0.4,
            ttft_sec=0.1,
            ttst_sec=0.2,
            prompt_len=5,
        )
    ]

    tokenizer = mock.MagicMock()
    dur_s = 1.0

    metrics = benchmark_serving.calculate_metrics(
        input_requests, outputs, dur_s, tokenizer
    )

    self.assertIsInstance(metrics, benchmark_serving.BenchmarkMetrics)
    self.assertEqual(metrics.completed, 1)
    self.assertEqual(metrics.total_input, 5)
    self.assertEqual(metrics.total_output, 4)

  def test_str2bool(self):
    self.assertTrue(benchmark_serving.str2bool("true"))
    self.assertTrue(benchmark_serving.str2bool("1"))
    self.assertTrue(benchmark_serving.str2bool("yes"))
    self.assertFalse(benchmark_serving.str2bool("false"))
    self.assertFalse(benchmark_serving.str2bool("0"))
    self.assertFalse(benchmark_serving.str2bool("no"))

    with self.assertRaises(ValueError):
      benchmark_serving.str2bool("test")


if __name__ == "__main__":
  unittest.main()
