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

"""Integration test of the orchestrator.

This test tests the multi-htreaded orchestrator, where a prefill request is
popped onto a prefill queue, prefilled, sent to a generation queue and run for
a number of decoding steps.

In operation, it will use gRPC so we can 'yield' from the function to get return
values in the same way that they would be streamed back.

Similar to 'mock_engine_test' we can use known token values and a singleton
weight to test our operation.

Let the prefill engine have a weight of [2] and the generate engine have a
weight of [3].

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] / 4  = 332
266 + [266, 332] / 4 = 415
I.e. ['Ċ', 'Ō', 'Ɵ'] when converted back with chr().

Therefore we should get back the character sequence '$lǔ' if we request 3 tokens
decoded (these are the ascii chars at those indices which is what the test
tokenizer returns).
"""

from absl.testing import absltest
from jetstream.core import orchestrator
from jetstream.core.proto import jetstream_pb2
from jetstream.engine import mock_engine


class OrchestratorTest(absltest.TestCase):

  def _setup_driver(self):
    prefill_engine = mock_engine.TestEngine(
        batch_size=32, cache_length=256, weight=2.0
    )
    # Create a generate engine with a different set of weights
    # so that we can test that the right one is in use at a given time.
    generate_engine = mock_engine.TestEngine(
        batch_size=4, cache_length=32, weight=4.0
    )
    driver = orchestrator.Driver(
        prefill_engines=[prefill_engine],
        generate_engines=[generate_engine],
        prefill_params=[prefill_engine.load_params()],
        generate_params=[generate_engine.load_params()],
    )
    return driver

  async def test_orchestrator(self):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver()
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        session_cache="",
        additional_text=text,
        priority=1,
        max_tokens=3,
    )
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_tokens = ["Ċ", "Ō", "Ɵ", ""]
    counter = 0
    async for token in iterator:
      # Tokens come through as bytes.
      print(
          "actual output: "
          + bytes(token.response[0], encoding="utf-8").decode()
      )
      assert (
          bytes(token.response[0], encoding="utf-8").decode()
          == expected_tokens[counter]
      )
      counter += 1
    driver.stop()


if __name__ == "__main__":
  absltest.main()
