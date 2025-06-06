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

This test tests the multithreaded orchestrator, where a prefill request is
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

import unittest
import jax.numpy as jnp
from parameterized import parameterized
from jetstream.core import orchestrator
from jetstream.core import prefix_cache
from jetstream.core.lora import adapter_tensorstore as adapterstore
from jetstream.core.proto import jetstream_pb2
from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import mock_engine


class OrchestratorTest(unittest.IsolatedAsyncioTestCase):

  def _setup_driver(
      self, interleaved_mode: bool = True, multi_sampling: bool = False
  ):
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
        interleaved_mode=interleaved_mode,
        multi_sampling=multi_sampling,
    )
    return driver

  def _setup_driver_chunked_prefill(
      self, interleaved_mode: bool = True, use_prefix_cache: bool = False
  ):
    prefill_engine = mock_engine.TestEngine(
        batch_size=32,
        cache_length=32,
        weight=2.0,
        use_chunked_prefill=True,
        prefill_chunk_size=4,
    )
    # Create a generate engine with a different set of weights
    # so that we can test that the right one is in use at a given time.
    generate_engine = mock_engine.TestEngine(
        batch_size=4,
        cache_length=32,
        weight=4.0,
        use_chunked_prefill=True,
    )
    prefix_cache_inst = None
    if use_prefix_cache:
      prefix_cache_inst = prefix_cache.PrefixCache(1_000_000, 10_000_000)
    driver = orchestrator.Driver(
        prefill_engines=[prefill_engine],
        generate_engines=[generate_engine],
        prefill_params=[prefill_engine.load_params_dict()],
        generate_params=[generate_engine.load_params()],
        interleaved_mode=interleaved_mode,
        prefix_cache_inst=prefix_cache_inst,
    )
    return driver

  async def _setup_driver_with_adapterstore(
      self, interleaved_mode: bool = True, multi_sampling: bool = False
  ):
    prefill_engine = mock_engine.TestEngine(
        batch_size=32, cache_length=256, weight=2.0
    )
    # Create a generate engine with a different set of weights
    # so that we can test that the right one is in use at a given time.
    generate_engine = mock_engine.TestEngine(
        batch_size=4, cache_length=32, weight=4.0
    )

    prefill_adapterstore = adapterstore.AdapterTensorStore(
        engine=prefill_engine,
        adapters_dir_path="/tmp/",
        hbm_memory_budget=20 * (1024**3),  # 20 GB HBM
        cpu_memory_budget=100 * (1024**3),  # 100 GB RAM
        total_slots=8,
    )

    generate_adapterstore = adapterstore.AdapterTensorStore(
        engine=generate_engine,
        adapters_dir_path="/tmp/",
        hbm_memory_budget=20 * (1024**3),  # 20 GB HBM
        cpu_memory_budget=100 * (1024**3),  # 100 GB RAM
        total_slots=8,
    )

    await prefill_adapterstore.register_adapter(
        adapter_id="test_adapter_1", adapter_config={"r": 4, "alpha": 32}
    )

    adapter_params = jnp.array([3.0], dtype=jnp.float32)
    await prefill_adapterstore.load_adapter(
        adapter_id="test_adapter_1", adapter_weights=adapter_params, to_hbm=True
    )

    await generate_adapterstore.register_adapter(
        adapter_id="test_adapter_1", adapter_config={"r": 4, "alpha": 32}
    )

    await generate_adapterstore.load_adapter(
        adapter_id="test_adapter_1", adapter_weights=adapter_params, to_hbm=True
    )

    driver = orchestrator.Driver(
        prefill_engines=[prefill_engine],
        generate_engines=[generate_engine],
        prefill_params=[prefill_engine.load_params()],
        generate_params=[generate_engine.load_params()],
        prefill_adapterstore=[prefill_adapterstore],
        generate_adapterstore=[generate_adapterstore],
        interleaved_mode=interleaved_mode,
        multi_sampling=multi_sampling,
    )
    return driver

  @parameterized.expand(
      [(True, True), (True, False), (False, True), (False, False)]
  )
  async def test_orchestrator_chunked_prefill(
      self, interleaved_mode: bool, use_prefix_cache: bool
  ):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver_chunked_prefill(
        interleaved_mode, use_prefix_cache
    )
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66, 67, 68]]), [2] will be
    # prepend as BOS. The length > than prefill_chunk_size[4] to use chunked
    # prefill.
    text = "ABCD"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=5,
    )
    iterator = client.Decode(request)
    # first token is (2 + 65 + 66 + 67 + 68) *  2
    expected_token_ids = [536, 670, 837, 1046, 1308, None]
    # token ids is text unicode
    expected_text = [
        chr(token) if token is not None else "" for token in expected_token_ids
    ]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  async def test_orchestrator(self, interleaved_mode: bool):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver(interleaved_mode)
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
    )
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_text = ["Ċ", "Ō", "Ɵ", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([1, 2, 3, 4])
  async def test_orchestrator_multi_sampling(self, num_samples: int):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver(interleaved_mode=True, multi_sampling=True)
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
        num_samples=num_samples,
    )
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_text = ["Ċ", "Ō", "Ɵ", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      for sample in resp.stream_content.samples:
        output_text = sample.text
        token_ids = sample.token_ids
        output_token_id = token_ids[0] if len(token_ids) > 0 else None
        print(f"actual output: {output_text=} {output_token_id=}")
        assert output_text == expected_text[counter]
        assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  async def test_orchestrator_client_tokenization_chunked_prefill(
      self, interleaved_mode: bool
  ):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver_chunked_prefill(interleaved_mode)
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The token ids of  string "ABC", [2] will be prepend
    # as BOS.
    token_ids = [65, 66, 67, 68]

    request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(
            token_ids=token_ids
        ),
        max_tokens=5,
    )
    iterator = client.Decode(request)
    # Return token ids only when in client side tokenization mode.
    expected_text = ["", "", "", "", "", ""]
    # first token is (2 + 65 + 66 + 67 + 68) *  2
    expected_token_ids = [536, 670, 837, 1046, 1308, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  async def test_orchestrator_client_tokenization(self, interleaved_mode: bool):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver(interleaved_mode)
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The token ids of  string "AB", [2] will be prepend
    # as BOS.
    token_ids = [65, 66]

    request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(
            token_ids=token_ids
        ),
        max_tokens=3,
    )
    iterator = client.Decode(request)
    # Return token ids only when in client side tokenization mode.
    expected_text = ["", "", "", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  def test_should_buffer_response(self, interleaved_mode: bool):
    driver = self._setup_driver(interleaved_mode)
    client = orchestrator.LLMOrchestrator(driver=driver)
    self.assertTrue(
        client.should_buffer_response(
            [ReturnSample(text=["<0xAB>"], token_ids=[13])]
        )
    )
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  async def test_orchestrator_with_adapterstore(self, interleaved_mode: bool):
    """Test the multithreaded orchestration with LoRA adapterStore."""
    driver = await self._setup_driver_with_adapterstore(interleaved_mode)
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
        lora_adapter_id="test_adapter_1",
    )
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_text = ["Ċ", "Ō", "Ɵ", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  @parameterized.expand([True, False])
  async def test_load_unload_adapter(self, interleaved_mode: bool):
    """Test loading of adapter to adapter_tensorstore."""
    driver = await self._setup_driver_with_adapterstore(interleaved_mode)

    await driver.load_adapter_to_tensorstore("test_adapter_2", "/tmp/")
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
        lora_adapter_id="test_adapter_2",
    )

    # results = asyncio.run(_consume_decode_iterator(client, request))
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_text = ["Ċ", "Ō", "Ɵ", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1

    adapters = driver.list_adapters_from_tensorstore()

    assert "test_adapter_2" in adapters

    metadata = adapters["test_adapter_2"]
    assert metadata.status in (
        adapterstore.AdapterStatus.LOADED_HBM,
        adapterstore.AdapterStatus.LOADED_CPU,
    )

    await driver.unload_adapter_from_tensorstore("test_adapter_2")

    adapters = driver.list_adapters_from_tensorstore()

    assert "test_adapter_2" in adapters

    metadata = adapters["test_adapter_2"]
    assert metadata.status == adapterstore.AdapterStatus.UNLOADED

    driver.stop()
    print("Orchestrator driver stopped.")

  async def test_drivers_with_none_engine_and_params(self):
    """Test should raise error when driver is init with none engine/driver."""
    prefill_engine = mock_engine.TestEngine(
        batch_size=32, cache_length=256, weight=2.0
    )
    # Create a generate engine with a different set of weights
    # so that we can test that the right one is in use at a given time.
    generate_engine = mock_engine.TestEngine(
        batch_size=4, cache_length=32, weight=4.0
    )

    with self.assertRaisesRegex(ValueError, "No prefill engine provided."):
      driver = orchestrator.Driver(
          generate_engines=[generate_engine],
          prefill_params=[prefill_engine.load_params()],
          generate_params=[generate_engine.load_params()],
      )
      del driver

    with self.assertRaisesRegex(ValueError, "No generate engine provided."):
      driver = orchestrator.Driver(
          prefill_engines=[prefill_engine],
          prefill_params=[prefill_engine.load_params()],
          generate_params=[generate_engine.load_params()],
      )
      del driver

    with self.assertRaisesRegex(ValueError, "No prefill parameter provided."):
      driver = orchestrator.Driver(
          generate_engines=[generate_engine],
          prefill_engines=[prefill_engine],
          generate_params=[generate_engine.load_params()],
      )
      del driver

    with self.assertRaisesRegex(ValueError, "No generate parameter provided."):
      driver = orchestrator.Driver(
          generate_engines=[generate_engine],
          prefill_engines=[prefill_engine],
          prefill_params=[prefill_engine.load_params()],
      )
      del driver

  async def test_adapterstores_exceptions(self, interleaved_mode: bool = True):
    driver = await self._setup_driver_with_adapterstore(interleaved_mode)

    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
        lora_adapter_id="test_adapter_fail",
    )
    iterator = client.Decode(request)

    # chr of [266, 332, 415].
    expected_text = "An error occurred"
    output_text = ""
    async for resp in iterator:
      output_text += resp.stream_content.samples[0].text

    self.assertIn(expected_text, output_text)
    driver.stop()
    print("Orchestrator driver stopped.")
