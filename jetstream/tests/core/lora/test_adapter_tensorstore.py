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

"""Unit tests for adapter_tensorstore.
"""
import asyncio
import time
import unittest
from unittest.mock import patch, MagicMock
import logging
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized  # Keep for parameterized tests

# Assuming the adapter_tensorstore code is in this path relative to the tests
# NOTE: Adjust the import path based on your project structure
from jetstream.core.lora import adapter_tensorstore
from jetstream.engine import engine_api  # For mocking engine type

# --- Mocking Helpers ---
# Use helpers directly from the module
_get_size_of_pytree = adapter_tensorstore._get_size_of_pytree  # pylint: disable=protected-access
_as_np_array = adapter_tensorstore._as_np_array  # pylint: disable=protected-access
_as_jnp_array = adapter_tensorstore._as_jnp_array  # pylint: disable=protected-access
AdapterStatus = adapter_tensorstore.AdapterStatus
AdapterMetadata = adapter_tensorstore.AdapterMetadata


def create_mock_weights(
    size_multiplier: int = 1, dtype=np.float32
) -> Dict[str, np.ndarray]:
  """Creates a dummy PyTree of NumPy arrays for LoRA weights."""
  rank = 8
  input_dim = 128
  output_dim = 256
  # Simple structure for testing purposes
  return {
      "lora_A": (np.ones((input_dim, rank)) * size_multiplier).astype(dtype),
      "lora_B": (np.ones((rank, output_dim)) * size_multiplier).astype(dtype),
  }


def get_mock_config(rank=8, alpha=16):
  return {"rank": rank, "alpha": alpha}


def get_mock_size(weights):
  weights_as_jnp_array = _as_jnp_array(weights)
  weights_as_np_array = _as_np_array(weights)

  size_hbm = _get_size_of_pytree(weights_as_jnp_array)
  size_cpu = _get_size_of_pytree(weights_as_np_array)

  return size_hbm, size_cpu


# Mock function for engine's load_single_adapter
def load_single_adapter_sync_mock(adapter_path, store_instance):
  """Synchronous mock loading function for run_in_executor."""
  logging.info("SYNC MOCK: Loading from %s", adapter_path)
  time.sleep(0.01)  # Simulate slight delay
  adapter_id = adapter_path.split("/")[-1]
  if adapter_id == "adapter_a":
    return store_instance.mock_weights_a, store_instance.mock_config_a
  elif adapter_id == "adapter_b":
    return store_instance.mock_weights_b, store_instance.mock_config_b
  elif adapter_id == "adapter_c":  # Add another for eviction tests
    return store_instance.mock_weights_c, store_instance.mock_config_c
  elif adapter_id == "adapter_fail":
    raise FileNotFoundError(f"Mock intentionally failed for {adapter_path}")
  elif "no_config" in adapter_path:
    return create_mock_weights(99), None  # Simulate missing config return
  else:
    raise FileNotFoundError(f"Mock sync path not found: {adapter_path}")


# --- Test Class ---


class AdapterTensorStoreTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):  # Use IsolatedAsyncioTestCase

  async def asyncSetUp(self):
    """Set up mocks and the AdapterTensorStore instance before each test."""
    await super().asyncSetUp()

    self.mock_engine = MagicMock(spec=engine_api.Engine)
    self.mock_engine.load_single_adapter = MagicMock()
    self.mock_engine.load_single_adapter.side_effect = (
        lambda path: load_single_adapter_sync_mock(path, self)
    )

    self.adapters_dir_path = "/test/adapters"

    self.mock_weights_a = create_mock_weights(1)
    self.mock_config_a = get_mock_config(rank=8)
    self.mock_size_hbm_a = _get_size_of_pytree(
        _as_jnp_array(self.mock_weights_a)
    )
    self.mock_size_cpu_a = _get_size_of_pytree(self.mock_weights_a)

    self.mock_weights_b = create_mock_weights(2)
    self.mock_config_b = get_mock_config(rank=4)
    self.mock_size_hbm_b = _get_size_of_pytree(
        _as_jnp_array(self.mock_weights_b)
    )
    self.mock_size_cpu_b = _get_size_of_pytree(self.mock_weights_b)

    self.mock_weights_c = create_mock_weights(4)
    self.mock_config_c = get_mock_config(rank=12)
    self.mock_size_hbm_c = _get_size_of_pytree(
        _as_jnp_array(self.mock_weights_c)
    )
    self.mock_size_cpu_c = _get_size_of_pytree(self.mock_weights_c)

    # Default budgets
    self.hbm_budget = self.mock_size_hbm_a + self.mock_size_hbm_b + 100
    self.cpu_budget = self.mock_size_cpu_a + self.mock_size_cpu_b + 100

    # Patch time.time
    self.time_patcher = patch("time.time")
    self.mock_time = self.time_patcher.start()
    self.current_time = 1000.0
    self.mock_time.return_value = self.current_time
    self.addCleanup(self.time_patcher.stop)

    # Create the store instance
    self.store = adapter_tensorstore.AdapterTensorStore(
        engine=self.mock_engine,
        adapters_dir_path=self.adapters_dir_path,
        hbm_memory_budget=self.hbm_budget,
        cpu_memory_budget=self.cpu_budget,
    )

    # Pre-register adapters for most tests to simplify setup
    # Use await now because register_adapter is async
    await self.store.register_adapter(
        "adapter_a", adapter_config=self.mock_config_a
    )
    await self.store.register_adapter(
        "adapter_b", adapter_config=self.mock_config_b
    )
    await self.store.register_adapter(
        "adapter_c", adapter_config=self.mock_config_c
    )
    # Reset mock call count after potential loads during registration
    self.mock_engine.load_single_adapter.reset_mock()

  def advance_time(self, seconds: float = 1.0):
    """Helper to advance the mocked time."""
    self.current_time += seconds
    self.mock_time.return_value = self.current_time

  # === Test Initialization ===
  def test_initialization(self):
    """Test basic attribute initialization."""
    self.assertEqual(self.store.engine, self.mock_engine)
    self.assertEqual(self.store.adapters_dir_path, self.adapters_dir_path)
    self.assertEqual(self.store.hbm_memory_budget, self.hbm_budget)
    self.assertEqual(self.store.cpu_memory_budget, self.cpu_budget)
    # Registry will have pre-registered adapters from setUp
    self.assertIn("adapter_a", self.store.adapter_registry)
    self.assertIn("adapter_b", self.store.adapter_registry)
    self.assertIn("adapter_c", self.store.adapter_registry)
    self.assertEqual(self.store.loaded_adapters_hbm, {})
    self.assertEqual(self.store.loaded_adapters_cpu, {})
    self.assertEqual(self.store.current_hbm_usage, 0)
    self.assertEqual(self.store.current_cpu_usage, 0)
    self.assertEqual(self.store.running_requests, 0)

  # === Test register_adapter ===
  async def test_register_adapter_with_config_only(self):
    """Test registration when only config is provided (no auto-load)."""
    adapter_id = "adapter_new_config"
    adapter_path = "/custom/path/adapter_new"
    adapter_config = {"rank": 16}
    # Clear registry for this test
    self.store.adapter_registry.clear()

    await self.store.register_adapter(adapter_id, adapter_path, adapter_config)

    self.assertIn(adapter_id, self.store.adapter_registry)
    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.adapter_id, adapter_id)
    self.assertEqual(metadata.adapter_path, adapter_path)
    self.assertEqual(metadata.config, adapter_config)
    self.assertEqual(metadata.status, AdapterStatus.UNLOADED)
    self.assertEqual(self.store.current_hbm_usage, 0)
    self.assertEqual(self.store.current_cpu_usage, 0)
    self.mock_engine.load_single_adapter.assert_not_called()

  async def test_register_adapter_without_config_triggers_load(self):
    """Test registration without config triggers engine load and store load."""
    adapter_id = "adapter_new_load"
    adapter_path = f"{self.adapters_dir_path}/{adapter_id}"
    # Configure engine mock for this specific ID
    mock_weights_new = create_mock_weights(9)
    mock_config_new = get_mock_config(9)
    self.mock_engine.load_single_adapter.side_effect = (
        lambda p: (mock_weights_new, mock_config_new)
        if p == adapter_path
        else FileNotFoundError
    )

    # Call register - this will call engine.load_single_adapter synchronously
    # and then internally call await self.load_adapter(...)
    await self.store.register_adapter(adapter_id)  # No path, no config

    # 1. Check registration occurred and config populated
    self.assertIn(adapter_id, self.store.adapter_registry)
    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.adapter_id, adapter_id)
    self.assertEqual(metadata.adapter_path, adapter_path)
    self.assertEqual(
        metadata.config, mock_config_new
    )  # Config populated by load

    # 2. Check engine was called to get weights/config
    self.mock_engine.load_single_adapter.assert_called_once_with(adapter_path)

    # 3. Check the final state (since mocked run was synchronous)
    self.assertEqual(
        metadata.status, AdapterStatus.LOADED_HBM
    )  # Default load is HBM
    self.assertIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertTrue(self.store.current_hbm_usage > 0)

  async def test_register_adapter_load_fails_no_config(self):
    """Test register raises error if engine load fails to provide config."""
    adapter_id = "adapter_no_config"
    adapter_path = f"{self.adapters_dir_path}/{adapter_id}"
    # Mock is configured in setUp to return None for config

    with self.assertRaisesRegex(
        ValueError, f"Failed to read adapter_config from {adapter_path}"
    ):
      await self.store.register_adapter(adapter_id)

    self.mock_engine.load_single_adapter.assert_called_once_with(adapter_path)
    self.assertNotIn(adapter_id, self.store.adapter_registry)

  async def test_register_adapter_duplicate_logs_warning(self):
    """Test registering a duplicate adapter ID logs a warning and is no-op."""
    adapter_id = "adapter_a"  # Already registered in setUp
    initial_metadata = self.store.adapter_registry[adapter_id]
    self.mock_engine.load_single_adapter.reset_mock()

    with self.assertLogs(level="WARNING") as log:
      await self.store.register_adapter(
          adapter_id, adapter_config={"rank": 32}
      )  # Duplicate

    self.assertIn(
        f"Adapter with ID '{adapter_id}' already registered.", log.output[0]
    )
    # Ensure registry wasn't overwritten
    self.assertIs(self.store.adapter_registry[adapter_id], initial_metadata)
    self.assertEqual(
        self.store.adapter_registry[adapter_id].config, self.mock_config_a
    )
    self.mock_engine.load_single_adapter.assert_not_called()

  # === Test load_adapter ===
  # (Includes tests for success, already loaded, transfers, waiting, eviction)

  async def test_load_unregistered_adapter_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "Adapter with ID 'unregistered' not registered."
    ):
      await self.store.load_adapter("unregistered")
    self.assertEqual(self.store.running_requests, 0)

  # Mock the executor call path specifically for load_adapter
  @parameterized.named_parameters(
      ("load_to_hbm", True, AdapterStatus.LOADED_HBM, "_as_jnp_array"),
      ("load_to_cpu", False, AdapterStatus.LOADED_CPU, "_as_np_array"),
  )
  async def test_load_new_adapter_success(
      self, to_hbm, final_status, convert_func_name
  ):
    """Test loading a new adapter successfully to HBM or CPU."""
    adapter_id = "adapter_a"
    # Reset mocks as register_adapter might have been called implicitly
    # if config was None
    self.mock_engine.load_single_adapter.reset_mock()

    # Patch the conversion functions to check they are called
    with patch(
        f"jetstream.core.lora.adapter_tensorstore.{convert_func_name}",
        wraps=getattr(adapter_tensorstore, convert_func_name),
    ) as mock_convert:

      await self.store.load_adapter(adapter_id, to_hbm=to_hbm)

      mock_convert.assert_called_once()  # Verify the correct conversion called

    # Verify engine load was called via executor
    self.mock_engine.load_single_adapter.assert_called_once_with(
        f"{self.adapters_dir_path}/{adapter_id}"
    )

    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, final_status)
    self.assertEqual(metadata.last_accessed, self.current_time)
    self.assertEqual(metadata.size_hbm, self.mock_size_hbm_a)
    self.assertEqual(metadata.size_cpu, self.mock_size_cpu_a)
    self.assertEqual(self.store.running_requests, 0)  # Should be decremented

    if to_hbm:
      self.assertIn(adapter_id, self.store.loaded_adapters_hbm)
      self.assertNotIn(adapter_id, self.store.loaded_adapters_cpu)
      self.assertEqual(self.store.current_hbm_usage, self.mock_size_hbm_a)
      self.assertEqual(self.store.current_cpu_usage, 0)
      jax.tree_util.tree_map(
          lambda x, y: self.assertIsInstance(x, jax.Array),
          self.store.loaded_adapters_hbm[adapter_id],
          self.mock_weights_a,  # Structure reference
      )
    else:  # Loaded to CPU
      self.assertIn(adapter_id, self.store.loaded_adapters_cpu)
      self.assertNotIn(adapter_id, self.store.loaded_adapters_hbm)
      self.assertEqual(self.store.current_cpu_usage, self.mock_size_cpu_a)
      self.assertEqual(self.store.current_hbm_usage, 0)
      jax.tree_util.tree_map(
          lambda x, y: self.assertIsInstance(x, np.ndarray),
          self.store.loaded_adapters_cpu[adapter_id],
          self.mock_weights_a,
      )

  async def test_load_adapter_with_preloaded_weights_success(self):
    """Test loading works when weights are passed directly."""
    adapter_id = "adapter_pre"
    weights_np = create_mock_weights(3)  # Use NumPy as if loaded
    config = get_mock_config()
    size_hbm, size_cpu = get_mock_size(weights_np)

    # Register first
    await self.store.register_adapter(adapter_id, adapter_config=config)

    await self.store.load_adapter(
        adapter_id, adapter_weights=weights_np, to_hbm=True
    )

    # Shouldn't call engine
    self.mock_engine.load_single_adapter.assert_not_called()

    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, AdapterStatus.LOADED_HBM)
    self.assertIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertEqual(self.store.current_hbm_usage, size_hbm)
    self.assertEqual(metadata.size_hbm, size_hbm)
    self.assertEqual(metadata.size_cpu, size_cpu)
    self.assertEqual(self.store.running_requests, 0)

  @parameterized.named_parameters(
      ("hbm_to_hbm", True, AdapterStatus.LOADED_HBM),
      ("cpu_to_cpu", False, AdapterStatus.LOADED_CPU),
  )
  async def test_load_adapter_already_loaded_correct_location(
      self, to_hbm, initial_status
  ):
    """Test loading when adapter is already in the desired location."""
    adapter_id = "adapter_a"
    # Manually set initial state
    if initial_status == AdapterStatus.LOADED_HBM:
      self.store.loaded_adapters_hbm[adapter_id] = _as_jnp_array(
          self.mock_weights_a
      )
      self.store.current_hbm_usage = self.mock_size_hbm_a
    else:
      self.store.loaded_adapters_cpu[adapter_id] = self.mock_weights_a
      self.store.current_cpu_usage = self.mock_size_cpu_a
    self.store.adapter_registry[adapter_id].status = initial_status
    initial_time = self.current_time
    self.advance_time(10)

    # Patch transfers to ensure they are not called
    with patch.object(
        self.store, "_unsafe_transfer_to_hbm"
    ) as mock_t_hbm, patch.object(
        self.store, "_unsafe_transfer_to_cpu"
    ) as mock_t_cpu:
      await self.store.load_adapter(adapter_id, to_hbm=to_hbm)
      mock_t_hbm.assert_not_called()
      mock_t_cpu.assert_not_called()

    self.mock_engine.load_single_adapter.assert_not_called()  # No reload
    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, initial_status)
    self.assertEqual(metadata.last_accessed, self.current_time)  # Time updated
    self.assertNotEqual(metadata.last_accessed, initial_time)
    self.assertEqual(self.store.running_requests, 0)

  @parameterized.named_parameters(
      (
          "cpu_to_hbm",
          False,
          True,
          AdapterStatus.LOADED_HBM,
          "_unsafe_transfer_to_hbm",
      ),
      (
          "hbm_to_cpu",
          True,
          False,
          AdapterStatus.LOADED_CPU,
          "_unsafe_transfer_to_cpu",
      ),
  )
  async def test_load_adapter_triggers_transfer(
      self, initial_hbm, to_hbm, final_status, transfer_method_name
  ):
    """Test loading when adapter needs transferring between HBM and CPU."""
    adapter_id = "adapter_a"
    # Manually set initial state
    if initial_hbm:
      self.store.loaded_adapters_hbm[adapter_id] = _as_jnp_array(
          self.mock_weights_a
      )
      self.store.adapter_registry[adapter_id].status = AdapterStatus.LOADED_HBM
      self.store.current_hbm_usage = self.mock_size_hbm_a
    else:
      self.store.loaded_adapters_cpu[adapter_id] = self.mock_weights_a
      self.store.adapter_registry[adapter_id].status = AdapterStatus.LOADED_CPU
      self.store.current_cpu_usage = self.mock_size_cpu_a
    self.advance_time(10)

    # Patch the specific internal transfer method we expect to be called
    with patch.object(
        self.store,
        transfer_method_name,
        wraps=getattr(self.store, transfer_method_name),
    ) as mock_transfer:
      await self.store.load_adapter(adapter_id, to_hbm=to_hbm)
      mock_transfer.assert_called_once_with(adapter_id)

    # Verify final state
    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, final_status)
    self.assertEqual(metadata.last_accessed, self.current_time)
    self.mock_engine.load_single_adapter.assert_not_called()  # No reload
    self.assertEqual(self.store.running_requests, 0)

  # This decorator temporarily replaces the real 'asyncio.get_running_loop'
  # function with a fake one ('mock_get_loop') for this test only.
  @patch(
      "asyncio.get_running_loop"
  )  # Need to mock the loop for run_in_executor
  async def test_load_adapter_waits_for_loading_state(self, mock_get_loop):
    """Test that a second load call waits if the adapter is already loading."""
    adapter_id = "adapter_a"

    # Create an asyncio Event. This will be used to later to signal a task
    load_finished_event = asyncio.Event()

    # --- Create a Fake 'run_in_executor' ---
    # 'run_in_executor' is the tool asyncio uses to run slow, blocking tasks
    # (like loading from disk) in the background. We need to fake this tool.

    mock_loop = MagicMock()  # Create a fake "manager" for async tasks.

    # Create a placeholder "box" for the result of the background task.
    load_task_future = asyncio.Future()  # Future to control completion

    # This is our FAKE function that will replace the real 'run_in_executor'.
    async def mock_run_in_executor(executor, func):  # pylint: disable=unused-argument
      # func is the actual loading function (self.engine.load_single_adapter)
      print(f"Test Executor: Fake background task started for {adapter_id}")

      # PAUSE HERE: Wait until the signal light (event) turns green.
      await load_finished_event.wait()

      print(f"Test Executor: Finishing fake background task for {adapter_id}")
      result = func()  # Execute the original sync function
      load_task_future.set_result(result)  # Set future result
      return await load_task_future  # Return awaitable future

    # When 'run_in_executor' is called, use the **fake** function.
    mock_loop.run_in_executor.side_effect = mock_run_in_executor
    # Tell the fake 'get_running_loop' (mock_get_loop) to return fake manager
    mock_get_loop.return_value = mock_loop

    # --- Start the Test Scenario ---

    # Start task 1: Try to load 'adapter_a'.
    # This will eventually call our fake 'run_in_executor' and pause at
    # 'await load_finished_event.wait()'
    task1 = asyncio.create_task(
        self.store.load_adapter(adapter_id, to_hbm=True)
    )
    await asyncio.sleep(0.02)  # Give task1 time to enter the executor call

    # Assert that task1 has marked the adapter as LOADING
    async with self.store.lock:
      self.assertEqual(
          self.store.adapter_registry[adapter_id].status, AdapterStatus.LOADING
      )
      self.assertEqual(self.store.running_requests, 1)

    # Start task 2: Try to load the *same* adapter 'adapter_a' again
    # while task 1 is "loading"
    task2 = asyncio.create_task(
        self.store.load_adapter(adapter_id, to_hbm=True)
    )

    # Give task 2 a tiny moment to start. It should see the status is 'LOADING',
    # release the lock, and enter its own waiting loop (calling asyncio.sleep)
    await asyncio.sleep(0.01)

    # Allow task 1's load (in executor) to finish
    load_finished_event.set()

    # Wait for both tasks to complete
    await asyncio.gather(task1, task2)

    ## ASSERTIONS
    # Load from disk only once
    self.mock_engine.load_single_adapter.assert_called_once()

    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, AdapterStatus.LOADED_HBM)
    self.assertIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertEqual(self.store.running_requests, 0)

  @patch(
      "asyncio.get_running_loop"
  )  # Need to mock the loop for run_in_executor
  async def test_load_adapter_inconsistent_loading_state(self, mock_get_loop):
    """Test that a second load call waits if the adapter is already loading."""
    adapter_id = "adapter_a"

    # Create an asyncio Event. This will be used to later to signal a task
    load_finished_event = asyncio.Event()

    # --- Create a Fake 'run_in_executor' ---
    # 'run_in_executor' is the tool asyncio uses to run slow, blocking tasks
    # (like loading from disk) in the background. We need to fake this tool.

    mock_loop = MagicMock()  # Create a fake "manager" for async tasks.

    # Create a placeholder "box" for the result of the background task.
    load_task_future = asyncio.Future()  # Future to control completion

    # This is our FAKE function that will replace the real 'run_in_executor'.
    async def mock_run_in_executor(executor, func):  # pylint: disable=unused-argument
      # func is the actual loading function (self.engine.load_single_adapter)
      print(f"Test Executor: Fake background task started for {adapter_id}")

      # PAUSE HERE: Wait until the signal light (event) turns green.
      await load_finished_event.wait()

      print(f"Test Executor: Finishing fake background task for {adapter_id}")
      result = func()  # Execute the original sync function
      load_task_future.set_result(result)  # Set future result
      return await load_task_future  # Return awaitable future

    # When 'run_in_executor' is called, use the **fake** function.
    mock_loop.run_in_executor.side_effect = mock_run_in_executor
    # Tell the fake 'get_running_loop' (mock_get_loop) to return fake manager
    mock_get_loop.return_value = mock_loop

    with self.assertRaisesRegex(
        RuntimeError,
        f"Inconsistent state: Adapter {adapter_id} is LOADING but has no event",
    ):
      task1 = asyncio.create_task(
          self.store.load_adapter(adapter_id, to_hbm=True)
      )
      await asyncio.sleep(0.02)  # Give task1 time to enter the executor call

      self.store.adapter_registry[adapter_id].loading_event = None

      # Start task 2: Try to load the *same* adapter 'adapter_a' again
      # while task 1 is "loading"
      task2 = asyncio.create_task(
          self.store.load_adapter(adapter_id, to_hbm=True)
      )

      # Allow task 1's load (in executor) to finish
      load_finished_event.set()

      # Wait for both tasks to complete
      await asyncio.gather(task1, task2)

  async def test_load_single_adapter_returning_none(self):
    """Test when load_single_adapter returns adapter_weights=None."""
    adapter_id = "adapter_a"
    adapter_path = f"{self.adapters_dir_path}/{adapter_id}"

    self.mock_engine.load_single_adapter.side_effect = (
        lambda p: (None, None) if p == adapter_path else FileNotFoundError
    )

    with self.assertRaisesRegex(
        ValueError, f"Failed to load adapter_weights from {adapter_path}."
    ):
      await self.store.load_adapter(adapter_id)

  async def test_load_adapter_with_changed_status_before_loading(self):
    """Test corner case of LOADING status change before loading weights."""
    adapter_id = "adapter_a"

    event_load_finished = asyncio.Event()

    # Mock run_in_executor to control load duration
    async def mock_executor(executor, func):  # pylint: disable=unused-argument
      print(f"Test Executor: Started load {adapter_id}")
      await event_load_finished.wait()
      print(f"Test Executor: Finishing load {adapter_id}")
      return func()

    with patch("asyncio.get_running_loop") as mock_get_loop, self.assertLogs(
        level="WARNING"
    ) as cm:
      mock_loop = MagicMock()
      mock_loop.run_in_executor.side_effect = mock_executor
      mock_get_loop.return_value = mock_loop

      # Start loading task
      load_task = asyncio.create_task(self.store.load_adapter(adapter_id))
      await asyncio.sleep(0.01)  # Let load start

      # Update the metadata.status to not-LOADING
      self.store.adapter_registry[adapter_id].status = AdapterStatus.UNLOADED

      # Allow register_adapter to finish
      event_load_finished.set()

      # Wait for both tasks
      await asyncio.gather(load_task)

    self.assertEqual(len(cm.output), 1)  # Expect exactly one warning message
    expected_log = (
        f"Load cancelled for {adapter_id}, "
        f"status changed to AdapterStatus.UNLOADED"
    )
    # Check if the expected message is present in the captured output lines
    self.assertIn(
        expected_log, cm.output[0]
    )  # Check the first (and only) logged line

  # --- Eviction Tests ---

  async def test_load_triggers_hbm_eviction(self):
    """Test loading to HBM triggers HBM LRU eviction (transfer to CPU)."""
    self.store.hbm_memory_budget = (
        self.mock_size_hbm_a + self.mock_size_hbm_b + self.mock_size_hbm_c // 2
    )  # Fits A & B, but not A+B+C

    # Load A (HBM), advance time (A is LRU)
    await self.store.load_adapter("adapter_a", to_hbm=True)
    self.advance_time(10)
    # Load B (HBM), advance time
    await self.store.load_adapter("adapter_b", to_hbm=True)
    self.advance_time(5)

    # Patch the internal methods involved in eviction
    with patch.object(
        self.store, "_evict", wraps=self.store._evict  # pylint: disable=protected-access
    ) as mock_evict, patch.object(
        self.store,
        "_unsafe_transfer_to_cpu",
        wraps=self.store._unsafe_transfer_to_cpu,  # pylint: disable=protected-access
    ) as mock_transfer_cpu:

      await self.store.load_adapter(
          "adapter_c", to_hbm=True
      )  # Load C, should evict B

      # Verify eviction happened
      mock_evict.assert_called_with(from_hbm=True)
      # Check that _unsafe_transfer_to_cpu was called within the evict logic
      mock_transfer_cpu.assert_called_once_with("adapter_a")  # A was LRU

    # Verify final state
    self.assertEqual(
        self.store.adapter_registry["adapter_a"].status,
        AdapterStatus.LOADED_CPU,
    )  # A remains
    self.assertEqual(
        self.store.adapter_registry["adapter_b"].status,
        AdapterStatus.LOADED_HBM,
    )  # B transferred
    self.assertEqual(
        self.store.adapter_registry["adapter_c"].status,
        AdapterStatus.LOADED_HBM,
    )  # C loaded
    self.assertIn("adapter_a", self.store.loaded_adapters_cpu)
    self.assertIn("adapter_b", self.store.loaded_adapters_hbm)
    self.assertIn("adapter_c", self.store.loaded_adapters_hbm)
    self.assertNotIn("adapter_a", self.store.loaded_adapters_hbm)
    self.assertEqual(
        self.store.current_hbm_usage,
        self.mock_size_hbm_b + self.mock_size_hbm_c,
    )
    self.assertEqual(self.store.current_cpu_usage, self.mock_size_cpu_a)

  async def test_load_triggers_cpu_eviction(self):
    """Test loading to CPU triggers CPU LRU eviction (unload)."""
    # self.store.cpu_memory_budget = self.mock_size_cpu_a # Budget fits A
    self.store.cpu_memory_budget = (
        self.mock_size_hbm_a + self.mock_size_hbm_b + self.mock_size_hbm_c // 2
    )  # Fits A & B, but not A+B+C

    # Load A (CPU), advance time (A is LRU)
    await self.store.load_adapter("adapter_a", to_hbm=False)
    self.advance_time(10)
    # Load B (CPU), advance time
    await self.store.load_adapter("adapter_b", to_hbm=False)
    self.advance_time(5)

    with patch.object(
        self.store, "_evict", wraps=self.store._evict  # pylint: disable=protected-access
    ) as mock_evict, patch.object(
        self.store,
        "_unsafe_unload_adapter",
        wraps=self.store._unsafe_unload_adapter,  # pylint: disable=protected-access
    ) as mock_unload:

      await self.store.load_adapter(
          "adapter_c", to_hbm=False
      )  # Load C to CPU, should evict A

      mock_evict.assert_called_with(from_hbm=False)
      # Check that _unsafe_unload_adapter was called within the evict logic
      mock_unload.assert_called_once_with("adapter_a")  # A was LRU

    # Verify final state
    self.assertEqual(
        self.store.adapter_registry["adapter_a"].status, AdapterStatus.UNLOADED
    )  # A unloaded
    self.assertEqual(
        self.store.adapter_registry["adapter_b"].status,
        AdapterStatus.LOADED_CPU,
    )  # B remains
    self.assertEqual(
        self.store.adapter_registry["adapter_c"].status,
        AdapterStatus.LOADED_CPU,
    )  # C loaded
    self.assertNotIn("adapter_a", self.store.loaded_adapters_cpu)
    self.assertIn("adapter_b", self.store.loaded_adapters_cpu)
    self.assertIn("adapter_c", self.store.loaded_adapters_cpu)
    self.assertEqual(
        self.store.current_cpu_usage,
        self.mock_size_cpu_b + self.mock_size_cpu_c,
    )
    self.assertEqual(self.store.current_hbm_usage, 0)

  async def test_load_hbm_eviction_fails_raises_error(self):
    """Test RuntimeError when HBM eviction fails (no suitable adapter)."""
    self.store.hbm_memory_budget = self.mock_size_hbm_a  # Fits A
    await self.store.load_adapter("adapter_a", to_hbm=True)  # Load A

    # Mock _evict to simulate no adapter can be evicted
    with patch.object(self.store, "_evict", return_value=False) as mock_evict:
      with self.assertRaisesRegex(
          RuntimeError, "Not enough HBM to load adapter, and eviction failed."
      ):
        await self.store.load_adapter("adapter_b", to_hbm=True)  # Try load B
      mock_evict.assert_called_once_with(from_hbm=True)

    # Check state reverted
    self.assertEqual(
        self.store.adapter_registry["adapter_b"].status, AdapterStatus.UNLOADED
    )
    self.assertEqual(
        self.store.adapter_registry["adapter_a"].status,
        AdapterStatus.LOADED_HBM,
    )  # A remains
    self.assertEqual(self.store.current_hbm_usage, self.mock_size_hbm_a)
    self.assertEqual(self.store.running_requests, 0)

  async def test_load_cpu_eviction_fails_raises_error(self):
    """Test RuntimeError when CPU eviction fails (no suitable adapter)."""
    self.store.cpu_memory_budget = self.mock_size_cpu_a  # Fits A
    await self.store.load_adapter("adapter_a", to_hbm=False)  # Load A

    # Mock _evict to simulate no adapter can be evicted
    with patch.object(self.store, "_evict", return_value=False) as mock_evict:
      with self.assertRaisesRegex(
          RuntimeError,
          "Not enough CPU RAM to load adapter, and eviction failed.",
      ):
        await self.store.load_adapter("adapter_b", to_hbm=False)  # Try load B
      mock_evict.assert_called_once_with(from_hbm=False)

    # Check state reverted
    self.assertEqual(
        self.store.adapter_registry["adapter_b"].status, AdapterStatus.UNLOADED
    )
    self.assertEqual(
        self.store.adapter_registry["adapter_a"].status,
        AdapterStatus.LOADED_CPU,
    )  # A remains
    self.assertEqual(self.store.current_cpu_usage, self.mock_size_cpu_a)
    self.assertEqual(self.store.running_requests, 0)

  async def test_load_fails_during_io(self):
    """Test status reset if engine load fails during I/O."""
    adapter_id = "adapter_fail"  # Mock engine will raise FileNotFoundError
    await self.store.register_adapter(
        adapter_id, adapter_config=get_mock_config()
    )  # Register first
    self.mock_engine.load_single_adapter.reset_mock()

    # Expect FileNotFoundError from our mock side effect wrapped in executor
    with self.assertRaises(FileNotFoundError):
      await self.store.load_adapter(adapter_id, to_hbm=True)

    # Check state reverted correctly in finally block
    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, AdapterStatus.UNLOADED)
    self.assertEqual(self.store.running_requests, 0)
    self.assertEqual(self.store.current_hbm_usage, 0)
    self.assertEqual(self.store.current_cpu_usage, 0)
    self.assertNotIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertNotIn(adapter_id, self.store.loaded_adapters_cpu)

  # === Test unload_adapter ===
  @parameterized.named_parameters(
      ("unload_hbm", True),
      ("unload_cpu", False),
  )
  async def test_unload_adapter_success(self, loaded_hbm):
    """Test unloading a loaded adapter from HBM or CPU."""
    adapter_id = "adapter_a"
    await self.store.load_adapter(
        adapter_id, to_hbm=loaded_hbm
    )  # Load it first
    self.assertTrue(
        self.store.adapter_registry[adapter_id].status
        in (AdapterStatus.LOADED_HBM, AdapterStatus.LOADED_CPU)
    )
    initial_hbm = self.store.current_hbm_usage
    initial_cpu = self.store.current_cpu_usage

    with patch.object(
        self.store,
        "_unsafe_unload_adapter",
        wraps=self.store._unsafe_unload_adapter,  # pylint: disable=protected-access
    ) as mock_unsafe_unload:
      await self.store.unload_adapter(adapter_id)
      mock_unsafe_unload.assert_called_once_with(adapter_id)

    metadata = self.store.adapter_registry[adapter_id]
    self.assertEqual(metadata.status, AdapterStatus.UNLOADED)
    self.assertEqual(
        self.store.current_hbm_usage, 0 if loaded_hbm else initial_hbm
    )
    self.assertEqual(
        self.store.current_cpu_usage, 0 if not loaded_hbm else initial_cpu
    )
    self.assertNotIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertNotIn(adapter_id, self.store.loaded_adapters_cpu)
    self.assertEqual(metadata.size_hbm, 0)
    self.assertEqual(metadata.size_cpu, 0)

  async def test_unload_unregistered_adapter_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "Adatper with ID 'unknown' not found."
    ):
      await self.store.unload_adapter("unknown")

  async def test_unload_already_unloaded_adapter_is_noop(self):
    adapter_id = "adapter_a"  # Registered but not loaded
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.UNLOADED
    )

    with patch.object(
        self.store,
        "_unsafe_unload_adapter",
        wraps=self.store._unsafe_unload_adapter,  # pylint: disable=protected-access
    ) as mock_unsafe_unload:
      await self.store.unload_adapter(adapter_id)  # Should do nothing
      mock_unsafe_unload.assert_called_once_with(
          adapter_id
      )  # Unsafe method called once

    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.UNLOADED
    )

  async def test_unload_waits_for_loading(self):
    """Test unload waits if adapter is currently loading."""
    adapter_id = "adapter_a"
    event_load_finished = asyncio.Event()

    # Mock run_in_executor to control load duration
    async def mock_executor(executor, func):  # pylint: disable=unused-argument
      print(f"Test Executor: Started load {adapter_id}")
      await event_load_finished.wait()
      print(f"Test Executor: Finishing load {adapter_id}")
      return func()

    with patch("asyncio.get_running_loop") as mock_get_loop:
      mock_loop = MagicMock()
      mock_loop.run_in_executor.side_effect = mock_executor
      mock_get_loop.return_value = mock_loop

      # Start loading task
      load_task = asyncio.create_task(
          self.store.load_adapter(adapter_id, to_hbm=True)
      )
      await asyncio.sleep(0.01)  # Let load start

      self.assertEqual(
          self.store.adapter_registry[adapter_id].status, AdapterStatus.LOADING
      )

      # Start unload task concurrently
      unload_task = asyncio.create_task(self.store.unload_adapter(adapter_id))
      await asyncio.sleep(0.01)  # Let unload start and potentially wait

      # Allow loading to finish
      event_load_finished.set()

      # Wait for both tasks
      await asyncio.gather(load_task, unload_task)

    # Final state should be unloaded
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.UNLOADED
    )
    self.assertNotIn(adapter_id, self.store.loaded_adapters_hbm)
    self.assertEqual(self.store.current_hbm_usage, 0)

  # === Test get_lora_config ===

  async def test_get_lora_config_success(self):
    adapter_id = "adapter_a"
    # Already registered in setUp
    config = await self.store.get_lora_config(adapter_id)
    self.assertEqual(config, self.mock_config_a)

  async def test_get_lora_config_unregistered_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "LoRA adapter with id=unknown is not loaded."
    ):
      await self.store.get_lora_config("unknown")

  async def test_get_lora_config_unregistered_with_load_flag(self):
    """Test get_lora_config triggers registration (and potential load)."""
    adapter_id = "adapter_new_config"
    adapter_path = f"{self.adapters_dir_path}/{adapter_id}"
    mock_weights_new = create_mock_weights(9)
    mock_config_new = get_mock_config(9)
    # Configure engine mock for this specific ID
    self.mock_engine.load_single_adapter.side_effect = (
        lambda p: (mock_weights_new, mock_config_new)
        if p == adapter_path
        else FileNotFoundError
    )

    config = await self.store.get_lora_config(
        adapter_id, load_if_not_loaded=True
    )

    self.assertEqual(config, mock_config_new)
    self.assertIn(adapter_id, self.store.adapter_registry)
    self.mock_engine.load_single_adapter.assert_called_once_with(adapter_path)
    # Check status after the mocked synchronous load
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.LOADED_HBM
    )

  # === Test get_lora_weights ===

  async def test_get_lora_weights_hbm_loaded(self):
    """Test getting weights already in HBM."""
    adapter_id = "adapter_a"
    await self.store.load_adapter(adapter_id, to_hbm=True)
    start_time = self.current_time

    weights = await self.store.get_lora_weights(adapter_id, to_hbm=True)

    # Ensure it's the correct weights and type
    self.assertIsInstance(jax.tree_util.tree_leaves(weights)[0], jax.Array)
    # Basic check, assumes structure matches mock_weights_a
    self.assertTrue(
        jnp.allclose(
            jax.tree_util.tree_leaves(weights)[0],
            jax.tree_util.tree_leaves(_as_jnp_array(self.mock_weights_a))[0],
        )
    )
    # Check access time updated
    self.assertEqual(
        self.store.adapter_registry[adapter_id].last_accessed, start_time
    )

  async def test_get_lora_weights_needs_register_and_load(self):
    """Test getting weights triggers loading."""
    adapter_id = "adapter_a"
    del self.store.adapter_registry[adapter_id]

    weights = await self.store.get_lora_weights(
        adapter_id, to_hbm=True, load_if_not_loaded=True
    )

    # Load should be triggered
    self.mock_engine.load_single_adapter.assert_called_once()

    self.assertIsInstance(jax.tree_util.tree_leaves(weights)[0], jax.Array)
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.LOADED_HBM
    )

  async def test_get_lora_weights_with_unregistered_adapter(self):
    """Test getting weights triggers loading."""
    adapter_id = "adapter_a"
    del self.store.adapter_registry[adapter_id]

    with self.assertRaisesRegex(
        ValueError, f"LoRA adapter with id={adapter_id} is not loaded."
    ):
      weights = await self.store.get_lora_weights(adapter_id, to_hbm=True)
      del weights

  async def test_get_lora_weights_needs_load(self):
    """Test getting weights triggers loading."""
    adapter_id = "adapter_a"
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.UNLOADED
    )

    weights = await self.store.get_lora_weights(adapter_id, to_hbm=True)

    # Load should be triggered
    self.mock_engine.load_single_adapter.assert_called_once()

    self.assertIsInstance(jax.tree_util.tree_leaves(weights)[0], jax.Array)
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.LOADED_HBM
    )

  @parameterized.named_parameters(
      (
          "cpu_to_hbm",
          False,
          True,
          AdapterStatus.LOADED_HBM,
          "_unsafe_transfer_to_hbm",
      ),
      (
          "hbm_to_cpu",
          True,
          False,
          AdapterStatus.LOADED_CPU,
          "_unsafe_transfer_to_cpu",
      ),
  )
  async def test_get_lora_weights_needs_transfer(
      self, initial_hbm, to_hbm, final_status, transfer_method_name
  ):
    """Test getting weights triggers transfer."""
    adapter_id = "adapter_a"
    await self.store.load_adapter(adapter_id, to_hbm=initial_hbm)  # Load to CPU

    with patch.object(
        self.store,
        transfer_method_name,
        wraps=getattr(self.store, transfer_method_name),
    ) as mock_transfer:
      weights = await self.store.get_lora_weights(
          adapter_id, to_hbm=to_hbm
      )  # Request HBM
      # The call to _unsafe_transfer_to_hbm happens *inside* the load_adapter
      # call triggered by get_lora_weights
      # We rely on the mocked asyncio.run to execute it.
      mock_transfer.assert_called_once_with(adapter_id)
      del weights

    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, final_status
    )

  # === Test list_adapters ===
  async def test_list_adapters_multiple_states(self):
    """Test listing adapters with various statuses."""
    # adapter_a, adapter_b, adapter_c are registered in setUp
    await self.store.load_adapter("adapter_b", to_hbm=False)  # Load B to CPU
    await self.store.load_adapter("adapter_c", to_hbm=True)  # Load C to HBM

    adapters = self.store.list_adapters()

    self.assertEqual(len(adapters), 3)  # a, b, c
    self.assertIn("adapter_a", adapters)
    self.assertIn("adapter_b", adapters)
    self.assertIn("adapter_c", adapters)
    self.assertEqual(adapters["adapter_a"].status, AdapterStatus.UNLOADED)
    self.assertEqual(adapters["adapter_b"].status, AdapterStatus.LOADED_CPU)
    self.assertEqual(adapters["adapter_c"].status, AdapterStatus.LOADED_HBM)

  # === Test get_hbm_loaded_adapters ===
  async def test_get_hbm_loaded_adapters_mixed(self):
    """Test getting only HBM loaded adapters."""
    await self.store.load_adapter("adapter_a", to_hbm=True)
    await self.store.load_adapter("adapter_b", to_hbm=False)  # B on CPU
    await self.store.load_adapter("adapter_c", to_hbm=True)

    hbm_list_str = await self.store.get_hbm_loaded_adapters()
    hbm_list = set(s.strip() for s in hbm_list_str.split(",") if s.strip())

    self.assertEqual(hbm_list, {"adapter_a", "adapter_c"})

  async def test_get_hbm_loaded_adapters_none(self):
    """Test getting HBM adapters when none are loaded."""
    await self.store.load_adapter("adapter_a", to_hbm=False)  # Load A to CPU

    hbm_list_str = await self.store.get_hbm_loaded_adapters()

    self.assertEqual(hbm_list_str, "")

  # === Other Tests ===
  async def test_unsafe_transfer_to_hbm_with_valueerror(self):
    """Test raises error in transfer_to_hbm if adapter not in CPU."""
    adapter_id = "adapter_not_in_cpu"

    with self.assertRaisesRegex(
        ValueError, f"Adapter '{adapter_id}' not loaded in CPU RAM."
    ):
      self.store._unsafe_transfer_to_hbm(adapter_id)  # pylint: disable=protected-access

  async def test_unsafe_transfer_to_hbm_with_evict_failure(self):
    """Test eviction failure during transfer_to_hbm."""
    adapter_id = "adapter_a"
    self.store.hbm_memory_budget = self.mock_size_hbm_a // 2  # Not enough for A
    await self.store.load_adapter("adapter_a", to_hbm=False)

    with self.assertRaisesRegex(
        RuntimeError,
        "Not enough HBM to transfer adapter, and HBM eviction failed.",
    ):
      self.store._unsafe_transfer_to_hbm(adapter_id)  # pylint: disable=protected-access

  async def test_unsafe_transfer_to_cpu_with_valueerror(self):
    """Test raises error in transfer_to_cpu if adapter not in hbm."""
    adapter_id = "adapter_not_in_hbm"

    with self.assertRaisesRegex(
        ValueError, f"Adapter '{adapter_id}' not loaded in HBM."
    ):
      self.store._unsafe_transfer_to_cpu(adapter_id)  # pylint: disable=protected-access

  async def test_unsafe_transfer_to_cpu_with_evict_failure(self):
    """Test eviction failure during transfer_to_cpu."""
    adapter_id = "adapter_a"
    self.store.cpu_memory_budget = self.mock_size_cpu_a // 2  # Not enough for A
    await self.store.load_adapter("adapter_a", to_hbm=True)

    with self.assertRaisesRegex(
        RuntimeError,
        "Not enough CPU RAM to transfer adapter, and CPU eviction failed.",
    ):
      self.store._unsafe_transfer_to_cpu(adapter_id)  # pylint: disable=protected-access

  async def test_unsafe_unload_adapter_with_unregistered_adapter(self):
    """Test failure with unregistered adapterd during unload_adapter."""
    adapter_id = "adapter_unregistered"

    with self.assertRaisesRegex(
        ValueError, f"Adapter with ID '{adapter_id}' not found."
    ):
      self.store._unsafe_unload_adapter(adapter_id)  # pylint: disable=protected-access

  async def test_register_adapter_with_concurrent_registrations(self):
    """Test register adapter scenario with concurrent registrations."""
    adapter_id = "adapter_a"

    adapter_metadata = self.store.adapter_registry[adapter_id]
    del self.store.adapter_registry[adapter_id]  # Delete already registered one
    event_load_finished = asyncio.Event()

    # Mock run_in_executor to control load duration
    async def mock_executor(executor, func):  # pylint: disable=unused-argument
      print(f"Test Executor: Started load {adapter_id}")
      await event_load_finished.wait()
      print(f"Test Executor: Finishing load {adapter_id}")
      return func()

    with patch("asyncio.get_running_loop") as mock_get_loop, self.assertLogs(
        level="WARNING"
    ) as cm:
      mock_loop = MagicMock()
      mock_loop.run_in_executor.side_effect = mock_executor
      mock_get_loop.return_value = mock_loop

      # Start register task
      register_task = asyncio.create_task(
          self.store.register_adapter(adapter_id)
      )
      await asyncio.sleep(0.01)  # Let load start

      self.store.adapter_registry[adapter_id] = adapter_metadata

      # Allow register_adapter to finish
      event_load_finished.set()

      # Wait for both tasks
      await asyncio.gather(register_task)

    # Final state should be unloaded
    self.assertEqual(
        self.store.adapter_registry[adapter_id].status, AdapterStatus.UNLOADED
    )
    self.assertEqual(self.store.current_hbm_usage, 0)

    self.assertEqual(len(cm.output), 1)  # Expect exactly one warning message
    expected_log = f"Adapter '{adapter_id}' registered concurrently."
    # Check if the expected message is present in the captured output lines
    self.assertIn(
        expected_log, cm.output[0]
    )  # Check the first (and only) logged line
