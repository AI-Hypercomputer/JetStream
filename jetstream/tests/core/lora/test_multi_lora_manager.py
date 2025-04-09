import asyncio
import logging
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock  # Use AsyncMock for async methods

import grpc  # For mocking context if needed, often None suffices
import numpy as np  # For dummy weights if needed by helpers
import jax.numpy as jnp  # For dummy weights if needed by helpers


# Assuming protos are generated and importable
from jetstream.core.proto import multi_lora_decoding_pb2
from jetstream.core.proto import multi_lora_decoding_pb2_grpc

# Assuming the class under test and its dependencies are importable
from jetstream.core.lora import multi_lora_inference_api  # Adjust import path
from jetstream.core import orchestrator
from jetstream.core.lora import adapter_tensorstore  # For status enum and metadata

AdapterStatus = adapter_tensorstore.AdapterStatus
AdapterMetadata = (
    adapter_tensorstore.AdapterMetadata
)  # Assuming this is accessible
MultiLoraManager = multi_lora_inference_api.MultiLoraManager


# --- Mocking Helpers ---
def create_mock_adapter_metadata(adapter_id, status, last_accessed_offset=0):
  """Creates a mock AdapterMetadata object."""
  return AdapterMetadata(
      adapter_id=adapter_id,
      adapter_path=f"/fake/path/{adapter_id}",
      status=status,
      size_hbm=1024 * 1024 * 10,  # 10 MiB
      size_cpu=1024 * 1024 * 12,  # 12 MiB
      last_accessed=time.time() - last_accessed_offset,
      config={"rank": 8},
  )


async def mock_load_adapter_to_tensorstore(adapter_id: str, adapter_path: str):
  print(f"Test Executor: Fake load_adapter_to_tensorstore")
  del adapter_id
  del adapter_path


# --- Test Class ---


class MultiLoraManagerTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    """Set up mocks before each test."""
    self.mock_driver = MagicMock(spec=orchestrator.Driver)
    # Mock the async methods on the driver using AsyncMock
    self.mock_driver.load_adapter_to_tensorstore = AsyncMock(return_value=None)
    self.mock_driver.unload_adapter_from_tensorstore = AsyncMock(
        return_value=None
    )

    # list_adapters_from_tensorstore is synchronous in the example
    self.mock_driver.list_adapters_from_tensorstore = MagicMock()

    # Create the instance of the class under test
    self.manager = MultiLoraManager(driver=self.mock_driver)

  # === Test models (ListAdapters) ===

  def test_models_success_multiple_adapters(self):
    """Test listing adapters successfully with various statuses."""
    mock_registry_data = {
        "adapter1": create_mock_adapter_metadata(
            "adapter1", AdapterStatus.LOADED_HBM, 10
        ),
        "adapter2": create_mock_adapter_metadata(
            "adapter2", AdapterStatus.LOADED_CPU, 5
        ),
        "adapter3": create_mock_adapter_metadata(
            "adapter3", AdapterStatus.UNLOADED, 20
        ),
        "adapter4": create_mock_adapter_metadata(
            "adapter4", AdapterStatus.LOADING, 1
        ),
    }
    self.mock_driver.list_adapters_from_tensorstore.return_value = (
        mock_registry_data
    )

    request = multi_lora_decoding_pb2.ListAdaptersRequest()
    response = self.manager.models(request)  # Call the sync method

    self.mock_driver.list_adapters_from_tensorstore.assert_called_once()
    self.assertTrue(response.success)
    self.assertEqual(response.error_message, "")
    self.assertEqual(len(response.adapter_infos), 4)

    # Check mapping and content (order might vary depending on dict iteration)
    response_map = {info.adapter_id: info for info in response.adapter_infos}
    self.assertIn("adapter1", response_map)
    self.assertIn("adapter2", response_map)
    self.assertIn("adapter3", response_map)
    self.assertIn("adapter4", response_map)

    # Check loading_cost mapping based on status
    self.assertEqual(response_map["adapter1"].loading_cost, 0)  # LOADED_HBM
    self.assertEqual(response_map["adapter2"].loading_cost, 1)  # LOADED_CPU
    self.assertEqual(response_map["adapter3"].loading_cost, 2)  # UNLOADED
    self.assertEqual(
        response_map["adapter4"].loading_cost, -1
    )  # LOADING (or other)

    # Check other fields are copied correctly
    self.assertEqual(
        response_map["adapter1"].size_hbm,
        mock_registry_data["adapter1"].size_hbm,
    )
    self.assertEqual(
        response_map["adapter2"].size_cpu,
        mock_registry_data["adapter2"].size_cpu,
    )
    self.assertEqual(
        response_map["adapter3"].status,
        mock_registry_data["adapter3"].status.value,
    )

  def test_models_success_no_adapters(self):
    """Test listing when no adapters are registered."""
    self.mock_driver.list_adapters_from_tensorstore.return_value = (
        {}
    )  # Empty dict

    request = multi_lora_decoding_pb2.ListAdaptersRequest()
    response = self.manager.models(request)

    self.mock_driver.list_adapters_from_tensorstore.assert_called_once()
    self.assertTrue(response.success)
    self.assertEqual(response.error_message, "")
    self.assertEqual(len(response.adapter_infos), 0)

  def test_models_driver_exception(self):
    """Test error handling when the driver raises an exception."""
    error_message = "Driver failed!"
    self.mock_driver.list_adapters_from_tensorstore.side_effect = Exception(
        error_message
    )

    request = multi_lora_decoding_pb2.ListAdaptersRequest()
    with self.assertLogs(level="INFO") as log:
      response = self.manager.models(request)

    self.mock_driver.list_adapters_from_tensorstore.assert_called_once()
    self.assertFalse(response.success)
    self.assertEqual(response.error_message, error_message)
    self.assertEqual(len(response.adapter_infos), 0)
    self.assertIn("Listing of adapters failed with error:", log.output[0])
    self.assertIn(error_message, log.output[0])

  # === Test load_lora_adapter ===

  def test_load_lora_adapter_success(self):
    """Test successful loading of an adapter."""
    adapter_id = "adapter_to_load"
    adapter_path = "/path/to/load"
    request = multi_lora_decoding_pb2.LoadAdapterRequest(
        adapter_id=adapter_id, adapter_path=adapter_path
    )

    response = self.manager.load_lora_adapter(request)  # Call sync method

    self.mock_driver.load_adapter_to_tensorstore.assert_awaited_once_with(
        adapter_id, adapter_path
    )
    self.assertEqual(response.error_message, "")
    self.assertTrue(response.success)

  def test_load_lora_adapter_driver_exception(self):
    """Test error handling when driver load fails."""
    adapter_id = "adapter_fail_load"
    adapter_path = "/path/to/fail"
    error_message = "Loading failed in driver!"
    request = multi_lora_decoding_pb2.LoadAdapterRequest(
        adapter_id=adapter_id, adapter_path=adapter_path
    )

    # Configure the async mock to raise an exception
    self.mock_driver.load_adapter_to_tensorstore.side_effect = Exception(
        error_message
    )

    with self.assertLogs(level="INFO") as log:
      response = self.manager.load_lora_adapter(request)

    self.mock_driver.load_adapter_to_tensorstore.assert_awaited_once_with(
        adapter_id, adapter_path
    )
    self.assertFalse(response.success)
    self.assertEqual(response.error_message, error_message)
    self.assertIn(
        f"Loading of adapter_id={adapter_id} failed with error:", log.output[0]
    )
    self.assertIn(error_message, log.output[0])

  # === Test unload_lora_adapter ===

  def test_unload_lora_adapter_success(self):
    """Test successful unloading of an adapter."""
    adapter_id = "adapter_to_unload"
    request = multi_lora_decoding_pb2.UnloadAdapterRequest(
        adapter_id=adapter_id
    )

    self.mock_driver.unload_adapter_from_tensorstore.return_value = None

    response = self.manager.unload_lora_adapter(request)

    self.mock_driver.unload_adapter_from_tensorstore.assert_awaited_once_with(
        adapter_id
    )
    self.assertTrue(response.success)
    self.assertEqual(response.error_message, "")

  def test_unload_lora_adapter_driver_exception(self):
    """Test error handling when driver unload fails."""
    adapter_id = "adapter_fail_unload"
    error_message = "Unloading failed in driver!"
    request = multi_lora_decoding_pb2.UnloadAdapterRequest(
        adapter_id=adapter_id
    )

    self.mock_driver.unload_adapter_from_tensorstore.side_effect = Exception(
        error_message
    )

    # Logging is same as load error in the original code, adjust if needed
    with self.assertLogs(level="INFO") as log:
      response = self.manager.unload_lora_adapter(request)

    self.mock_driver.unload_adapter_from_tensorstore.assert_awaited_once_with(
        adapter_id
    )
    self.assertFalse(response.success)
    self.assertEqual(response.error_message, error_message)
    self.assertIn(
        f"Loading of adapter_id={adapter_id} failed with error:", log.output[0]
    )  # Check log message
    self.assertIn(error_message, log.output[0])
