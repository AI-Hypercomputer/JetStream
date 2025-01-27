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

"""Manages the list of fine-tuned adapters loaded on top of the base model for serving.
"""

import logging
import grpc

from typing import Optional

from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core import orchestrator


def calculate_loading_cost(adapter_path: str):
  return 1


class MultiLoraManager(jetstream_pb2_grpc.MultiAdapterManagerServicer):
  """Manages the parameters of multiple lora requests and their lifelines."""

  _driver: orchestrator.Driver

  def __init__(self, driver: orchestrator.Driver):
    self._driver = driver
    self.loaded_adapters = {} # Dictionary to track loaded adapters

  def ListAdapters(
      self,
	  request: jetstream_pb2.ListAdaptersRequest,
	  context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.ListAdaptersResponse:
    """ListAdapters all loaded LoRA adapters."""

    try:
      logging.info("AMANGU LOG: Before making call to mayBeListLoadedAdapters.")
      self._driver.mayBeListLoadedAdapters()
      logging.info("AMANGU LOG: After making call to mayBeListLoadedAdapters.")

      adapter_infos = []
      for adapter_id, adapter_data in self.loaded_adapters.items():
        adapter_info = jetstream_pb2.AdapterInfo(
              adapter_id=adapter_id,
              loading_cost=adapter_data["loading_cost"]
        )
        adapter_infos.append(adapter_info)

    # logging.info("AMANGU Log (adapter_manager.py): ListAdapters is still under implementation")
      logging.info("AMANGU LOG: List adapters --> Before returning success.")
      logging.info(f"AMANGU LOG: List of adapters --> {adapter_infos}.")

      return jetstream_pb2.ListAdaptersResponse(success=True, adapter_infos=adapter_infos)
    except Exception as e:
      logging.info("AMANGU LOG: List adapters --> Before returning failure.")
      return jetstream_pb2.ListAdaptersResponse(success=False, error_message=str(e))


  def LoadAdapter(
      self,
      request: jetstream_pb2.LoadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.LoadAdapterResponse:
    """Load a LoRA adapter as mentioned in the request."""

    try:
      # Load the adapter using MaxEngine in the Driver
      # Implmentation to load adatper using MaxEnbine and request.adapter_path

      # Store adapter info (e.g. loading cost
      self._driver.loadAndApplyAdapter(request.adapter_id,
                                       request.adapter_config_path,
                                       request.adapter_weights_path)

      self.loaded_adapters[request.adapter_id] = {
            "adapter_path": request.adapter_weights_path,
            "loading_cost": calculate_loading_cost(request.adapter_weights_path)
      }

      return jetstream_pb2.LoadAdapterResponse(success=True)
    except Exception as e:
      return jetstream_pb2.LoadAdapterResponse(success=False, error_message=str(e))


  def UnloadAdapter(
      self,
      request: jetstream_pb2.UnloadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.UnloadAdapterResponse:
    """Unload a LoRA adapter as mentioned in the request."""
    
    # logging.info("AMANGU Log (adapter_manager.py): UnloadAdapter is still under implementation")
    try:
      # Unload the adapter
      # Implementation to unload adapter from MaxEngine
      self._driver.unloadAdapter(request.adapter_id)

      del self.loaded_adapters[request.adapter_id]
      return jetstream_pb2.UnloadAdapterResponse(success=True)
    except Exception as e:
      logging.info(f"AMANGU Log(adapter_manager.py): UnloadAdapter failed with error {str(e)}")
      return jetstream_pb2.UnloadAdapterResponse(success=False, error_message=str(e))

    
  
