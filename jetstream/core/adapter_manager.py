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
from jetstream.core import adapter_tensorstore
from jetstream.core import orchestrator
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core.proto import jetstream_pb2


def calculate_loading_cost(adapter_path: str):
  return 1


class MultiLoraManager(jetstream_pb2_grpc.MultiAdapterManagerServicer):
  """Manages the parameters of multiple lora requests and their lifelines."""

  _driver: orchestrator.Driver

  def __init__(self, driver: orchestrator.Driver):
    self._driver = driver

  def ListAdapters(
      self,
	  request: jetstream_pb2.ListAdaptersRequest,
	  context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.ListAdaptersResponse:
    """ListAdapters all loaded LoRA adapters."""

    try:
      adapters = self._driver.listAdaptersFromTensorstore()

      adapter_infos = []
      for adapter_id, adapter_data in adapters.items():
        if adapter_data.status == "loaded_hbm":
          loading_cost = 0
        elif adapter_data.status == "loaded_cpu":
          loading_cost = 1
        elif adapter_data.status == "unloaded":
          loading_cost = 2
        else:
          loading_cost = -1

        adapter_info = jetstream_pb2.AdapterInfo(
            adapter_id=adapter_id,
            loading_cost=loading_cost,
            size_hbm=adapter_data.size_hbm,
            size_cpu=adapter_data.size_cpu,
            last_accessed=adapter_data.last_accessed,
            status=adapter_data.status)

        adapter_infos.append(adapter_info)

      return jetstream_pb2.ListAdaptersResponse(success=True, adapter_infos=adapter_infos)
    except Exception as e:
      logging.info(f"Listing of adapters failed with error: {str(e)}")
      return jetstream_pb2.ListAdaptersResponse(success=False, error_message=str(e))


  def LoadAdapter(
      self,
      request: jetstream_pb2.LoadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.LoadAdapterResponse:
    """Load a LoRA adapter as mentioned in the request."""

    try:
      self._driver.loadAdapterToTensorstore(request.adapter_id, request.adapter_path)

      return jetstream_pb2.LoadAdapterResponse(success=True)
    except Exception as e:
      logging.info(f"Loading of adapter_id={request.adapter_id} failed with error: {str(e)}")
      return jetstream_pb2.LoadAdapterResponse(success=False, error_message=str(e))


  def UnloadAdapter(
      self,
      request: jetstream_pb2.UnloadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.UnloadAdapterResponse:
    """Unload a LoRA adapter as mentioned in the request."""
    
    try:
      self._driver.unloadAdapterFromTensorstore(request.adapter_id)
      return jetstream_pb2.UnloadAdapterResponse(success=True)
    except Exception as e:
      logging.info(f"Loading of adapter_id={request.adapter_id} failed with error: {str(e)}")
      return jetstream_pb2.UnloadAdapterResponse(success=False, error_message=str(e))

    
  
