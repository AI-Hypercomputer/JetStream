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
import dataclasses

import jax
import jax.numpy as jnp
from flax import struct
import time
import asyncio
import functools
from typing import Dict, Optional, Any
import numpy as np


@dataclasses.dataclass
class AdapterMetadata:
  adapter_id: str
  adapter_path: str
  status: str = "unloaded"      # "loaded_hbm", "loaded_cpu", "loading", "unloading"
  size_hbm: int = 0   # Size in HBM (bytes)
  size_cpu: int = 0   # Size in CPU RAM (bytes)
  last_accessed: float = 0.0    # timestamp
  # rank: int = 8
  config: Dict[str, Any] = None


class AdapterTensorStore:
  def __init__(self, hbm_memory_budget: int, cpu_memory_budget: int):
    self.hbm_memory_budget = hbm_memory_budget
    self.cpu_memory_budget = cpu_memory_budget
    self.adapter_registry: Dict[str, AdapterMetadata] = {}  # All known adapters
    self.loaded_adapters_hbm: Dict[str, jnp.ndarray] = {}     # adapter_id -> Unified LoRA params (in HBM)
    self.loaded_adapters_cpu: Dict[str, np.ndarray] = {}     # adapter_id -> Unified LoRA params (in CPU RAM)
    self.current_hbm_usage: int = 0
    self.current_cpu_usage: int = 0
    self.running_requests: int = 0    # Number of async tasks which are in "loading" state
    self.lock = asyncio.Lock()        # Use an asyncio Lock for thread safety


  def register_adapter(self, adapter_id: str, adapter_path: str, config: Dict[str, Any]):
    """Registers a new LoRA adatper."""
    if adapter_id in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' already registered.")
    self.adapter_registry[adapter_id] = AdapterMetadata(
        adapter_id=adapter_id,
        adapter_path=adapter_path,
        config=config)


  def _get_size(self, arr: jnp.ndarray | np.ndarray) -> int:
    """Calculates the size of a JAX or NumPy array in bytes."""
    # Use asarray to handle both JAX and NumPy arrays consistently
    return np.asarray(arr).nbytes

  async def _transfer_to_hbm(self, adapter_id: str):
    """Transfers an adapter from CPU RAM to HBM."""
    if adapter_id not in self.loaded_adapters_cpu:
      raise ValueError(f"Adapter '{adapter_id}' not loaded in CPU RAM.")

    async with self.lock: #Acquire lock
      metadata = self.adapter_registry[adapter_id]

      # Check if we have enough space in HBM; evict if necessary
      while (self.current_hbm_usage + metadata.size_hbm) > self.hbm_memory_budget:
        if not self._evict(from_hbm=True):
          raise RuntimeError("Not enough HBM to transfer adapter, and eviction failed.")

      # Move from CPU to HBM
      self.loaded_adapters_hbm[adapter_id] = self._as_jnp_array(self.loaded_adapters_cpu[adapter_id]) # Convert to JAX array
      del self.loaded_adapters_cpu[adapter_id]

      self.current_cpu_usage -= metadata.size_cpu
      self.current_hbm_usage += metadata.size_hbm

      metadata.status = "loaded_hbm"
      metadata.last_accessed = time.time()


  async def _transfer_to_cpu(self, adapter_id: str):
    """Transfers an adapter from HBM to CPU RAM."""

    if adapter_id not in self.loaded_adapters_hbm:
      raise ValueError(f"Adapter '{adapter_id}' not loaded in HBM.")

    async with self.lock:
      metadata = self. adapter_registry[adapter_id]

      # Check if we have enough space in CPU; evict if necessary.
      while (self.current_cpu_usage + metadata.size_cpu) > self.cpu_memory_budget:
        if not self._evict(from_hbm=False):
          raise RuntimeError("Not enough CPU RAM to transfer adapter, and eviction failed.")

      # Move from HBM to CPU
      self.loaded_adapters_cpu[adapter_id] = self._as_np_array(self.loaded_adapters_hbm[adapter_id])
      del self.loaded_adapters_hbm[adapter_id]

      self.current_hbm_usage -= metadata.size_hbm
      self.current_cpu_usage += metadata.size_cpu

      metadata.status = "loaded_cpu"
      metadata.last_accessed = time.time()


  def _get_size_of_pytree(self, params):
    params_bytes = jax.tree_util.tree_map(lambda x: x.nbytes, params)
    total_bytes = jax.tree_util.tree_reduce(lambda x, y: x + y, params_bytes)
    return total_bytes


  def _as_np_array(self, params):

    def convert_if_jnp(leaf):
      return np.array(leaf)
    
    return jax.tree_util.tree_map(convert_if_jnp, params)


  def _as_jnp_array(self, params):

    def convert_if_np(leaf):
      return jnp.array(leaf)

    return jax.tree_util.tree_map(convert_if_np, params)


  async def get_hbm_loaded_adapters(self):
    hbm_loaded_adapters = []

    async with self.lock:
      for adapter_id, metadata in self.adapter_registry.items():
        if metadata.status == "loaded_hbm":
          hbm_loaded_adapters.append(adapter_id)

    return ", ".join(hbm_loaded_adapters)


  async def load_adapter(self, adapter_id: str, adapter_weights = None, to_hbm: bool = True):
    """Loads a LoRA adapter's weights, managing HBM and CPU memory."""
    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' not registered.")
    
    metadata = self.adapter_registry[adapter_id]

    async with self.lock:       # Acquire lock for thread safety
      #logging.info(f"AMANGU Logs: Lock aquired by loading section of coroutine {asyncio.current_task().get_name()}.")
      if metadata.status in ("loaded_hbm", "loaded_cpu"):
        metadata.last_accessed = time.time()

        # if already loaded in HBM and we want HBM, or
        # already loaded in CPU and we want CPU, we're done.
        if ((to_hbm and metadata.status == "loaded_hbm") or
            not to_hbm and metadata.status == "loaded_cpu"):
          return
        elif to_hbm and metadata.status == "loaded_cpu":
          # Transfer from cpu to hbm
          self._transfer_to_hbm(adapter_id)
          return
        elif not to_hbm and metadata.status == "loaded_hbm":
          # Transfer from hbm to cpu
          self._transfer_to_cpu(adapter_id)
          return

      if metadata.status == "loading":
        # Wait untill loading is done.
        while metadata.status == "loading":
          await asyncio.sleep(0.1)    # Short sleep to avoid busy-waiting

        # Make recursive call to load_adapter to copy to device
        await self.load_adapter(adapter_id, adapter_weights, to_hbm)
        return

      metadata.status = "loading"
      self.running_requests += 1
      #logging.info(f"AMANGU Logs: Lock released by loading section of coroutine {asyncio.current_task().get_name()}.")

    # Load the adapter (asynchronous)
    loop = asyncio.get_running_loop()
    try:

      # TODO(amangu): Placeholder for the loading logic. Replace with code to load
      # the LoRA weights from the specific path.

      # --- ASYNCHRONOUS LOADING (CRITICAL!) ---
      # Use asyncio.to_thread or similar to avoid blocking

      # TODO(amangu): Assumed that load_lora_weights is defined elsewhere
      # which returns a dictionary: {"lora_A": ..., "lora_B": ...}. Adapt this part
      # based on the actual structure of the loaded LoRA weights.

      if adapter_weights is None:
        adapter_weights = await loop.run_in_executor(
            None,
            functools.partial(load_lora_weights, metadata.adapter_path))

      async with self.lock:       # Critical section for memory management
        # Combine lora_a and lora_b to form a unified parameter.
        # TODO(amangu): Check if combining and storing is having any optimization.
        # unified_lora_params = self._combine_lora_params(lora_weights, metadata.rank)
        #logging.info(f"AMANGU Logs: Lock aquired by saving section of coroutine {asyncio.current_task().get_name()}.")

        unified_lora_params = adapter_weights
        unified_lora_params_as_jnp_array = self._as_jnp_array(unified_lora_params)
        unified_lora_params_as_np_array = self._as_np_array(unified_lora_params)
        del unified_lora_params

        # Get size of unified_lora_params when they are saved in HBM as JAX array
        adapter_size_hbm = self._get_size_of_pytree(unified_lora_params_as_jnp_array)

        # Get size of unified_lora_params when they are saved in CPU RAM as NumPy array
        adapter_size_cpu = self._get_size_of_pytree(unified_lora_params_as_np_array)

        metadata.size_hbm = adapter_size_hbm
        metadata.size_cpu = adapter_size_cpu

        # --- EVICTION (if needed) ---
        # Evict if necessary *before* loading into the target memory
        if to_hbm:
          while (self.current_hbm_usage + adapter_size_hbm) > self.hbm_memory_budget:
            if not self._evict(from_hbm=True):
              raise RuntimeError("Not enough HBM to load adapter, and eviction failed.")
        else: #to_cpu
          while (self.current_cpu_usage + adapter_size_cpu) > self.cpu_memory_budget:
            if not self._evict(from_hbm=False):
              raise RuntimeError("Not enough CPU RAM to load adapter, and eviction failed.")

        # Now that we have space (potentially), do the actual loading
        if to_hbm:
          self.loaded_adapters_hbm[adapter_id] = unified_lora_params_as_jnp_array # Convert the PyTree to Jax Array
          self.current_hbm_usage += adapter_size_hbm
          metadata.status = "loaded_hbm"

        else: #to cpu
          self.loaded_adapters_cpu[adapter_id] = unified_lora_params_as_np_array # Convert the PyTree to NumPy Array
          self.current_cpu_usage += adapter_size_cpu
          metadata.status = "loaded_cpu"
          
        metadata.last_accessed = time.time()
        #logging.info(f"AMANGU Logs: Lock released by saving section of coroutine {asyncio.current_task().get_name()}.")

    except Exception as e:
      async with self.lock:
        metadata.status = "unloaded"    # Mark as unloaded on error
        raise e   # Re-Raise the exception
    finally:
      async with self.lock:
        self.running_requests -= 1


  def _combine_lora_params(self, lora_weights, rank):
    # Create a list to hold the combined LoRA parameters
    combined_lora_params = []
    
    for i in range(0, len(lora_weights), 2):
      lora_a = lora_weights[i]
      lora_b = lora_weights[i+1]

      # Reshape and concatenate lora_a and lora_b
      # Assuming 'br,rnd->bnd' einsum configuration, where 'b' is batch,
      # 'r' is rank, 'n' is num_heads, and 'd' is head_dim
      num_heads = lora_a.shape[1]   # Get number of heads from lora_a
      head_dim = lora_a.shape[2]    # Get head dimension from lora_a

      lora_a = jnp.transpose(lora_a, (1, 2, 0)) # (r, n, d) -> (n, d, r)
      lora_b_reshaped = jnp.reshape(lora_b, (num_heads, head_dim, rank))  # (n * d, r) -> (n, d, r)

      combined_lora_param = jnp.einsum("ndr,ndr->ndr", lora_a, lora_b_reshaped)
      combined_lora_params.append(combined_lora_param)

    # Concatenate the parameters for all layers to form a single unified parameter
    unified_lora_params = jnp.stack(combined_lora_params, axis=0)
    return unified_lora_params


  def get_stacked_lora_weights(self, lora_ids: jnp.ndarray, to_hbm: bool = True):
    """Retrieves the unified LoRA parameters for the given adapter IDs.
       Handles HBM/CPU placement.
    """

    # The logic here is crucial. We have `lora_ids`, an array of shape
    #(batch_size,), where each element is the ID of the LoRA adapter
    # to use for that request in the batch. You need to use this to
    # select the appropriate slices from the *unified* LoRA paramters.

    # 1. Get the unified LoRA paramters for the requested IDs. This
    #    might involve waiting if some adapters are still loading.

    required_adapters = set(lora_ids.tolist())    # Get unique adapter IDs
    for adapter_id in required_adapters:
      metadata = self.adapter_registry.get(adapter_id)

      if metadata is None:
        raise ValueError(f"Adapter with ID '{adapter_id}' not registered.")

      if metadata.status != "loaded_hbm" and metadata.status != "loaded_cpu":
        asyncio.run(self.load_adapter(adapter_id, to_hbm))    # Start loading (async)
      elif to_hbm and metadata.status == "loaded_cpu":
        asyncio.run(self._transfer_to_hbm(adapter_id))
      elif not to_hbm and metadata.status == "loaded_hbm":
        asyncio.run(self._transfer_to_cpu(adapter_id))

    # Wait till all the running requests are completed
    while self.running_requests > 0:
      time.sleep(0.1)

    # Now all required adapters should be loaded in correct memory (HBM or CPU), get them
    if to_hbm:
      required_adapters_params = [self.loaded_adapters_hbm[adapter_id] for adapter_id in required_adapters]
    else:
      required_adapters_params = [self.loaded_adapters_cpu[adapter_id] for adapter_id in required_adapters]
    
    # Stack the parameters for the required adapters
    stacked_params = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *required_adapters_params)

    # Extract paramters using jnp.take() function for the lora_ids.
    retrieved_lora_params = jax.tree_util.tree_map(
        lambda arr: jnp.take(arr, lora_ids, axis=0, fill_value=0),
        stacked_params)

    return retrieved_lora_params


  def get_lora_config(self, adapter_id):
    metadata = self.adapter_registry.get(adapter_id)
    return metadata.config


  def get_lora_weights(self, adapter_id, to_hbm: bool = True):
    """Retrieves the unified LoRA parameters for the given adapter IDs.
       Handles HBM/CPU placement.
    """

    # The logic here is crucial. We have `lora_ids`, an array of shape
    #(batch_size,), where each element is the ID of the LoRA adapter
    # to use for that request in the batch. You need to use this to
    # select the appropriate slices from the *unified* LoRA paramters.

    # 1. Get the unified LoRA paramters for the requested IDs. This
    #    might involve waiting if some adapters are still loading.

    metadata = self.adapter_registry.get(adapter_id)

    if metadata is None:
      raise ValueError(f"Adapter with ID '{adapter_id}' not registered.")

    if metadata.status != "loaded_hbm" and metadata.status != "loaded_cpu":
      asyncio.run(self.load_adapter(adapter_id, None, to_hbm))    # Start loading (async)
    elif to_hbm and metadata.status == "loaded_cpu":
      asyncio.run(self._transfer_to_hbm(adapter_id))
    elif not to_hbm and metadata.status == "loaded_hbm":
      asyncio.run(self._transfer_to_cpu(adapter_id))

    # Wait till all the running requests are completed
    while self.running_requests > 0:
      time.sleep(0.1)

    # Now all required adapters should be loaded in correct memory (HBM or CPU), get them
    adapter_params = None
    if to_hbm:
      adapter_params = self.loaded_adapters_hbm[adapter_id]
    else:
      adapter_params = self.loaded_adapters_cpu[adapter_id]
    
    return adapter_params


  async def unload_adapter(self, adapter_id: str):
    """Unloads a LoRA adapter's weights and removes it from the TensorStore."""
    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adatper with ID '{adapter_id}' not found.")

    metadata = self.adapter_registry[adapter_id]

    async with self.lock:
      if metadata.status == "unloaded":
        return    # Already unloaded
      if metadata.status == "loading":
        # Wait for the loading to get complete.
        while metadata.status == "loading":
          await asyncio.sleep(0.1)
      if metadata.status == "loaded_hbm":
        del self.loaded_adapters_hbm[adapter_id]
        self.current_hbm_usage -= metadata.size_hbm
        metadata.status = "unloaded"
      elif metadata.status == "loaded_cpu":
        del self.loaded_adapters_cpu[adapter_id]
        self.current_cpu_usage -= metadata.size_cpu
        metadata.status = "unloaded"

      metadata.last_accessed = 0.0    # Reset last accessed time
      metadata.size_hbm = 0
      metadata.size_cpu = 0


  def list_adapters(self) -> Dict[str, AdapterMetadata]:
    """Lists all registered adatpers and their metadata."""
    return self.adapter_registry


  def _evict(self, from_hbm: bool = True) -> bool:
    """Evicts the least recently used adapter from memory (HBM or CPU)."""

    # Find the least recently used adapter that is currently loaded.
    lru_adapter_id = None
    lru_time = float('inf')

    if from_hbm:
      adapters_dict = self.loaded_adapters_hbm
    else:
      adapters_dict = self.loaded_adapters_cpu

    for adapter_id, metadata in self.adapter_registry.items():
      if metadata.status == "loaded_hbm" if from_hbm else metadata.status == "loaded_cpu":
        if metadata.last_accessed < lru_time:
          lru_time = metadata.last_accessed
          lru_adapter_id = adapter_id

    # If no adapter found to evict, return False
    if lru_adapter_id is None:
      return False

    # Unload the LRU adapter
    self.unload_adapter(lru_adapter_id)   # This is not synchronous, but ONLY within the lock
    return True

