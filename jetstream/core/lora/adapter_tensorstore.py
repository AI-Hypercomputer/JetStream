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


def _get_size_of_pytree(params):
  """Get the size of the PyTree."""

  params_bytes = jax.tree_util.tree_map(lambda x: x.nbytes, params)
  total_bytes = jax.tree_util.tree_reduce(lambda x, y: x + y, params_bytes)
  return total_bytes


def _as_np_array(params):
  """Create a new PyTree with Tensors as np.array."""

  def convert_if_jnp(leaf):
    return np.array(leaf)
    
  return jax.tree_util.tree_map(convert_if_jnp, params)


def _as_jnp_array(params):
  """Create a new PyTree with Tensors as jnp.array."""

  def convert_if_np(leaf):
    return jnp.array(leaf)

  return jax.tree_util.tree_map(convert_if_np, params)


@dataclasses.dataclass
class AdapterMetadata:
  adapter_id: str
  adapter_path: str
  status: str = "unloaded"      # "loaded_hbm", "loaded_cpu", "loading", "unloading"
  size_hbm: int = 0             # Size in HBM (bytes)
  size_cpu: int = 0             # Size in CPU RAM (bytes)
  last_accessed: float = 0.0    # timestamp
  config: Dict[str, Any] = dataclasses.field(default_factory=dict)


class AdapterTensorStore:
  def __init__(self, hbm_memory_budget: int, cpu_memory_budget: int, total_slots: int):
    self.hbm_memory_budget = hbm_memory_budget
    self.cpu_memory_budget = cpu_memory_budget
    self.adapter_registry: Dict[str, AdapterMetadata] = {}    # All known adapters
    self.loaded_adapters_hbm: Dict[str, jnp.ndarray] = {}     # adapter_id -> Unified LoRA params (in HBM)
    self.loaded_adapters_cpu: Dict[str, np.ndarray] = {}      # adapter_id -> Unified LoRA params (in CPU RAM)
    self.current_hbm_usage: int = 0
    self.current_cpu_usage: int = 0
    self.running_requests: int = 0    # Number of async tasks which are in "loading" state
    self.decoding_adapters_cache: Dict[str, Any] = {}
    self.total_slots = total_slots
    self.lock = asyncio.Lock()        # Use an asyncio Lock for thread safety


  def register_adapter(self, adapter_id: str, adapter_path: str, config: Dict[str, Any]):
    """Registers a new LoRA adatper."""
    if adapter_id in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' already registered.")
    self.adapter_registry[adapter_id] = AdapterMetadata(
        adapter_id=adapter_id,
        adapter_path=adapter_path,
        config=config)


  async def _transfer_to_hbm(self, adapter_id: str):
    """Transfers an adapter from CPU RAM to HBM."""
    if adapter_id not in self.loaded_adapters_cpu:
      raise ValueError(f"Adapter '{adapter_id}' not loaded in CPU RAM.")

    async with self.lock: #Acquire lock
      metadata = self.adapter_registry[adapter_id]

      if metadata.status == "loaded_hbm":
        return

      # Check if we have enough space in HBM; evict if necessary
      while (self.current_hbm_usage + metadata.size_hbm) > self.hbm_memory_budget:
        if not self._evict(from_hbm=True):
          raise RuntimeError("Not enough HBM to transfer adapter, and eviction failed.")

      # Move from CPU RAM to HBM
      self.loaded_adapters_hbm[adapter_id] = _as_jnp_array(self.loaded_adapters_cpu[adapter_id]) # Convert to JAX array

      # TODO(amangu): We can avoid deleting cpu_loaded adapters if RAM is not a concern
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

      if metadata.status == "loaded_cpu":
        return

      # Check if we have enough space in CPU; evict if necessary.
      while (self.current_cpu_usage + metadata.size_cpu) > self.cpu_memory_budget:
        if not self._evict(from_hbm=False):
          raise RuntimeError("Not enough CPU RAM to transfer adapter, and eviction failed.")

      # Move from HBM to CPU RAM
      self.loaded_adapters_cpu[adapter_id] = _as_np_array(self.loaded_adapters_hbm[adapter_id])
      del self.loaded_adapters_hbm[adapter_id]

      self.current_hbm_usage -= metadata.size_hbm
      self.current_cpu_usage += metadata.size_cpu

      metadata.status = "loaded_cpu"
      metadata.last_accessed = time.time()


  def _initialize_decoding_adapters_cache(self, adapter_weights):
    """
    Create a new PyTree with zero tensors at the paths corresponding to non-None leaves
    in the input PyTree. The zero tensors have an added dimension of size `self.totol_slots`.

    Args:
      adatper_weights: The input PyTree, whose structure will be mirrored.

    Returns:
      A new PyTree with zero Tensors or None values, mirroring the structure of the input PyTree.
    """
    def create_zero_leaf(leaf):
      if leaf is not None:
        original_shape = leaf.shape
        if not original_shape: # handle scalar case
          zero_tensor_shape = (self.total_slots,)
        else:
          zero_tensor_shape = (self.total_slots,) + original_shape  # Prepend a new dimension

        return jnp.zeros(zero_tensor_shape, dtype=leaf.dtype)
      else:
        return None   # Maintain None structure for None leaves

    return jax.tree_util.tree_map(create_zero_leaf, adapter_weights)


  def insert_adapter_in_cache(self, adapter_id: str, slot_id: int):
    """
    Insert the specific adapter tensors into a slot in the serving_adapters_cache.

    Args:
      adapter_id: The id of the adapter, whose tensors will be inserted
      slot_id: The id of slot, which represents the index in the serving_adapter_cache
        where the adapter tensors will be inserted.
    """

    def insert_leaf(dest_leaf, source_leaf):
      if dest_leaf is not None and source_leaf is not None:
        return dest_leaf.at[slot_id].set(source_leaf)   # Insert at the specific index
      elif dest_leaf is not None:
        return dest_leaf      # If source_leaf is None, keep the zero_leaf as is
      elif source_leaf is not None:       # In this case the adapters have different target modules 
        original_shape = source_leaf.shape
        if not original_shape:        # Handle scalar case
          zero_tensor_shape = (self.total_slots,)
        else:
          zero_tensor_shape = (self.total_slots,) + original_shape
        new_dest_leaf = jnp.zeros(zero_tensor_shape, dtype=source_leaf.dtype)
        return new_dest_leaf.at[slot_id].set(source_leaf)
      else:
        return None           # If both are None, return None

    if adapter_id == "":
      logging.info("Empty adapter id. So no LoRA tensors inserted into the cache in adapter_tensorStore.")
      return

    metadata = self.adapter_registry[adapter_id]

    asyncio.run(self.load_adapter(adapter_id, True))

    adapter_weights = self.loaded_adapters_hbm[adapter_id]

    if not self.decoding_adapters_cache:
      self.decoding_adapters_cache = self._initialize_decoding_adapters_cache(adapter_weights)

    self.decoding_adapters_cache = jax.tree_util.tree_map(insert_leaf,
                                                         self.decoding_adapters_cache,
                                                         adapter_weights)


  async def get_hbm_loaded_adapters(self):
    """Returns a comma separated list of adapters loaded into HBM."""

    hbm_loaded_adapters = []

    async with self.lock:
      for adapter_id, metadata in self.adapter_registry.items():
        if metadata.status == "loaded_hbm":
          hbm_loaded_adapters.append(adapter_id)

    return ", ".join(hbm_loaded_adapters)


  async def load_adapter(
      self,
      adapter_id: str,
      adapter_weights = None, 
      to_hbm: bool = True,
      force_load: bool = False):
    """Loads a LoRA adapter's weights, managing HBM and CPU memory."""

    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' not registered.")
    
    metadata = self.adapter_registry[adapter_id]

    async with self.lock:       # Acquire lock for thread safety
      if not force_load and metadata.status in ("loaded_hbm", "loaded_cpu"):
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
        await self.load_adapter(adapter_id, adapter_weights, to_hbm, force_load)
        return

      metadata.status = "loading"
      self.running_requests += 1

    # Load the adapter (asynchronous)
    loop = asyncio.get_running_loop()

    try:
      if adapter_weights is None:
        raise ValueError("Adapter weights for adapter_id={adapter_id} is None.")

      async with self.lock:       # Critical section for memory management
        adapter_weights_as_jnp_array = _as_jnp_array(adapter_weights)
        adapter_weights_as_np_array = _as_np_array(adapter_weights)
        del adapter_weights

        # Get size of unified_lora_params when they are saved in HBM as JAX array
        adapter_size_hbm = _get_size_of_pytree(adapter_weights_as_jnp_array)

        # Get size of unified_lora_params when they are saved in CPU RAM as NumPy array
        adapter_size_cpu = _get_size_of_pytree(adapter_weights_as_np_array)

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
          self.loaded_adapters_hbm[adapter_id] = adapter_weights_as_jnp_array # Convert the PyTree to Jax Array
          self.current_hbm_usage += adapter_size_hbm
          metadata.status = "loaded_hbm"

        else: #to cpu
          self.loaded_adapters_cpu[adapter_id] = adapter_weights_as_np_array # Convert the PyTree to NumPy Array
          self.current_cpu_usage += adapter_size_cpu
          metadata.status = "loaded_cpu"
          
        metadata.last_accessed = time.time()

    except Exception as e:
      async with self.lock:
        metadata.status = "unloaded"    # Mark as unloaded on error
        raise e   # Re-Raise the exception
    finally:
      async with self.lock:
        self.running_requests -= 1


  def get_lora_config(self, adapter_id):
    """Getter for the LoRA adapter config."""
    metadata = self.adapter_registry.get(adapter_id)
    return metadata.config


  def get_lora_weights(self, adapter_id, to_hbm: bool = True):
    """Retrieves the unified LoRA parameters for the given adapter IDs.
       Handles HBM/CPU placement.
    """

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

      metadata.last_accessed = time.time()    # Unload time
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

    for adapter_id, metadata in self.adapter_registry.items():
      if metadata.status == "loaded_hbm" if from_hbm else metadata.status == "loaded_cpu":
        if metadata.last_accessed < lru_time:
          lru_time = metadata.last_accessed
          lru_adapter_id = adapter_id

    # If no adapter found to evict, return False
    if lru_adapter_id is None:
      return False

    if from_hbm:
      # Instead of completely unloading it, kept it in CPU RAM.
      # It can be loaded to HBM if any request demanded it, or
      # it will be evicted from CPU when cpu memory budget reached.
      self._transfer_to_cpu(lru_adapter_id)
    else:
      # Unload the LRU adapter
      self.unload_adapter(lru_adapter_id)   # This is not synchronous, but ONLY within the lock
    return True

