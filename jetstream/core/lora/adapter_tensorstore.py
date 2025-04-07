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
from jetstream.engine import engine_api
from enum import Enum


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


class AdapterStatus(str, Enum):
  UNLOADED = "unloaded"
  LOADING = "loading"
  LOADED_HBM = "loaded_hbm"
  LOADED_CPU = "loaded_cpu"


@dataclasses.dataclass
class AdapterMetadata:
  adapter_id: str
  adapter_path: str
  status: AdapterStatus = AdapterStatus.UNLOADED
  size_hbm: int = 0  # Size in HBM (bytes)
  size_cpu: int = 0  # Size in CPU RAM (bytes)
  last_accessed: float = 0.0  # timestamp
  config: Dict[str, Any] = dataclasses.field(default_factory=dict)
  loading_event: Optional[asyncio.Event] = None  # Add Event


class AdapterTensorStore:
  """
  Manages the storage and retrieval of LoRA adapter weights, handling
  placement in either HBM (High Bandwidth Memory, on the TPU/GPU) or CPU RAM.

  This class implements an LRU (Least Recently Used) eviction policy
  to manage memory usage.  It supports asynchronous loading and unloading
  of adapters to avoid blocking the main inference thread.

  Args:
    engine: The instance of the JetStream Engine for this AdapterTensorStore
    adapters_dir_path: Location of all the adapters
    hbm_memory_budget (int): The maximum amount of HBM (in bytes) to use for
        storing LoRA adapter weights.
    cpu_memory_budget (int): The maximum amount of CPU RAM (in bytes) to use
        for storing LoRA adapter weights.
  """

  def __init__(
      self,
      engine: engine_api.Engine,
      adapters_dir_path: str,
      hbm_memory_budget: int,
      cpu_memory_budget: int,
  ):
    """Initializes the AdapterTensorStore."""
    self.engine = engine  # Possibly MaxEngine object
    self.adapters_dir_path = adapters_dir_path.rstrip(
        "/"
    )  # All Adapters path without trailing `/`
    self.hbm_memory_budget = hbm_memory_budget
    self.cpu_memory_budget = cpu_memory_budget
    self.adapter_registry: Dict[str, AdapterMetadata] = {}  # All known adapters
    self.loaded_adapters_hbm: Dict[str, jnp.ndarray] = (
        {}
    )  # adapter_id -> LoRA params (in HBM)
    self.loaded_adapters_cpu: Dict[str, np.ndarray] = (
        {}
    )  # adapter_id -> LoRA params (in CPU RAM)
    self.current_hbm_usage: int = 0
    self.current_cpu_usage: int = 0
    self.running_requests: int = (
        0  # Number of async tasks which are in "loading" state
    )
    self.lock = asyncio.Lock()  # Use an asyncio Lock for thread safety

  # --- Unsafe Internal methods which assumes that lock is held ---
  def _unsafe_transfer_to_hbm(self, adapter_id: str):
    """Internal: Transfers an adapter from CPU RAM to HBM. Assumes lock is held."""
    if adapter_id not in self.loaded_adapters_cpu:
      raise ValueError(f"Adapter '{adapter_id}' not loaded in CPU RAM.")

    metadata = self.adapter_registry[adapter_id]

    # Check if we have enough space in HBM; evict if necessary
    while (self.current_hbm_usage + metadata.size_hbm) > self.hbm_memory_budget:
      if not self._evict(from_hbm=True):
        raise RuntimeError(
            "Not enough HBM to transfer adapter, and HBM eviction failed."
        )

    # Move from CPU RAM to HBM
    logging.info(f"Transferring {adapter_id} from CPU to HBM.")
    self.loaded_adapters_hbm[adapter_id] = _as_jnp_array(
        self.loaded_adapters_cpu[adapter_id]
    )  # Convert to JAX array

    # TODO: We can avoid deleting cpu_loaded adapters if RAM is not a concern
    del self.loaded_adapters_cpu[adapter_id]

    self.current_cpu_usage -= metadata.size_cpu
    self.current_hbm_usage += metadata.size_hbm

    metadata.status = AdapterStatus.LOADED_HBM
    metadata.last_accessed = time.time()  # Update time on transfer

  def _unsafe_transfer_to_cpu(self, adapter_id: str):
    """Internal: Transfers an adapter from HBM to CPU RAM. Assumes lock is held."""

    if adapter_id not in self.loaded_adapters_hbm:
      raise ValueError(f"Adapter '{adapter_id}' not loaded in HBM.")

    metadata = self.adapter_registry[adapter_id]

    # Check if we have enough space in CPU; evict if necessary.
    while (self.current_cpu_usage + metadata.size_cpu) > self.cpu_memory_budget:
      if not self._evict(from_hbm=False):
        raise RuntimeError(
            "Not enough CPU RAM to transfer adapter, and CPU eviction failed."
        )

    # Move from HBM to CPU RAM
    logging.info(f"Transferring {adapter_id} from HBM to CPU.")
    self.loaded_adapters_cpu[adapter_id] = _as_np_array(
        self.loaded_adapters_hbm[adapter_id]
    )
    del self.loaded_adapters_hbm[adapter_id]

    self.current_hbm_usage -= metadata.size_hbm
    self.current_cpu_usage += metadata.size_cpu

    metadata.status = AdapterStatus.LOADED_CPU
    metadata.last_accessed = time.time()  # Update time on transfer

  def _unsafe_unload_adapter(self, adapter_id: str):
    """Internal: Unload adapter. Assumes lock is held."""

    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' not found.")

    metadata = self.adapter_registry[adapter_id]
    if metadata.status == AdapterStatus.UNLOADED:
      return

    logging.info(f"Unloading adapter {adapter_id}.")
    if metadata.status == AdapterStatus.LOADED_HBM:
      del self.loaded_adapters_hbm[adapter_id]
      self.current_hbm_usage -= metadata.size_hbm
    elif metadata.status == AdapterStatus.LOADED_CPU:
      del self.loaded_adapters_cpu[adapter_id]
      self.current_cpu_usage -= metadata.size_cpu

    metadata.status = AdapterStatus.UNLOADED
    metadata.last_accessed = time.time()
    metadata.size_hbm = 0
    metadata.size_cpu = 0

  # --- Public Methods (Acquire lock, then call unsafe methods) ---

  async def register_adapter(
      self,
      adapter_id: str,
      adapter_path: str | None = None,
      adapter_config: Dict[str, Any] | None = None,
  ):
    """Registers a new LoRA adatper."""
    """
    Registers a LoRA adapter with the TensorStore.  This also loads the adapter; 
    IF called without adapter_config. Because in this case, it needs
    to get adapter_config from the engine's load_single_adapter() call, which
    also provides the adapter_params. So in that case it is beneficial to load
    the adapter to HBM. This call path is expected only from the direct inference
    request.
    OTHERWISE, it simply adds metadata about the adapter to the registry.

    Args:
      adapter_id (str): A unique identifier for the adapter.
      adapter_path (str): The path to the adapter weights (file or directory).
      adapter_config (dict): Config of the loRA adapter.

    Raises:
      ValueError: If an adapter with the same ID is already registered.
    """
    if adapter_id in self.adapter_registry:
      logging.warning(f"Adapter with ID '{adapter_id}' already registered.")
      return

    if adapter_path is None:
      adapter_path = f"{self.adapters_dir_path}/{adapter_id}"

    adapter_params = None
    if adapter_config is None:
      # This call happens *outside* the lock for potentially slow I/O.
      loop = asyncio.get_running_loop()
      adapter_params, adapter_config = await loop.run_in_executor(
          None, functools.partial(self.engine.load_single_adapter, adapter_path)
      )

    if adapter_config is None:
      raise ValueError(f"Failed to read adapter_config from {adapter_path}")

    async with self.lock:
      # Double check registration inside lock
      if adapter_id in self.adapter_registry:
        logging.warning(f"Adapter '{adapter_id}' registered concurrently.")
        return

      self.adapter_registry[adapter_id] = AdapterMetadata(
          adapter_id=adapter_id,
          adapter_path=adapter_path,
          config=adapter_config,
      )

    # If params were loaded outside lock, now load them into store.
    if adapter_params is not None:
      await self.load_adapter(adapter_id, adapter_params, True)

  async def get_hbm_loaded_adapters(self):
    """Returns a comma separated list of adapters loaded into HBM."""

    hbm_loaded_adapters = []

    async with self.lock:
      for adapter_id, metadata in self.adapter_registry.items():
        if metadata.status == AdapterStatus.LOADED_HBM:
          hbm_loaded_adapters.append(adapter_id)

    return ", ".join(hbm_loaded_adapters)

  async def load_adapter(
      self, adapter_id: str, adapter_weights=None, to_hbm: bool = True
  ):
    """
    Loads a LoRA adapter's weights into memory (either HBM or CPU RAM).

    This method is asynchronous to avoid blocking the main thread during
    potentially slow I/O operations.  It handles:
      - Checking if the adapter is already loaded.
      - Checking if there's enough memory (and evicting if necessary).
      - Loading the weights (in a separate thread).
      - Updating the adapter's status and metadata.

    Args:
      adapter_id (str): The ID of the adapter to load.
      adapter_weights: In the form of a PyTree.
      to_hbm (bool): Whether to load the adapter into HBM (True) or
          CPU RAM (False). Defaults to True (HBM).

    Raises:
      ValueError: If the adapter ID is not registered.
      RuntimeError: If there is not enough memory to load the adapter,
                and eviction fails to free up enough space.
    """
    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adapter with ID '{adapter_id}' not registered.")

    event_to_wait_on: Optional[asyncio.Event] = None

    async with self.lock:
      metadata = self.adapter_registry[adapter_id]

      if metadata.status in (
          AdapterStatus.LOADED_HBM,
          AdapterStatus.LOADED_CPU,
      ):
        metadata.last_accessed = time.time()

        # if already loaded in HBM and we want HBM, or
        # already loaded in CPU and we want CPU, we're done.
        if (to_hbm and metadata.status == AdapterStatus.LOADED_HBM) or (
            not to_hbm and metadata.status == AdapterStatus.LOADED_CPU
        ):
          return  # Already in correct expected state
        elif to_hbm and metadata.status == AdapterStatus.LOADED_CPU:
          # Transfer from cpu to hbm
          self._unsafe_transfer_to_hbm(adapter_id)
          return
        elif not to_hbm and metadata.status == AdapterStatus.LOADED_HBM:
          # Transfer from hbm to cpu
          self._unsafe_transfer_to_cpu(adapter_id)
          return

      # --- Handle LOADING state ---
      if metadata.status == AdapterStatus.LOADING:
        # Wait untill loading is done.
        logging.info(
            f"Adapter {adapter_id} is already loading by another task, waiting..."
        )

        # Get the event created by the first loading task
        event_to_wait_on = metadata.loading_event
        if event_to_wait_on is None:
          # Should not happen if status is LOADING, indicates inconsistency
          raise RuntimeError(
              f"Inconsistent state: Adapter {adapter_id} is LOADING but has no event."
          )

        logging.info(f"Adapter {adapter_id} is loading, will wait.")

      if metadata.status == AdapterStatus.UNLOADED:  # Check if it was UNLOADED
        logging.info(f"Beginning load for adapter {adapter_id}...")

        metadata.loading_event = (
            asyncio.Event()
        )  # Create event *before* releasing lock
        metadata.status = AdapterStatus.LOADING
        self.running_requests += 1

    # ---- Wait if needed (Unlocked) ----
    if event_to_wait_on:
      await event_to_wait_on.wait()
      # After waiting, the original loader finished (or failed).
      # Re-call load_adapter to ensure desired state (HBM/CPU) and update timestamp.
      logging.info(f"Finished waiting for {adapter_id}. Re-checking state.")
      await self.load_adapter(adapter_id, adapter_weights, to_hbm)
      return  # Recursive call handled the final state

    # --- Perform actual loading outside the main lock ---
    load_successful = False
    try:
      if adapter_weights is None:
        adapter_path = metadata.adapter_path  # Use path from metadata

        # TODO: Compare performance improvements
        # Option 1: Low performant (Run blocking I/O on main thread)
        # adapter_weights, adapter_config = self.engine.load_single_adapter(adapter_path)

        # Option 2: Better performant
        # Run blocking I/O in executor
        loop = asyncio.get_running_loop()
        adapter_weights, adapter_config = await loop.run_in_executor(
            None,
            functools.partial(self.engine.load_single_adapter, adapter_path),
        )

      if adapter_weights is None:
        raise ValueError(f"Failed to load adapter_weights from {adapter_path}.")

      # Convert to JAX/NumPy outside main lock if possible (CPU heavy)
      adapter_weights_as_jnp_array = _as_jnp_array(adapter_weights)
      adapter_weights_as_np_array = _as_np_array(adapter_weights)
      del adapter_weights

      # --- Re-acquire lock for final memory check and update ---
      async with self.lock:  # Critical section for memory management
        metadata = self.adapter_registry[adapter_id]  # Re-fetch latest metadata

        # If status changed while loading (e.g., unloaded), abort
        if metadata.status != AdapterStatus.LOADING:
          logging.warning(
              f"Load cancelled for {adapter_id}, status changed to {metadata.status}"
          )
          return

        # Get size of unified_lora_params when they are saved in HBM as JAX array
        adapter_size_hbm = _get_size_of_pytree(adapter_weights_as_jnp_array)

        # Get size of unified_lora_params when they are saved in CPU RAM as NumPy array
        adapter_size_cpu = _get_size_of_pytree(adapter_weights_as_np_array)

        metadata.size_hbm = adapter_size_hbm
        metadata.size_cpu = adapter_size_cpu

        # --- EVICTION (if needed) ---
        # Evict if necessary *before* loading into the target memory
        if to_hbm:
          while (
              self.current_hbm_usage + adapter_size_hbm
          ) > self.hbm_memory_budget:
            if not self._evict(from_hbm=True):
              raise RuntimeError(
                  "Not enough HBM to load adapter, and eviction failed."
              )
        else:  # to_cpu
          while (
              self.current_cpu_usage + adapter_size_cpu
          ) > self.cpu_memory_budget:
            if not self._evict(from_hbm=False):
              raise RuntimeError(
                  "Not enough CPU RAM to load adapter, and eviction failed."
              )

        # Now that we have space (potentially), do the actual loading
        if to_hbm:
          self.loaded_adapters_hbm[adapter_id] = (
              adapter_weights_as_jnp_array  # Convert the PyTree to Jax Array
          )
          self.current_hbm_usage += adapter_size_hbm
          metadata.status = AdapterStatus.LOADED_HBM

        else:  # to cpu
          self.loaded_adapters_cpu[adapter_id] = (
              adapter_weights_as_np_array  # Convert the PyTree to NumPy Array
          )
          self.current_cpu_usage += adapter_size_cpu
          metadata.status = AdapterStatus.LOADED_CPU

        metadata.last_accessed = time.time()
        load_successful = True

    except Exception as e:
      async with self.lock:
        metadata = self.adapter_registry[adapter_id]
        metadata.status = AdapterStatus.UNLOADED  # Mark as unloaded on error
        if metadata.loading_event is not None:
          metadata.loading_event.set()
          metadata.loading_event = None  # Clear the event

        raise e  # Re-Raise the exception
    finally:
      # --- Decrement running_requests, ensure status is correct ---
      async with self.lock:
        metadata = self.adapter_registry[adapter_id]
        self.running_requests -= 1

        if metadata.loading_event is not None:
          metadata.loading_event.set()
          metadata.loading_event = None  # Clear the event

        # If load failed after marking LOADING, reset status
        if (
            not load_successful
            and self.adapter_registry[adapter_id].status
            == AdapterStatus.LOADING
        ):
          metadata.status = AdapterStatus.UNLOADED  # Mark as unloaded on error

  async def get_lora_config(
      self, adapter_id: str, load_if_not_loaded: bool = False
  ):
    """Getter for the LoRA adapter config."""
    metadata = self.adapter_registry.get(adapter_id)

    if load_if_not_loaded and metadata is None:
      await self.register_adapter(adapter_id)
      metadata = self.adapter_registry.get(adapter_id)

    if metadata is None:
      raise ValueError(f"LoRA adapter with id={adapter_id} is not loaded.")

    return metadata.config

  async def get_lora_weights(
      self, adapter_id, to_hbm: bool = True, load_if_not_loaded: bool = False
  ):
    """Retrieves the unified LoRA parameters for the given adapter IDs.
    Handles HBM/CPU placement.
    """

    metadata = self.adapter_registry.get(adapter_id)

    if load_if_not_loaded and metadata is None:
      await self.register_adapter(adapter_id)
      metadata = self.adapter_registry.get(adapter_id)

    if metadata is None:
      raise ValueError(f"LoRA adapter with id={adapter_id} is not loaded.")

    if metadata.status not in (
        AdapterStatus.LOADED_HBM,
        AdapterStatus.LOADED_CPU,
    ):
      await self.load_adapter(adapter_id, None, to_hbm)  # Start loading (async)
    elif to_hbm and metadata.status == AdapterStatus.LOADED_CPU:
      async with self.lock:
        self._unsafe_transfer_to_hbm(adapter_id)
    elif not to_hbm and metadata.status == AdapterStatus.LOADED_HBM:
      async with self.lock:
        self._unsafe_transfer_to_cpu(adapter_id)

    # Now all required adapters should be loaded in correct memory (HBM or CPU), get them
    adapter_params = None
    if to_hbm:
      if adapter_id not in self.loaded_adapters_hbm:
        raise RuntimeError(
            f"Adapter {adapter_id} should be in HBM but wasn't found after loading."
        )
      adapter_params = self.loaded_adapters_hbm[adapter_id]
    else:
      if adapter_id not in self.loaded_adapters_cpu:
        raise RuntimeError(
            f"Adapter {adapter_id} should be in CPU but wasn't found after loading."
        )
      adapter_params = self.loaded_adapters_cpu[adapter_id]

    return adapter_params

  async def unload_adapter(self, adapter_id: str):
    """Unloads a LoRA adapter's weights and removes it from the TensorStore."""
    if adapter_id not in self.adapter_registry:
      raise ValueError(f"Adatper with ID '{adapter_id}' not found.")

    event_to_wait_on: Optional[asyncio.Event] = None
    async with self.lock:
      metadata = self.adapter_registry[adapter_id]
      if metadata.status == AdapterStatus.LOADING:
        event_to_wait_on = metadata.loading_event

    if event_to_wait_on:
      await event_to_wait_on.wait()

    async with self.lock:
      metadata = self.adapter_registry[adapter_id]
      if metadata.status == AdapterStatus.LOADING:
        raise RuntimeError(
            f"Inconsistent state: Adapter {adapter_id} is LOADING after just finishing one."
        )

      self._unsafe_unload_adapter(adapter_id)

  def list_adapters(self) -> Dict[str, AdapterMetadata]:
    """Lists all registered adatpers and their metadata."""
    return self.adapter_registry

  def _evict(self, from_hbm: bool = True) -> bool:
    """Evicts the least recently used adapter from memory (HBM or CPU)."""

    # Find the least recently used adapter that is currently loaded.
    lru_adapter_id = None
    lru_time = float("inf")

    for adapter_id, metadata in self.adapter_registry.items():
      if (
          metadata.status == AdapterStatus.LOADED_HBM
          if from_hbm
          else metadata.status == AdapterStatus.LOADED_CPU
      ):
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
      self._unsafe_transfer_to_cpu(lru_adapter_id)
    else:
      # Unload the LRU adapter
      self._unsafe_unload_adapter(
          lru_adapter_id
      )  # This is not synchronous, but ONLY within the lock
    return True
