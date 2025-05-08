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

"""Hierarchical prefix cache implementation for accelerating LLM prefill.

This module provides a `PrefixCache` class that implements a two-level cache
(HBM and DRAM) for storing Key-Value (KV) caches generated during the prefill
stage of language model inference.

Key Features:
- **Hierarchical Storage:** Utilizes both HBM (High Bandwidth Memory) for fast
  access and host DRAM for larger capacity.
- **LRU Eviction:** Employs a Least Recently Used (LRU) strategy to manage cache
  entries in both HBM and DRAM layers.
- **Trie-based Key Lookup:** Uses a `PrefixCacheTrie` for efficient retrieval of
  the longest matching prefix key based on input tokens.
- **Thread Safety:** The `PrefixCache` class uses a lock to ensure thread-safe
  operations.
- **Performance Statistics:** The `PrefixCache` can track and provide statistics
  on cache hit rates and hit length proportions.

Core Components:
- `Value`: Dataclass holding the KVCache (`prefix`), token information, and
  metadata like size and device placement.
- `PrefixCacheTrie`: A Trie data structure optimized for finding the longest
  common prefix among token sequences (keys).
- `ValueStorageInterface`: Abstract base class defining the storage layer API.
- `HBMStorage`, `DRAMStorage`: Concrete implementations of
  `ValueStorageInterface`
  for HBM and DRAM respectively. They handle device placement (`jax.device_put`,
  `jax.device_get`).
- `LRUStrategy`: Manages the order of keys based on usage for LRU eviction.
- `HierarchicalCache`: Manages the two storage layers (`HBMStorage`,
  `DRAMStorage`) and their respective LRU strategies. Handles adding,
  retrieving, and evicting entries across the hierarchy.
- `PrefixCache`: The main user-facing class that integrates the Trie lookup
  (`_trie`) and the hierarchical storage (`_cache`).

Basic Usage:

1.  **Initialization:**
    ```python
    # Define HBM and DRAM capacity in bytes
    hbm_bytes = 64 * 1024**3  # 64 GiB
    dram_bytes = 512 * 1024**3 # 512 GiB

    prefix_cache = PrefixCache(hbm_bytes=hbm_bytes, dram_bytes=dram_bytes)
    ```

2.  **Saving a Prefix:**
    After computing the KVCache (`full_kv_cache`) for a prompt
    (`full_key_tokens`):
    ```python
    from jetstream.core.prefix_cache import Value

    # padded_kv_cache_length: Length used for KVCache generation
    value_to_save = Value(
        prefix=full_kv_cache,
        true_length=len(full_key_tokens), # Actual length without padding
        padded_length=padded_kv_cache_length
        tokens=full_key_tokens,
    )
    # Only save if the key (or a longer version) isn't already present
    if not prefix_cache.contains(full_key_tokens):
        prefix_cache.save(full_key_tokens, value_to_save)
    ```

3.  **Loading a Prefix (before prefill):**
    Given new input tokens (`input_tokens`):
    ```python
    # Find the longest matching prefix in the cache
    # Optionally set a minimum length for the match (e.g., chunk_size)
    cached_value = prefix_cache.load(input_tokens,
        min_common_prefix_key_length=chunk_size)

    if cached_value:
        # Cache hit!
        existing_prefix_cache = cached_value.prefix
        # Figure out how many tokens matched
        common_len = cal_common_prefix_length(input_tokens, cached_value.tokens)
        # Prefill only the remaining tokens
        remaining_tokens = input_tokens[common_len:]
        # ... proceed with prefill using existing_prefix_cache and 
        # remaining_tokens ...
    else:
        # Cache miss - prefill the entire input_tokens
        # ... proceed with full prefill ...
    ```

Helper Functions:
- `load_existing_prefix_and_get_remain_tokens`: A convenience function that
  attempts to load the longest matching prefix from the cache for a given token
  sequence. If a suitable prefix is found (meeting minimum length and
  `chunk_size` alignment criteria), it returns an `engine_api.ExistingPrefix`
  object containing the cached KVCache and the common tokens (length adjusted
  to be a multiple of `chunk_size`, and potentially reduced to ensure subsequent
  prefill is possible), along with the remaining tokens that still need to be
  processed.
- `save_existing_prefix`: Convenience function to save a KVCache. The key (token
  sequence) is truncated to a multiple of `chunk_size`. The function checks if a
  prefix for the truncated key already exists before saving.
- `cal_common_prefix_length`: Utility function to calculate the length of the
  common prefix between two token sequences.
"""

from collections import deque
from collections import OrderedDict
from typing import Any, Optional
import abc
import dataclasses
import jax
import jax.numpy as jnp
import logging
import numpy as np
import os
import sys
import threading

from jetstream.engine import engine_api

# Use for test prefix caching.
# Print statistic every load existing prefix success.
# Set just above logging.DEBUG(10) to prevent too many debug logs.
# Use custom integer log level may not well handle in other modules.
LOG_LEVEL_STATISTIC = 11

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# convert type to int if log_level is int
try:
  log_level = int(log_level)
except ValueError:
  pass

if isinstance(log_level, str):
  log_level = getattr(logging, log_level, logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

Token = int
# Tuple of tokens from prompt
Key = tuple[Token, ...]
Prefix = Any  # KVCache for one prompt


class Value:
  """Object stored contains the actual KVcache

  Attributes:
    prefix:
      Readonly. Prefix Cache using in model. Should be pytree of all jax.Array.
    true_length:
      Readonly. True length of tokens calculate prefix.
      Should be <= than len(tokens).
      true_length will be min(true_length, len(tokens))
    padded_length:
      Readonly. Length of tokens including padding calculate prefix.
    tokens:
      Readonly. Tokens calculate prefix. may include partial of padding.
    prefix_size_bytes:
      Readonly. bytes of prefix.
    device:
      Readonly. Devices of prefix. The same structure of pytree to prefix.
      The device may be different from actually prefix is in.
      It used for retrieved back to original device.
  """

  def __init__(
      self,
      *,
      prefix: Prefix,
      true_length: int,
      padded_length: int,
      tokens: tuple[Token, ...],
      prefix_size_bytes: Optional[int] = None,
      device=None,
  ):
    """Init Value with attributes.

    If true_length shorter than len(tokens),
    true_length will adjust to len(tokens).
    If prefix_size_bytes is not provided, calculate automatically.
    prefix should be pytree of all jax.Array.
    It may raise exception if there is anything not a jax.Array,
    If either prefix_size_bytes and device is None, get from prefix.
    """
    self._prefix = prefix
    self._true_length = self._maybe_adjust_true_length(true_length, tokens)
    self._padded_length = padded_length
    self._tokens = tokens
    if prefix_size_bytes is None:
      self._prefix_size_bytes: int = jax.tree.reduce(
          lambda acc, array: acc + array.nbytes,
          prefix,
          0,
      )
    else:
      self._prefix_size_bytes = prefix_size_bytes

    if device is None:
      self._device = jax.tree.map(lambda x: x.device, prefix)
    else:
      self._device = device

  @property
  def prefix(self) -> Prefix:
    return self._prefix

  @property
  def true_length(self) -> int:
    return self._true_length

  @property
  def padded_length(self) -> int:
    return self._padded_length

  @property
  def tokens(self) -> tuple[Token, ...]:
    return self._tokens

  @property
  def prefix_size_bytes(self) -> int:
    return self._prefix_size_bytes

  @property
  def device(self) -> int:
    return self._device

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Value):
      return False
    return (
        other.padded_length == self.padded_length
        and other.tokens == self.tokens
        and jax.tree.all(
            jax.tree.map(jnp.array_equal, other.prefix, self.prefix)
        )
        and other.prefix_size_bytes == self.prefix_size_bytes
    )

  def _maybe_adjust_true_length(
      self, true_length: int, tokens: tuple[Token, ...]
  ) -> int:
    if true_length > len(tokens):
      logger.warning(
          "true_length=%d should <= len(tokens)=%d.", true_length, len(tokens)
      )

    return min(true_length, len(tokens))


def device_put_value(value: Value, device: Any = None) -> Value:
  """Creates a new `Value`, optionally placing its `prefix` on a JAX device.

  This function always returns a new `Value` object.
  If `device` is provided, `jax.device_put(value.prefix, device)` is used for
  the new `Value`'s `prefix`. Otherwise, `value.prefix` is used directly (no
  copy or device transfer for the prefix).

  Note: The returned `Value`'s `device` attribute is always `value.device`,
  irrespective of the `device` argument.

  Args:
    value: The input `Value` object.
    device: Target JAX device for `value.prefix`. If `None`, `jax.device_put`
      is not called.
  Returns:
    A new `Value` instance with its `prefix` potentially on the specified
    `device`, and its `device` attribute set to `value.device`.
  """
  return_prefix = value.prefix
  # Prevent overhead that device_put need to parse the PyTree.
  if device is not None:
    return_prefix = jax.device_put(value.prefix, device)
  return Value(
      prefix=return_prefix,
      true_length=value.true_length,
      padded_length=value.padded_length,
      tokens=value.tokens,
      prefix_size_bytes=value.prefix_size_bytes,
      device=value.device,
  )


class PrefixCacheTrie:
  """Stores prefix tokens as a trie for fast lookup index.

  Insert longer Key replace shorter key to be the longest common prefix key.
  The shorter key will never be returned even if longer key is erased,
  and should got evicted in the future.

  Assume Key is equal length to tokens,
  which can be used to slice prompt and cache Value.
  Should check the return key common prefix length by the caller.

  If erase the Key not the leaf, nothing will happen.
  If erased key match at a leaf,
  delete the node and ancestors would be the leaf after deleted.
  """

  @dataclasses.dataclass
  class Node:
    """Trie Node."""

    parent: Optional["PrefixCacheTrie.Node"] = None
    token: Optional[Token] = None
    children: dict[Token, "PrefixCacheTrie.Node"] = dataclasses.field(
        default_factory=dict
    )

    def is_leaf(self):
      return len(self.children) == 0

    def get_one_child_token(self) -> Optional[Token]:
      if len(self.children) == 0:
        return None
      return next(iter(self.children.keys()))

  def __init__(self):
    self._saved_keys: list[Key] = []
    self._root = PrefixCacheTrie.Node()

  def insert(self, key: Key):
    """Insert key into the trie.

    If the key already exists, this operation does nothing.
    """
    node = self._root
    for token in key:
      if token not in node.children:
        node.children[token] = PrefixCacheTrie.Node(parent=node, token=token)
      node = node.children[token]

  def get_longest_common_prefix_key(
      self, key: Key
  ) -> tuple[Optional[Key], int]:
    """Get the key stored in the trie that has the longest common prefix
    with the input 'key', and the length of that common prefix.

    Returns:
      A tuple containing:
        - The full key stored in the trie that shares the longest prefix with
          the input `key`. None if no common prefix is found.
        - The length of the common prefix found. 0 if no common prefix.
    """
    common_prefix_tokens_from_input: list[Token] = []
    node = self._root

    for token_from_input in key:
      if token_from_input not in node.children:
        break
      node = node.children[token_from_input]
      common_prefix_tokens_from_input.append(token_from_input)

    length_of_common_prefix = len(common_prefix_tokens_from_input)

    if length_of_common_prefix == 0:
      return None, 0

    full_stored_key_tokens = list(common_prefix_tokens_from_input)
    current_node_for_extension = node

    while not current_node_for_extension.is_leaf():
      token_to_extend = current_node_for_extension.get_one_child_token()
      if (
          token_to_extend is None
      ):  # Should ideally not happen if is_leaf() is correct
        break
      full_stored_key_tokens.append(token_to_extend)
      current_node_for_extension = current_node_for_extension.children[
          token_to_extend
      ]

    return tuple(full_stored_key_tokens), length_of_common_prefix

  def contains(self, key: Key) -> bool:
    """Check if the exact key exists as a path in the trie.

    Args:
      key: The key (tuple of tokens) to search for.

    Returns:
      True if the key exists as a complete path in the trie, False otherwise.
    """
    node = self._root
    for token in key:
      if token not in node.children:
        return False
      node = node.children[token]
    return True

  def erase(self, key: Key) -> None:
    """Erase key in trie if it is leaf."""
    node = self._root
    for token in key:
      if token not in node.children:
        return
      node = node.children[token]

    while node.is_leaf():
      parent = node.parent
      if parent is None or node.token not in parent.children:
        return
      del parent.children[node.token]
      node = parent


class ValueStorageInterface(abc.ABC):
  """Interface for Value storage."""

  @abc.abstractmethod
  def get_max_size_bytes(self) -> int:
    """Get the max size bytes in storage."""

  @abc.abstractmethod
  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""

  @abc.abstractmethod
  def add_async(self, key: Key, value: Value) -> bool:
    """Add value and return True. If storage is full, return False."""

  @abc.abstractmethod
  def flush(self, key: Key) -> None:
    """Block and flush the key added to the storage."""

  @abc.abstractmethod
  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Return value from storage or None if not found.

    Args:
      key: key to retrieve value.
      device:
        The same as device in jax.device_put. Retrieve the value to device.
        If device is None,
        retrieve the value to it's original device while saved.
    Returns:
      Value retrieved from storage or None if not found.
    """

  @abc.abstractmethod
  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage."""

  @abc.abstractmethod
  def contains(self, key: Key) -> bool:
    """If there is key in storage."""


class BasicStorage:
  """Calculating size and save value into dict without modify."""

  def __init__(self, max_size_bytes: int):
    """
    Args:
      max_size_bytes: Maximum bytes use
    """
    self._max_size_bytes = max_size_bytes
    self._remain_size_bytes = max_size_bytes
    self._saved_values: dict[Key, Value] = {}

  def get_max_size_bytes(self) -> int:
    return self._max_size_bytes

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._remain_size_bytes >= needed_bytes

  def add(self, key: Key, value: Value) -> bool:
    """Add value and return True. If storage is full, return False.

    The value will not copied.
    Be aware not to modify the value after add to storage.
    Storage is expected to have enough space.
    """
    if self.contains(key):
      logger.warning("Key %r already exists in storage. Skipping add.", key)
      return False

    if not self.has_enough_space(value.prefix_size_bytes):
      logger.warning(
          (
              "should check enough space before add to storage, "
              "but remain=%d not enough for value=%d"
          ),
          self._remain_size_bytes,
          value.prefix_size_bytes,
      )
      return False

    self._saved_values[key] = value
    self._remain_size_bytes -= value.prefix_size_bytes
    return True

  def retrieve(self, key: Key) -> Optional[Value]:
    """Return value from storage or None if not found.

    Be aware the storage is not return a copy.
    Clone the Value first if additional modification needed.
    """
    if key not in self._saved_values:
      logger.warning(
          "key=%r should exist in storage before retrieve, but not found", key
      )
      return None
    return self._saved_values[key]

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage.
    Key is expected to be found.
    """
    if key not in self._saved_values:
      logger.warning(
          "key=%r should exist in storage before evict, but not found", key
      )
      return None
    value = self._saved_values.pop(key)
    self._remain_size_bytes += value.prefix_size_bytes
    return value

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return key in self._saved_values


class HBMStorage(ValueStorageInterface):
  """Stores kv storage values in HBM.

  Store the Value into the specific HBM device,
  which is the same type as device in jax.device_put.
  The Value would be jax.device_put to the HBM device after add,
  and retrieve back to the original device.
  """

  def __init__(self, max_size_bytes: int, device: Any = None):
    """Init the HBMStorage with max size limit and device to store the Value.

    Args:
      max_size_bytes: Maximum bytes of HBM to use for storage
      device:
        the same type as jax.device_put. It is used to store the cache Value.
        If None, do not move the Value.
    """
    self._storage = BasicStorage(max_size_bytes)
    self._device = device

  def get_max_size_bytes(self) -> int:
    return self._storage.get_max_size_bytes()

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._storage.has_enough_space(needed_bytes)

  def add_async(self, key: Key, value: Value) -> bool:
    """Add key/value pair into the cache.

    Depend on jax.device_put,
    the Value will not be copied
    if the device storing the cache is the same as the origin device of Value.
    Storage is expected to have enough space.

    Args:
      key: key of cache index.
      value: Value to store.
    Returns:
      True if successful. False if failed due to not enough space.
    """
    logger.debug("add to HBMStorage key=%r", key)
    if self.contains(key):
      return False

    hbm_value = device_put_value(value, self._device)
    return self._storage.add(key, hbm_value)

  def flush(self, key: Key) -> None:
    """Block until the value associated with the key is ready in HBM."""
    logger.debug("Flushed HBM key %r", key)
    value = self._storage.retrieve(key)
    if value is None:
      logger.warning("Key %r not found during flush in HBMStorage.", key)
      return
    jax.block_until_ready(value.prefix)

  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Retrieve value back to the original device or None if not found.

    Be aware the storage may not return a copy
    if the original devices is the same as depend on jax.device_put.
    Key is expected to be found.
    """
    logger.debug("retrieve from HBMStorage key=%r", key)
    hbm_value = self._storage.retrieve(key)
    if hbm_value is None:
      return None

    put_device = device
    if put_device is None:
      put_device = hbm_value.device

    return device_put_value(hbm_value, put_device)

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage.
    Key is expected to be found.
    """
    return self._storage.evict(key)

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return self._storage.contains(key)


class DRAMStorage(ValueStorageInterface):
  """Stores KV Cache values in host DRAM."""

  def __init__(self, max_size_bytes: int):
    """
    Args:
      max_size_bytes: Maximum bytes of host DRAM to use for storage
    """
    self._storage = BasicStorage(max_size_bytes)
    self._non_flushed_key = set()

  def get_max_size_bytes(self) -> int:
    return self._storage.get_max_size_bytes()

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._storage.has_enough_space(needed_bytes)

  def add_async(self, key: Key, value: Value) -> bool:
    """Adds value to host DRAM, returning False if space is insufficient.

    `value.prefix` is moved to host DRAM (copied if on device, referenced if
    already on host) *before* space is checked.
    **Always call `has_enough_space()` first.**

    Args:
      key: Key for the cache entry.
      value: `Value` object to store. If its `prefix` was already on host,
        do not modify it externally after this call as it's now referenced
        by the cache.
    Returns:
      True if added successfully, False otherwise.
    """
    logger.debug("add to DRAMStorage key=%r", key)
    if self.contains(key):
      return False

    host_value = Value(
        prefix=jax.copy_to_host_async(value.prefix),
        true_length=value.true_length,
        padded_length=value.padded_length,
        tokens=value.tokens,
        prefix_size_bytes=value.prefix_size_bytes,
        device=value.device,
    )

    ok = self._storage.add(key, host_value)
    if ok:
      self._non_flushed_key.add(key)
    return ok

  def flush(self, key: Key) -> None:
    """Block until the value associated with the key is ready in DRAM."""
    logger.debug("Flushed DRAM key %r.", key)
    if key not in self._non_flushed_key:
      return

    value = self._storage.evict(key)
    if value is None:
      logger.warning("Key %r not found during flush in DRAMStorage.", key)
      return

    flushed_value = Value(
        prefix=jax.device_get(value.prefix),
        true_length=value.true_length,
        padded_length=value.padded_length,
        tokens=value.tokens,
        prefix_size_bytes=value.prefix_size_bytes,
        device=value.device,
    )
    ok = self._storage.add(key, flushed_value)
    if not ok:
      logger.warning(
          "Cannot add flushed Value back to storage. Should not happen."
      )
    self._non_flushed_key.remove(key)

  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Return value from storage to the original device or None if not found.

    If the original device save in the storage is cpu,
    the storage will not copied.
    Do not modify the storage prefix retrieved.
    """
    logger.debug("retrieve from DRAMStorage key=%r", key)
    host_value = self._storage.retrieve(key)
    if host_value is None:
      return None

    put_device = device
    if put_device is None:
      put_device = host_value.device

    return device_put_value(host_value, put_device)

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage."""
    if key in self._non_flushed_key:
      self._non_flushed_key.remove(key)
    return self._storage.evict(key)

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return self._storage.contains(key)


class LRUStrategy:
  """Least recently used cache strategy manage key."""

  def __init__(self):
    self._order: OrderedDict[Key, None] = OrderedDict()

  def evict(self) -> Optional[Key]:
    """Return and pop the least recently used key."""
    if len(self._order) == 0:
      return None
    return self._order.popitem(last=False)[0]

  def use(self, key: Key) -> None:
    """Updated the usage history."""
    if key not in self._order:
      self._order[key] = None
    else:
      self._order.move_to_end(key, last=True)


@dataclasses.dataclass
class CacheLayer:
  """Storage with corresponding strategy"""

  storage: ValueStorageInterface
  strategy: LRUStrategy
  key_trie: PrefixCacheTrie = dataclasses.field(default_factory=PrefixCacheTrie)


class HierarchicalCache:
  """Hierarchical Cache contains two layers of ValueStorageInterface.

  The first layer contains subset fo key / value pairs of the second layer.
  The second storage max size bytes should >= first storage max size bytes.
  Use LRU for each layer.
  Add the Value will save to all layers.
  Retrieve the Value will retrieve to HBM and then saved to all layers.
  The added value size should less than the first layer max size.
  If the first layer max size cannot contains the added Value, add will failed.
  """

  def __init__(
      self, layers: tuple[ValueStorageInterface, ValueStorageInterface]
  ):
    assert (
        layers[0].get_max_size_bytes() <= layers[1].get_max_size_bytes()
    ), "Bottom layer of storage need to be larger than top."

    self._layers = [CacheLayer(storage, LRUStrategy()) for storage in layers]

  def contains(self, key: Key) -> bool:
    """Checks if the key exists in any layer.

    Args:
      key: The key to check for existence.

    Returns:
      True if the key exists in at least one layer, False otherwise.
    """
    return any(layer.key_trie.contains(key) for layer in self._layers)

  def add(self, key: Key, value: Value) -> tuple[bool, dict[Key, Value]]:
    """Add to all layers and return result

    return (ok, dict[fully evicted from hierarchical cache key value pair]).
    Beware in some error case,
    there may be not ok but have some evicted key value.
    """
    needed_bytes = value.prefix_size_bytes
    if self._layers[0].storage.get_max_size_bytes() < needed_bytes:
      logging.warning(
          (
              "Trying to add value larger than top layer max size. "
              "need_bytes=%d, max_size_bytes=%d"
          ),
          needed_bytes,
          self._layers[0].storage.get_max_size_bytes(),
      )
      return False, {}

    # Only return last layers evicted key value pair
    # which is fully evicted from hierarchical cache.
    all_ok = True
    fully_evicted_key_values: dict[Key, Value] = {}
    for layer in self._layers:
      if not layer.storage.contains(key):
        ok, last_layer_evicted_key_values = self._evict_to_enough_space(
            layer, needed_bytes
        )
        all_ok = all_ok and ok
      else:
        last_layer_evicted_key_values = {}

      for evict_key in last_layer_evicted_key_values:
        layer.key_trie.erase(evict_key)

      # Value evict from previous layer need to flush to current layer.
      # For example full evicted from HBM and flush to DRAM.
      # flush after evict to prevent flush evicted value don't need
      keys_evicted_from_previous_layer = fully_evicted_key_values.keys()
      for evict_key in keys_evicted_from_previous_layer:
        layer.storage.flush(evict_key)

      fully_evicted_key_values = last_layer_evicted_key_values

    if not all_ok:
      logging.error(
          "Cannot evict enough space "
          "after checking max_size is enough for bytes=%d.",
          needed_bytes,
      )
      return False, fully_evicted_key_values

    for layer in self._layers:
      if not layer.storage.contains(key):
        if not layer.storage.add_async(key, value):
          logging.error(
              "Cannot add to storage. key=%r, needed_bytes=%d",
              key,
              needed_bytes,
          )
          return False, fully_evicted_key_values
        layer.key_trie.insert(key)
      layer.strategy.use(key)

    return True, fully_evicted_key_values

  def get_longest_common_prefix_key_from_layers(
      self, key: Key
  ) -> Optional[tuple[Key, int, int]]:
    """Finds the stored key, its layer index, and common prefix length.

    It prioritizes the match that yields the longest common prefix with the
    input key. Ties are broken by preferring higher cache layers
    (smaller layer_idx).

    Args:
      key: The key to search for a prefix match.

    Returns:
      A tuple (stored_key, layer_idx, common_prefix_len) or None if no match.
    """
    best_stored_key: Optional[Key] = None
    best_layer_idx: int = -1
    max_common_prefix_len: int = 0

    for idx, layer in enumerate(self._layers):
      # stored_key_this_layer will be None if no common prefix with key in
      #  this layer's trie
      (
          stored_key_this_layer,
          common_len_this_layer,
      ) = layer.key_trie.get_longest_common_prefix_key(key)

      if stored_key_this_layer is not None:
        if common_len_this_layer > max_common_prefix_len:
          max_common_prefix_len = common_len_this_layer
          best_stored_key = stored_key_this_layer
          best_layer_idx = idx

    if best_stored_key is None:
      return None
    else:
      return best_stored_key, best_layer_idx, max_common_prefix_len

  def retrieve(
      self, key: Key, layer_idx: int, device: Any = None
  ) -> Optional[Value]:
    """Retrieves value from a layer, promotes it, and updates LRU status.

    The value is fetched from `self._layers[layer_idx]`. After retrieval,
    `self.add(key, value)` is called to ensure its presence in higher layers
    and update its LRU status.

    Args:
      key: The key of the value to retrieve.
      layer_idx: The index of the layer to retrieve the value from.
      device: Target JAX device for the `prefix` in the retrieved `Value`.
        If `None`, `Value.device` is used.
    Returns:
      The retrieved `Value`, or `None` if not found. The `Value.device`
      attribute of the returned object is not changed to the target `device`.
    """
    value = self._layers[layer_idx].storage.retrieve(key, device)
    if value is None:
      logging.warning(
          "Should check key exist before retrieve, but fail for key=%r", key
      )
      return None

    # Add to all layers from the top.
    self.add(key, value)

    return value

  def _evict_to_enough_space(
      self, layer: CacheLayer, needed_bytes: int
  ) -> tuple[bool, dict[Key, Value]]:
    """Evict layer to enough bytes for add and return results.

    return (ok, dict[evicted key, evicted value]).
    """
    evicted_key_values: dict[Key, Value] = {}
    while not layer.storage.has_enough_space(needed_bytes):
      evicted_key = layer.strategy.evict()
      if evicted_key is None:
        logging.error("Cannot evict enough space for bytes=%d.", needed_bytes)
        return False, evicted_key_values

      evicted_value = layer.storage.evict(evicted_key)
      if evicted_value is None:
        logging.error(
            "Key should in storage before evict but not. key=%r", evicted_key
        )
        continue

      evicted_key_values[evicted_key] = evicted_value

    return True, evicted_key_values


class CacheHitLengthStatistic:
  """Calculate recent k loading average cache hit length."""

  @dataclasses.dataclass
  class _Event:
    """Internal representation of a cache access event."""

    is_hit: bool
    key_length: int
    hit_length: int = 0  # Only relevant if is_hit is True
    hit_layer_idx: int = -1  # Only relevant if is_hit is True

  def __init__(self, recent_num: int, layer_num: int):
    """Initializes the statistic tracker.

    Args:
      recent_num: The number of recent cache access events to consider for
        statistics.
      layer_num: The number of cache layers being tracked.
    """
    if recent_num <= 0:
      raise ValueError("recent_num must be positive")
    if layer_num <= 0:
      raise ValueError("layer_num must be positive")
    self._history: deque[CacheHitLengthStatistic._Event] = deque(
        maxlen=recent_num
    )
    self._layer_num = layer_num

  def hit(self, hit_length: int, hit_layer_idx: int, key_length: int) -> None:
    """Records a cache hit event.

    Args:
      hit_length: The length of the prefix found in the cache.
      hit_layer_idx: The index of the cache layer where the hit occurred (0 for
        top layer).
      key_length: The total length of the key requested.
    """
    if hit_layer_idx < 0 or hit_layer_idx >= self._layer_num:
      raise ValueError(f"Invalid hit_layer_idx: {hit_layer_idx}")
    self._history.append(
        self._Event(
            is_hit=True,
            key_length=key_length,
            hit_length=hit_length,
            hit_layer_idx=hit_layer_idx,
        )
    )

  def no_hit(self, key_length: int) -> None:
    """Records a cache miss event."""
    self._history.append(self._Event(is_hit=False, key_length=key_length))

  def calculate_layers_hit_length_proportion(self) -> list[float]:
    """Calculates the hit length proportion for each layer based on history."""
    if not self._history:
      return [0.0] * self._layer_num

    total_requested_length = sum(event.key_length for event in self._history)
    if total_requested_length == 0:
      return [0.0] * self._layer_num

    layer_hit_lengths = [0] * self._layer_num
    for event in self._history:
      if event.is_hit:
        # Attribute hit length only to the layer it was found in
        layer_hit_lengths[event.hit_layer_idx] += event.hit_length

    # Calculate proportion for each layer based on total requested length
    proportions = [
        hit_len / total_requested_length for hit_len in layer_hit_lengths
    ]
    return proportions


class PrefixCache:
  """Store Prefix KV cache.

  Use hierarchical cache of two layers the first in the HBM
    and the second in the host DRAM.
  Assuming HBM is available, or the cache would degrade to two layers on DRAM.
  If cache is full, evict least-recently used entries (LRU).
  LRU strategy is apply to all layers.
  The cache in HBM will be subset of the cache in host DRAM.
  For example:
    For HBM can contain 2 values, and DRAM can contain 5 values,
    [1, 2, 3, 4, 5] LRU history,
    the [1, 2] will in HBM, and [1, 2, 3, 4, 5] will in DRAM.
  Always return cache after load into HBM.
  The value need to be <= to the max size in HBM.
  DRAM max size need to be >= than HBM max size.
  """

  def __init__(
      self, hbm_bytes: int, dram_bytes: int, hit_statistic_recent_num: int = 100
  ):
    """
    dram_bytes >= hbm_bytes
    Args:
      hbm_bytes: Total amount of HBM to use for cache.
      dram_bytes: Total amount of DRAM to use for cache.
      hit_statistic_recent_num: The number of recent cache access events to
        consider for statistics.
    """
    # TODO(yuyanpeng): way to disable DRAM cache
    assert (
        dram_bytes >= hbm_bytes
    ), "DRAM max size need to be >= than HBM max size."
    self._lock = threading.Lock()
    self._hbm_bytes = hbm_bytes
    self._dram_bytes = dram_bytes
    self._hit_statistic_recent_num = hit_statistic_recent_num
    # init in clear()
    self._cache: HierarchicalCache
    self._hit_statistic: CacheHitLengthStatistic
    self.clear()

  def contains(self, key: Key) -> bool:
    """Check if a key is in the cache.

    If the key is already in the cache. No need to save again assuming the
    value associated with the key will not change.

    Args:
      key: The key to check for existence in the cache.

    Returns:
      True if the key exists in the cache, False otherwise.
    """
    with self._lock:
      return self._cache.contains(key)

  def save(self, key: Key, value: Value) -> bool:
    """Save key/value to the cache."""
    logger.debug("save key=%r", key)
    with self._lock:
      ok, _ = self._cache.add(key, value)
      if not ok:
        logger.warning("Cannot add to cache")
        return False
      return True

  def load(
      self,
      key: Key,
      min_common_prefix_key_length: Optional[int] = None,
      device: Any = None,
  ) -> Optional[Value]:
    """Returns Value to the key with longest common prefix matched.

    Return None if not found.

    Args:
      key: key to search common prefix.
      min_common_prefix_key_length:
        If provided, only the longest common prefix key length
        >= the threshold will be loaded and return.
      device:
        The same type as device in the jax.device_put.
        Load the Value on the device.
        If None, load the Value on the original device Value.device.
    Return:
      Value stored with key or None if not found.
      The Value.device is not changed to device loaded on.
    """
    logger.debug("load key=%r", key)
    with self._lock:
      match_result = self._cache.get_longest_common_prefix_key_from_layers(key)
      if match_result is None:
        self._hit_statistic.no_hit(key_length=len(key))
        logger.debug("No matching key found in any layer for key=%r", key)
        return None
      (
          stored_key_from_cache,
          found_layer_idx,
          common_prefix_length,
      ) = match_result
      if (
          min_common_prefix_key_length is not None
          and common_prefix_length < min_common_prefix_key_length
      ):
        self._hit_statistic.no_hit(key_length=len(key))
        return None

      value = self._cache.retrieve(
          stored_key_from_cache, found_layer_idx, device=device
      )
      self._hit_statistic.hit(
          hit_length=common_prefix_length,
          hit_layer_idx=found_layer_idx,
          key_length=len(key),
      )
      if logger.isEnabledFor(LOG_LEVEL_STATISTIC):
        logger.log(
            LOG_LEVEL_STATISTIC, "%s", self._gen_statistic_str_without_lock()
        )
      return value

  def clear(self):
    """Clear entire cache."""
    logger.debug("clear cache")
    with self._lock:
      storage_layers = (
          HBMStorage(self._hbm_bytes),
          DRAMStorage(self._dram_bytes),
      )
      self._cache = HierarchicalCache(layers=storage_layers)
      self._hit_statistic = CacheHitLengthStatistic(
          recent_num=self._hit_statistic_recent_num,
          layer_num=len(storage_layers),
      )

  def gen_statistic_str(self):
    """Generates a string representation of the cache hit length proportion
      statistics.

    The statistics show the proportion of total requested token length that was
    served by each cache layer, based on the most recent access events.

    Returns:
      A string containing the formatted cache statistics. For example
      Cache Hit Length Proportion Statistics:
        Layer 0: 0.3396
        Layer 1: 0.1165
      (Based on the last 100 accesses)
    """
    with self._lock:
      return self._gen_statistic_str_without_lock()

  def _gen_statistic_str_without_lock(self) -> str:
    proportions = self._hit_statistic.calculate_layers_hit_length_proportion()
    result = "Cache Hit Length Proportion Statistics:\n"
    for i, prop in enumerate(proportions):
      result += f"  Layer {i}: {prop:.4f}\n"
    result += f"(Based on the last {self._hit_statistic_recent_num} accesses)\n"
    return result


def load_existing_prefix_and_get_remain_tokens(
    prefix_cache: PrefixCache,
    tokens: jax.Array | np.ndarray,
    chunk_size: int,
) -> tuple[Optional[engine_api.ExistingPrefix], jax.Array | np.ndarray]:
  """Return existing prefix and remained tokens need to calculate

  The existing prefix will truncate to a multiple of chunk_size and ensure at
  least one token remains to do prefill once.

  Args:
    prefix_cache: The PrefixCache instance to load from.
    tokens: The input token sequence for which to find a prefix.
    chunk_size: The chunk size used for prefilling. The returned common prefix
      length will be truncated to a multiple of this size.

  Returns:
    ExistingPrefix if at least one chunk or None.
    Remain tokens need to calculate. At least one token remains.

  """

  tuple_tokens = tuple(tokens.tolist())

  # Attempt to load the longest matching prefix from the cache.
  # We set a minimum length to ensure the match is at least one chunk long.
  cached_value = prefix_cache.load(
      tuple_tokens, min_common_prefix_key_length=chunk_size
  )

  if cached_value is None:
    logger.debug("No prefix found in cache for the given tokens.")
    return None, tokens

  # Calculate the actual length of the common prefix.
  common_prefix_len = cal_common_prefix_length(
      tuple_tokens, cached_value.tokens
  )

  if common_prefix_len < chunk_size:
    # This case might occur if the cache logic changes or if there's an
    # unexpected state.
    logger.debug(
        "Found prefix in cache, but common length (%d) is less than chunk"
        " size (%d).",
        common_prefix_len,
        chunk_size,
    )
    return None, tokens

  # Truncate the common prefix length to the nearest multiple of chunk_size.
  truncated_len = common_prefix_len - (common_prefix_len % chunk_size)

  # Ensure that at least one token remains for the next prefill step.
  # If the truncated length covers the entire input sequence,
  # reduce it by one chunk.
  if truncated_len == len(tokens):
    truncated_len -= chunk_size
    # If reducing by a chunk makes the length zero or less, it means the
    # original common prefix was exactly one chunk long and matched the
    # whole input. In this specific case, we can't use the cache effectively
    # for chunked prefill.
    if truncated_len <= 0:
      logger.debug(
          "Common prefix length %d reduced to %d after ensuring remaining"
          " tokens with chunk size %d. Cannot use cache.",
          common_prefix_len,
          truncated_len,
          chunk_size,
      )
      return None, tokens

  # Check if truncated_len became zero after the initial truncation (shouldn't
  # happen with min_common_prefix_key_length check).
  if truncated_len == 0:
    logger.debug(
        "Common prefix length %d truncated to 0 with chunk size %d.",
        common_prefix_len,
        chunk_size,
    )
    return None, tokens

  # Extract the truncated common prefix tokens.
  truncated_common_tokens = tokens[:truncated_len]
  remain_tokens = tokens[truncated_len:]

  # Create the ExistingPrefix object.
  existing_prefix = engine_api.ExistingPrefix(
      cache=cached_value.prefix, common_prefix_tokens=truncated_common_tokens
  )

  logger.debug(
      "Loaded existing prefix. Original common length: %d, Truncated length:"
      " %d",
      common_prefix_len,
      truncated_len,
  )

  return existing_prefix, remain_tokens


@jax.jit
def _copy_prefix_cache(cache):
  def _array_copy(x):
    return x.copy()

  return jax.tree.map(_array_copy, cache)


def save_existing_prefix(
    prefix_cache: PrefixCache,
    tokens: tuple[Token, ...],
    prefix: Prefix,
    chunk_size: int,
    padded_length: int,
    copy_prefix: bool,
) -> bool:
  """Save the existing prefix if needed.

  The tokens would be truncated to length of multiple of chunk_sizes.
  The truncated tokens would be the key to the cache.
  If key is in the prefix cache, it will not saved again.

  Args:
    prefix_cache: The PrefixCache instance to save to.
    tokens: The token sequence (as a tuple) representing the key.
    prefix: The KVCache (or relevant prefix data) to store.
    chunk_size: The chunk size used for prefilling. The saved length will be
      related to this.
    padded_length: The padded length of the KVCache in prefix.
    copy_prefix: Whether to copy the prefix before saving.

  Return:
    True if save happened and was successful, False otherwise.
  """

  save_len = len(tokens) - (len(tokens) % chunk_size)
  if save_len < chunk_size:
    logger.debug(
        "Token length %d is less than chunk size %d. Not saving prefix.",
        len(tokens),
        chunk_size,
    )
    return False

  truncated_tokens = tokens[:save_len]
  if prefix_cache.contains(truncated_tokens):
    logger.debug(
        "Prefix with key length %d already exists in cache. Not saving.",
        save_len,
    )
    return False

  copied_prefix = prefix
  if copy_prefix:
    copied_prefix = _copy_prefix_cache(prefix)

  value_to_save = Value(
      prefix=copied_prefix,
      true_length=len(truncated_tokens),
      padded_length=padded_length,
      tokens=truncated_tokens,
  )

  logger.debug("Saving prefix with key length %d to cache.", save_len)
  saved = prefix_cache.save(truncated_tokens, value_to_save)

  return saved


def cal_common_prefix_length(
    key: tuple[int, ...], matched_key: tuple[int, ...]
) -> int:
  length = 0
  for token1, token2 in zip(key, matched_key):
    if token1 != token2:
      break
    length += 1
  return length
