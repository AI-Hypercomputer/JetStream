"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""kv cache module"""

import math
import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import queue
from inference.nn import KVCache
from transformers import PretrainedConfig

_GB = 1024**3


class KVCacheStorage:

  def __init__(
      self,
      mesh: Mesh,
      model_config: PretrainedConfig,
      page_size: int,
      hbm_utilization: float,
  ):
    """Initializes the KVCacheStorage."""
    self._mesh = mesh
    self._num_layers = model_config.num_hidden_layers
    self._num_kv_heads = model_config.num_key_value_heads
    self._head_dim = model_config.head_dim
    self._page_size = page_size
    self._hbm_kv_caches = self._init_hbm_storage(jnp.bfloat16, hbm_utilization)
    self._num_hbm_pages_per_layer = self._hbm_kv_caches[0].k.shape[1]

  @property
  def hbm_kv_caches(self) -> list[KVCache]:
    """Returns the kv cache storage."""
    return self._hbm_kv_caches

  @hbm_kv_caches.setter
  def hbm_kv_caches(self, kv_caches: list[KVCache]):
    """Sets the kv cache storage."""
    self._hbm_kv_caches = kv_caches

  @property
  def num_hbm_pages_per_layer(self) -> int:
    """Returns the number of HBM pages per layer."""
    return self._num_hbm_pages_per_layer

  def _all_devices_hbm_bytes(self, hbm_utilization: float) -> int:
    """Returns the total usable HBM bytes across all devices."""
    assert 0.0 < hbm_utilization < 1.0
    print("per device memory stats:", end="")
    try:
      per_device_memory_stats = jax.devices()[0].memory_stats()
      limit = per_device_memory_stats["bytes_reservable_limit"]
      used = per_device_memory_stats["bytes_in_use"]
      usable = int(limit * hbm_utilization) - used
      limit_GB, used_GB, usable_GB = limit // _GB, used // _GB, usable // _GB
      print(f" limit={limit_GB}GB, used={used_GB}GB, usable={usable_GB}GB")
      return usable * self._mesh.devices.size
    except:
      print(" not available")
      return 0

  def _kv_bytes_per_page(self, dtype: jnp.dtype) -> int:
    """Returns the number of bytes per key _and_ value page."""
    item_size = np.dtype(dtype).itemsize
    per_kv_bytes = self._num_kv_heads * self._head_dim * item_size * 2
    return self._page_size * per_kv_bytes

  def _shape(self) -> tuple[int, int, int, int]:
    """Returns the shape of the kv cache."""
    return (
        self._num_kv_heads,
        self.num_pages_per_layer,
        self._page_size,
        self._head_dim,
    )

  def _sharding(self) -> NamedSharding:
    """Returns the sharding of the kv cache."""
    return NamedSharding(self._mesh, P(self._mesh.axis_names, None, None, None))

  def _gen_kv_cache(self, dtype: jnp.dtype) -> KVCache:
    """Generates one per-token kv cache item."""
    # TODO: support 2d sharding
    return KVCache(
        k=jnp.ones(shape=self._shape(), device=self._sharding(), dtype=dtype),
        v=jnp.ones(shape=self._shape(), device=self._sharding(), dtype=dtype),
    )

  def _init_hbm_storage(
      self,
      dtype: jnp.dtype,
      hbm_utilization: float,
  ) -> list[KVCache]:
    """Initializes the kv cache storage across all devices."""
    all_devices_hbm_bytes = self._all_devices_hbm_bytes(hbm_utilization)
    if not all_devices_hbm_bytes >= 1 * _GB:
      raise ValueError("Insufficient HBM memory")

    kv_bytes_per_page = self._kv_bytes_per_page(dtype)
    per_layer_hbm_bytes = all_devices_hbm_bytes // self._num_layers
    self.num_pages_per_layer = per_layer_hbm_bytes // kv_bytes_per_page

    kv_storage = [self._gen_kv_cache(dtype) for _ in range(self._num_layers)]
    return kv_storage


class KVCacheManager:
  """Logical KV Cache Manager"""

  def __init__(
      self,
      num_hbm_pages,
      page_size,
  ):
    """Initializes the KVCacheManager."""
    self._page_size = page_size
    self._dummy_page_idx = 0
    self._available_hbm_pages = queue.SimpleQueue()
    for p in range(1, num_hbm_pages):
      self._available_hbm_pages.put_nowait(p)

  @property
  def page_size(self):
    """Returns the page size in the number of per-token kv cache items."""
    return self._page_size

  @property
  def dummy_page_idx(self):
    """Returns the dummy page index (0)."""
    return self._dummy_page_idx

  def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
    """Allocates HBM pages for prompt prefill."""
    n = math.ceil(prompt_len / self._page_size)
    return self.alloc_hbm_pages(n)

  def alloc_hbm_pages(self, n: int) -> list[int]:
    """Allocates `n` HBM pages."""
    if 0 < n <= self._available_hbm_pages.qsize():
      return [self._available_hbm_pages.get(block=True) for _ in range(n)]
    else:
      return []

  def free_hbm_pages(self, pages: list[int]):
    """Frees the given HBM pages."""
    for p in pages:
      if p != self._dummy_page_idx:
        self._available_hbm_pages.put_nowait(p)
