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
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import queue
from inference.nn import KVCache
from transformers import PretrainedConfig


class KVCacheStorage:

  def __init__(
      self,
      mesh: Mesh,
      model_config: PretrainedConfig,
      page_size: int,
      hbm_utilization: float,
  ):
    """Initializes the KVCacheStorage."""
    self.__mesh = mesh
    self.__num_layers = model_config.num_hidden_layers
    self.__num_kv_heads = model_config.num_key_value_heads
    self.__head_dim = model_config.head_dim
    self.__page_size = page_size
    self.hbm_kv_caches = self.__init_hbm_storage(jnp.bfloat16, hbm_utilization)
    self.__num_hbm_pages_per_layer = self.hbm_kv_caches[0].k.shape[1]

  @property
  def num_hbm_pages_per_layer(self):
    """Returns the number of HBM pages per layer."""
    return self.__num_hbm_pages_per_layer

  def __all_devices_hbm_bytes(self, hbm_utilization: float) -> int:
    """Returns the total usable HBM bytes across all devices."""
    assert 0.0 < hbm_utilization < 1.0
    try:
      per_device_memory_stats = jax.devices()[0].memory_stats()
      limit = per_device_memory_stats["bytes_reservable_limit"]
      used = per_device_memory_stats["bytes_in_use"]
      usable = int(limit * hbm_utilization)
      GB = 1024**3
      print(
          f"per device memory stats: limit={limit//GB}GB, used={used//GB}GB, usable={usable//GB}GB"
      )
      return (usable - used) * self.__mesh.devices.size
    except:
      print(f"per device memory stats: not available")
      return 0

  def __kv_bytes_per_page(self, dtype: jnp.dtype) -> int:
    """Returns the number of bytes per key _and_ value page."""
    item_size = jnp.ones((1,), dtype).itemsize
    per_kv_bytes = self.__num_kv_heads * self.__head_dim * item_size * 2
    return self.__page_size * per_kv_bytes

  def __shape(self) -> tuple[int, int, int, int]:
    """Returns the shape of the kv cache."""
    return (
        self.__num_kv_heads,
        self.num_pages_per_layer,
        self.__page_size,
        self.__head_dim,
    )

  def __sharding(self) -> NamedSharding:
    """Returns the sharding of the kv cache."""
    return NamedSharding(
        self.__mesh, P(self.__mesh.axis_names, None, None, None)
    )

  def __gen_kv_cache(self, dtype: jnp.dtype) -> KVCache:
    """Generates one per-token kv cache item."""
    # TODO: support 2d sharding
    return KVCache(
        k=jnp.ones(shape=self.__shape(), device=self.__sharding(), dtype=dtype),
        v=jnp.ones(shape=self.__shape(), device=self.__sharding(), dtype=dtype),
    )

  def __init_hbm_storage(
      self,
      dtype: jnp.dtype,
      hbm_utilization: float,
  ) -> list[KVCache]:
    """Initializes the kv cache storage across all devices."""
    all_devices_hbm_bytes = self.__all_devices_hbm_bytes(hbm_utilization)
    if all_devices_hbm_bytes > 0:
      kv_bytes_per_page = self.__kv_bytes_per_page(dtype)
      per_layer_hbm_bytes = all_devices_hbm_bytes // self.__num_layers
      self.num_pages_per_layer = per_layer_hbm_bytes // kv_bytes_per_page
    else:
      self.num_pages_per_layer = 1000

    kv_storage = [self.__gen_kv_cache(dtype) for _ in range(self.__num_layers)]
    return kv_storage


class KVCacheManager:
  """Logical KV Cache Manager"""

  def __init__(
      self,
      num_hbm_pages,
      page_size,
  ):
    """Initializes the KVCacheManager."""
    self.__page_size = page_size
    self.__dummy_page_idx = 0
    self.__available_hbm_pages = queue.SimpleQueue()
    for p in range(1, num_hbm_pages):
      self.__available_hbm_pages.put_nowait(p)

  @property
  def page_size(self):
    """Returns the page size in the number of per-token kv cache items."""
    return self.__page_size

  @property
  def dummy_page_idx(self):
    """Returns the dummy page index (0)."""
    return self.__dummy_page_idx

  def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
    """Allocates HBM pages for prompt prefill."""
    n = math.ceil(prompt_len / self.__page_size)
    return self.alloc_hbm_pages(n)

  def alloc_hbm_pages(self, n: int) -> list[int]:
    """Allocates `n` HBM pages."""
    if 0 < n <= self.__available_hbm_pages.qsize():
      return [self.__available_hbm_pages.get(block=True) for _ in range(n)]
    else:
      return []

  def free_hbm_pages(self, pages: list[int]):
    """Frees the given HBM pages."""
    for p in pages:
      if p != self.__dummy_page_idx:
        self.__available_hbm_pages.put_nowait(p)
