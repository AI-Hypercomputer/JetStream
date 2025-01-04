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


class KVCacheStorage:

  def __init__(
      self,
      mesh: Mesh,
      model_config,
      page_size: int = 32,
      hbm_utilization: float = 0.8,
  ):
    self.mesh = mesh
    self.num_devices = mesh.devices.size
    self.num_layers = model_config.num_hidden_layers
    self.num_kv_heads = model_config.num_key_value_heads
    self.head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    self.page_size = page_size

    self.hbm_kv_caches: list[KVCache] = self.init_hbm_storage(hbm_utilization)
    self.num_hbm_pages = self.hbm_kv_caches[0].k.shape[1]

  def init_hbm_storage(
      self,
      hbm_utilization: float = 0.8,
      cache_dtype: jnp.dtype = jnp.bfloat16,
  ):
    memory_stats = jax.devices()[0].memory_stats()
    if memory_stats:
      print("per device memory_stats: ", memory_stats)
      available_hbm_bytes = (
          memory_stats["bytes_reservable_limit"] * hbm_utilization
          - memory_stats["bytes_in_use"]
      )
      item_size = jnp.ones((1), dtype=cache_dtype).itemsize
      kv_size_per_page = (
          item_size
          * self.num_kv_heads
          // self.num_devices
          * self.head_dim
          * self.page_size
          * 2
      )
      num_pages_per_layer = int(
          available_hbm_bytes // kv_size_per_page // self.num_layers
      )
    else:
      print("memory_stats not available, allocate 128 pages.")
      num_pages_per_layer = 128

    # TODO: support 2d sharding.
    kv_storage = [
        KVCache(
            k=jnp.ones(
                shape=(
                    self.num_kv_heads,
                    num_pages_per_layer,
                    self.page_size,
                    self.head_dim,
                ),
                device=NamedSharding(
                    self.mesh, P(self.mesh.axis_names, None, None, None)
                ),
                dtype=cache_dtype,
            ),
            v=jnp.ones(
                shape=(
                    self.num_kv_heads,
                    num_pages_per_layer,
                    self.page_size,
                    self.head_dim,
                ),
                device=NamedSharding(
                    self.mesh, P(self.mesh.axis_names, None, None, None)
                ),
                dtype=cache_dtype,
            ),
        )
        for _ in range(self.num_layers)
    ]

    return kv_storage


class KVCacheManager:
  """Logical KV Cache Manager"""

  def __init__(
      self,
      num_hbm_pages,
      page_size,
  ):
    self.available_hbm_pages = queue.SimpleQueue()
    self.page_size = page_size
    self.dummy_page_idx = 0

    for i in range(1, num_hbm_pages):
      self.available_hbm_pages.put_nowait(i)

  @property
  def num_available_hbm_pages(self):
    return self.available_hbm_pages.qsize()

  def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
    num_pages = math.ceil(prompt_len / self.page_size)
    if num_pages > self.num_available_hbm_pages:
      return []
    else:
      return self.alloc_hbm_pages(num_pages)

  def alloc_hbm_pages(self, num_pages: int) -> list[int]:
    pages_to_use = []
    if num_pages > self.num_available_hbm_pages:
      return pages_to_use

    for _ in range(num_pages):
      pages_to_use.append(self.available_hbm_pages.get())
    return pages_to_use

  def free_hbm_pages(self, pages: list[int]):
    for p in pages:
      if p != self.dummy_page_idx:
        self.available_hbm_pages.put_nowait(p)
