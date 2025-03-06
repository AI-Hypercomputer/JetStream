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

"""Utility module."""

import jax
from jax.sharding import PartitionSpec as P


def get_partition_spec(sharded_pytree):
  def pspec(a):
    if isinstance(a, jax.Array):
      return a.sharding.spec
    elif isinstance(a, int) or isinstance(a, float):
      return P()
    else:
      raise ValueError(f"unknown parition spec for {a}")

  return jax.tree_util.tree_map(pspec, sharded_pytree)
