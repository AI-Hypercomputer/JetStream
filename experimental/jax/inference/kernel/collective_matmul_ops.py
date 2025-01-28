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

from collections.abc import Callable
from typing import Sequence
import functools
import jax
from inference import parallel
from .linear.tpu.collective_matmul import *


def build_collective_matmul(
    type: parallel.CollectiveMatmulType,
    axis_names: str | Sequence[str],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
  if type == parallel.CollectiveMatmulType.ALL_GATHER:
    return functools.partial(
        all_gather_collective_matmul,
        axis_names=axis_names,
    )
  elif type == parallel.CollectiveMatmulType.REDUCE_SCATTER:
    return functools.partial(
        collective_matmul_reduce_scatter,
        axis_names=axis_names,
    )
  else:
    raise ValueError(f"Unsupported collective matmul type {type}")
