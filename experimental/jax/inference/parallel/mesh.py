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

"""mesh module"""

import math
from collections.abc import Iterable
from typing import Sequence
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils as jax_mesh_utils
from jaxlib.xla_client import Device
import numpy as np
from inference.parallel.config import ParallelAxis
from inference.parallel.device import platform


def create_device_mesh(
    devices: Sequence[Device],
    shape: tuple[int],
) -> Mesh:
  """Create a powerful mesh given the devices and shape.

  For fully connected topology, the devices topology defined in the mesh
  usually not affect the performance.
  For other cases, the devices topology defined in the mesh
  will affect the performance. (it depends on the collective algorithm
  implementation.)
  """
  assert devices, "no devices is provided for mesh creation"
  assert len(devices) == math.prod(shape)
  axis_names = (
      ParallelAxis.X.name,
      ParallelAxis.Y.name,
  )
  assert len(shape) == len(axis_names)

  p = platform()
  if p == "gpu" or p == "cpu":
    devices = jax_mesh_utils.create_device_mesh(
        mesh_shape=shape,
        devices=devices,
        allow_split_physical_axes=True,
    )
    return Mesh(devices=devices, axis_names=axis_names)
  else:
    assert p == "tpu"
    # TODO: Figure out a general method.
    # Current mesh builder is very limited.
    # only support (2,x) underlying topology shape to .
    # form a 1D ring for the devices.
    devices = devices[::2] + devices[1::2][::-1]
    devices = np.reshape(devices, shape)
    return Mesh(devices=devices, axis_names=axis_names)


def get_num_partitions(axis_names):
  """Get the total number of partitions across the axis.
  Args:
      axis_names: the name of the axis where the partition is
          relative to. If the number of axis is greater than 1,
          the axis names need to follow the "major to minor" order
          as defined in the mesh setup for consistency.
  """
  if not isinstance(axis_names, Iterable):
    return jax.lax.psum(1, axis_name=axis_names)
  product = 1
  for axis in axis_names[::-1]:
    product = product * jax.lax.psum(1, axis_name=axis)
  return product


def get_partition_index(axis_names):
  """Get the partition index across the axis for the device.
  Args:
      axis_names: the names of the axis where the partition index is
          relative to. If the number of axis is greater than 1,
          the axis names need to follow the "major to minor" order
          as defined in the mesh setup for consistency.
  """
  if not isinstance(axis_names, Iterable):
    return jax.lax.axis_index(axis_name=axis_names)
  cur_idx = 0
  multiplicand = 1
  for axis in axis_names[::-1]:
    cur_idx += jax.lax.axis_index(axis) * multiplicand
    multiplicand = multiplicand * jax.lax.psum(1, axis_name=axis)
  return cur_idx
