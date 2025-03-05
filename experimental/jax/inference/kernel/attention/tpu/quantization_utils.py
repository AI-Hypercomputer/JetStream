# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple
import jax
from jax import numpy as jnp

P = jax.sharding.PartitionSpec
MAX_INT8 = 127
MIN_INT8 = -128


class QuantizedTensor(NamedTuple):
  """A tensor which has been quantized to int8 and its scales.

  Attributes:
    weight: Weight
    scales: Scales
  """

  weight: jnp.ndarray
  scales: jnp.ndarray


def to_int8(x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
  """Converts a float array to an int8 array with a scale.

  Args:
    x: Float array.
    h: Quantization scale.

  Returns:
    Int8 array.
  """
  x = x * h
  return jnp.clip(jnp.round(x), MIN_INT8, MAX_INT8).astype(jnp.int8)


def from_int8(
    x: jnp.ndarray, h: jnp.ndarray, dtype: jnp.dtype = jnp.bfloat16
) -> jnp.ndarray:
  """Converts an int8 array to a float array with a scale.

  Args:
    x: Int8 array.
    h: Quantization scale.
    dtype: Float dtype to convert to.

  Returns:
    Float array.
  """
  x = x.astype(dtype=dtype) / h
  return x.astype(dtype=dtype)


def get_quantization_scales(x: jnp.ndarray, axis=-1) -> jnp.ndarray:
  """Computes the quantization scales for a float array.

  These are the maximum values of the trailing dimension.

  Args:
    x: Float array to quantize.

  Returns:
    Array of the same shape as input but with the trailing dimension reduced to
    a size 1 absolute max value.
  """
  scale_reciprocal = MAX_INT8 / jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  return scale_reciprocal.astype(jnp.float32)


def quantize_to_int8(
    x: jnp.ndarray,
    axis=-1,
) -> QuantizedTensor:
  """Quantizes a float array to an int8 QuantizedTensor.

  Args:
    x: Float array to quantize.

  Returns:
    Int8 QuantizedTensor.
  """
  x_scales = get_quantization_scales(x, axis=axis)
  return QuantizedTensor(weight=to_int8(x, x_scales), scales=x_scales)


def unquantize_from_int8(
    x: QuantizedTensor,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
  """Unquantizes an int8 QuantizedTensor to a float array.

  Args:
    x: Int8 QuantizedTensor to unquantize.
    dtype: Float dtype to unquantize to.

  Returns:
    Float array.
  """
  return from_int8(x.weight, x.scales, dtype)
