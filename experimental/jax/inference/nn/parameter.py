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

"""Simple Parameter.
The code is mainly from the flax.nnx.variables.
TODO: migrate to Flax nnx.
"""

import jax
from jax import numpy as jnp
import typing as tp
from typing import Any
from jax.sharding import NamedSharding

A = tp.TypeVar("A")
V = tp.TypeVar("V", bound="Parameter[Any]")


class Parameter:
  """Parameter class.

  It composes of the jax.Array and routes the computation
  function to the jax.Array itself.
  The code is mainly from the flax.nnx.variables.
  TODO: migrate to Flax nnx.
  """

  def __init__(self, value: jax.Array):
    self.value = jnp.ones((0,))
    self._defined_shape = value.shape
    self._defined_dtype = value.dtype
    self._defined_sharding = value.sharding

  def _shape(self):
    return self._defined_shape

  def _set_shape(self, shape: tuple):
    self._defined_shape = shape

  def _sharding(self):
    return self._defined_sharding

  def _set_sharding(self, sharding: NamedSharding):
    self._defined_sharding = sharding

  def _dtype(self):
    return self._defined_dtype

  def _set_dtype(self, dtype):
    self._dtype = dtype

  shape = property(_shape, _set_shape)
  dtype = property(_dtype, _set_dtype)
  sharding = property(_sharding, _set_sharding)

  def to_device(self):
    self.value = jax.device_put(self.value, self._defined_sharding)
    return self

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(shape: {self._defined_shape}, dtype: {self._defined_dtype}, sharding: {self._defined_sharding})"

  def __setattr__(self, name: str, value: Any) -> None:
    object.__setattr__(self, name, value)

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------
  # NOTE: we dont override __setattr__ to avoid cases where
  # you need to set an attribute on the variable instance
  # def __getattr__(self, name: str) -> tp.Any:
  #     print("zhihaoshan",name)
  #     if name == "sharding":
  #         return self.sharding
  #     return getattr(self.value, name)

  def __getitem__(self, key) -> tp.Any:
    return self.value[key]  # type: ignore

  def __setitem__(self, key, value) -> None:
    self.value[key] = value  # type: ignore

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.value(*args, **kwargs)  # type: ignore

  def __len__(self) -> int:
    return len(self.value)  # type: ignore

  def __iter__(self) -> tp.Iterator:
    return iter(self.value)  # type: ignore

  def __contains__(self, item) -> bool:
    return item in self.value  # type: ignore

  def __add__(self, other) -> A:
    return self.value.__add__(other)  # type: ignore

  def __sub__(self, other) -> A:
    return self.value.__sub__(other)  # type: ignore

  def __mul__(self, other) -> A:
    return self.value.__mul__(other)  # type: ignore

  def __matmul__(self, other) -> A:
    return self.value.__matmul__(other)  # type: ignore

  def __truediv__(self, other) -> A:
    return self.value.__truediv__(other)  # type: ignore

  def __floordiv__(self, other) -> A:
    return self.value.__floordiv__(other)  # type: ignore

  def __mod__(self, other) -> A:
    return self.value.__mod__(other)  # type: ignore

  def __divmod__(self, other) -> A:
    return self.value.__divmod__(other)  # type: ignore

  def __pow__(self, other) -> A:
    return self.value.__pow__(other)  # type: ignore

  def __lshift__(self, other) -> A:
    return self.value.__lshift__(other)  # type: ignore

  def __rshift__(self, other) -> A:
    return self.value.__rshift__(other)  # type: ignore

  def __and__(self, other) -> A:
    return self.value.__and__(other)  # type: ignore

  def __xor__(self, other) -> A:
    return self.value.__xor__(other)  # type: ignore

  def __or__(self, other) -> A:
    return self.value.__or__(other)  # type: ignore

  def __radd__(self, other) -> A:
    return self.value.__radd__(other)  # type: ignore

  def __rsub__(self, other) -> A:
    return self.value.__rsub__(other)  # type: ignore

  def __rmul__(self, other) -> A:
    return self.value.__rmul__(other)  # type: ignore

  def __rmatmul__(self, other) -> A:
    return self.value.__rmatmul__(other)  # type: ignore

  def __rtruediv__(self, other) -> A:
    return self.value.__rtruediv__(other)  # type: ignore

  def __rfloordiv__(self, other) -> A:
    return self.value.__rfloordiv__(other)  # type: ignore

  def __rmod__(self, other) -> A:
    return self.value.__rmod__(other)  # type: ignore

  def __rdivmod__(self, other) -> A:
    return self.value.__rdivmod__(other)  # type: ignore

  def __rpow__(self, other) -> A:
    return self.value.__rpow__(other)  # type: ignore

  def __rlshift__(self, other) -> A:
    return self.value.__rlshift__(other)  # type: ignore

  def __rrshift__(self, other) -> A:
    return self.value.__rrshift__(other)  # type: ignore

  def __rand__(self, other) -> A:
    return self.value.__rand__(other)  # type: ignore

  def __rxor__(self, other) -> A:
    return self.value.__rxor__(other)  # type: ignore

  def __ror__(self, other) -> A:
    return self.value.__ror__(other)  # type: ignore

  def __iadd__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__iadd__"):
      value.__iadd__(other)
    else:
      self.value = value.__add__(other)
    return self

  def __isub__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__isub__"):
      value.__isub__(other)
    else:
      self.value = value.__sub__(other)
    return self

  def __imul__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__imul__"):
      value.__imul__(other)
    else:
      self.value = value.__mul__(other)
    return self

  def __imatmul__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__imatmul__"):
      value.__imatmul__(other)
    else:
      self.value = value.__matmul__(other)
    return self

  def __itruediv__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__itruediv__"):
      value.__itruediv__(other)
    else:
      self.value = value.__truediv__(other)
    return self

  def __ifloordiv__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__ifloordiv__"):
      value.__ifloordiv__(other)
    else:
      self.value = value.__floordiv__(other)
    return self

  def __imod__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__imod__"):
      value.__imod__(other)
    else:
      self.value = value.__mod__(other)
    return self

  def __ipow__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__ipow__"):
      value.__ipow__(other)
    else:
      self.value = value.__pow__(other)
    return self

  def __ilshift__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__ilshift__"):
      value.__ilshift__(other)
    else:
      self.value = value.__lshift__(other)
    return self

  def __irshift__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__irshift__"):
      value.__irshift__(other)
    else:
      self.value = value.__rshift__(other)
    return self

  def __iand__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__iand__"):
      value.__iand__(other)
    else:
      self.value = value.__and__(other)
    return self

  def __ixor__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__ixor__"):
      value.__ixor__(other)
    else:
      self.value = value.__xor__(other)
    return self

  def __ior__(self: V, other) -> V:
    value = self.value
    if hasattr(value, "__ior__"):
      value.__ior__(other)
    else:
      self.value = value.__or__(other)
    return self

  def __neg__(self) -> A:
    return self.value.__neg__()  # type: ignore

  def __pos__(self) -> A:
    return self.value.__pos__()  # type: ignore

  def __abs__(self) -> A:
    return self.value.__abs__()  # type: ignore

  def __invert__(self) -> A:
    return self.value.__invert__()  # type: ignore

  def __complex__(self) -> A:
    return self.value.__complex__()  # type: ignore

  def __int__(self) -> A:
    return self.value.__int__()  # type: ignore

  def __float__(self) -> A:
    return self.value.__float__()  # type: ignore

  def __index__(self) -> A:
    return self.value.__index__()  # type: ignore

  def __round__(self, ndigits: int) -> A:
    return self.value.__round__(ndigits)  # type: ignore

  def __trunc__(self) -> A:
    return self.value.__trunc__()  # type: ignore

  def __floor__(self) -> A:
    return self.value.__floor__()  # type: ignore

  def __ceil__(self) -> A:
    return self.value.__ceil__()  # type: ignore
