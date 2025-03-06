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

"""Simple NN Module.

TODO: migrate to Flax NNX.
"""

import logging
from typing import Any
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from inference.nn.parameter import Parameter


class Module:
  """Simple NN module."""

  def __init__(self):
    self._parameters: dict[str, Parameter] = {}
    self._submodules: dict[str, Module] = {}

  def __setattr__(self, name: str, value: Any):
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Module | ModuleList):
      self._submodules[name] = value
    else:
      self.__dict__[name] = value

  def __getattr__(self, name: str):
    if name in self._parameters:
      return self._parameters[name]
    elif name in self._submodules:
      return self._submodules[name]
    elif name in self.__dict__:
      return self.__dict__[name]
    return None

  def init_weights(self):
    res = {}
    rng = jax.random.key(0)
    for k, param in self._parameters.items():
      param.value = jax.random.uniform(rng, param.shape, dtype=jnp.bfloat16)
      param.to_device()
      res[k] = param.value

    for k, module in self._submodules.items():
      res[k] = module.init_weights()
    return res

  def load_weights_dict(self, weights_dict):
    res = {}
    for k, v in weights_dict.items():
      attr = getattr(self, k)
      if isinstance(attr, Parameter):
        param = self._parameters[k]
        if v.shape != param.shape:
          logging.warning(
              f"Not matched shape"
              + f": defined {param.shape},"
              + f"loaded {v.shape} for module {self.__class__}"
          )
        param.value = v
        if isinstance(param.sharding, NamedSharding):
          param.to_device()
        res[k] = param.value
      elif isinstance(attr, Module) or isinstance(attr, ModuleList):
        sub_weights_dict = self._submodules[k].load_weights_dict(v)
        res[k] = sub_weights_dict
      else:
        logging.warning(
            f"Unknown checkpoint key {k} for module {self.__class__}"
        )
    return res

  def _weights_assignment_in_jit(self, weights_dict):
    for key in self._parameters:
      param = self._parameters[key]
      if isinstance(param, Parameter):
        param.value = weights_dict[key]
    for key, module in self._submodules.items():
      if key in weights_dict:
        module._weights_assignment_in_jit(weights_dict[key])

  def jittable_call(self, weights_dict, *args):
    self._weights_assignment_in_jit(weights_dict)
    return self(*args)

  # Following methods "__repr__", "_repr_with_indent" and "_spec" are
  # for debugging purpose which provide a clean string representation
  # for the model.
  def _spec(self) -> str:
    return ""

  def _repr_with_indent(self, indent) -> str:
    indent = indent + "  "
    if len(self._parameters) == 0 and len(self._submodules) == 0:
      return "{}"
    res = "{"
    for k, v in self._parameters.items():
      res += "\n" + (indent) + f"'{k}': {v}"

    for k, v in self._submodules.items():
      res += (
          "\n"
          + (indent)
          + f"'{k}': <{v.__class__.__name__}{v._spec()}> {v._repr_with_indent(indent)}"
      )

    res += "\n" + indent[:-2] + "}"
    return res

  def __repr__(self) -> str:
    return "\n" + self._repr_with_indent("")


class ModuleList:

  def __init__(self, modules: list[Module]) -> None:
    self._modules: dict[int, Module] = {}
    for i, m in enumerate(modules):
      self._modules[i] = m

  def __getitem__(self, key):
    return self._modules[key]

  def __setitem__(self, key, value):
    self._modules[key] = value

  def _spec(self) -> str:
    return ""

  def _repr_with_indent(self, indent) -> str:
    indent = indent + "  "
    if len(self._modules) == 0:
      return "{}"
    res = "{"

    for k, v in self._modules.items():
      res += (
          "\n"
          + (indent)
          + f"'{k}': <{v.__class__.__name__}{v._spec()}> {v._repr_with_indent(indent)}"
      )

    res += "\n" + indent[:-2] + "}"
    return res

  def __repr__(self) -> str:
    return "\n" + self._repr_with_indent("")

  def init_weights(self):
    res = {}
    for k, module in self._modules.items():
      if isinstance(k, int):
        res[k] = module.init_weights()
      else:
        logging.warning(f"Unknown checkpoint key {k} for module list")
    return res

  def load_weights_dict(self, weights_dict):
    res = {}
    for k, v in weights_dict.items():
      if isinstance(k, int):
        ws = self._modules[k].load_weights_dict(v)
        res[k] = ws
      else:
        logging.warning(f"Unknown checkpoint key {k} for module list")
    return res

  def _weights_assignment_in_jit(self, weights_dict):
    for layer_num, module in self._modules.items():
      module._weights_assignment_in_jit(weights_dict[layer_num])


class Model(Module):

  def jittable_call(
      self,
      weights_dict,
      input_ids,
      positions,
      kv_caches,
      attn_metadata,
  ) -> tuple[jax.Array, list[Any]]:
    self._weights_assignment_in_jit(weights_dict)
    return self(input_ids, positions, kv_caches, attn_metadata)


class CausalLM(Module):

  def jittable_call(
      self,
      weights_dict,
      input_ids: jax.Array,
      positions: jax.Array,
      kv_caches: Any,
      attn_metadata: Any,
      sampling_params: Any,
  ) -> tuple[jax.Array, list[Any]]:
    self._weights_assignment_in_jit(weights_dict)
    return self(
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        sampling_params,
    )
