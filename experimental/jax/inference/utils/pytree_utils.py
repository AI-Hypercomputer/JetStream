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

from jax.tree_util import register_pytree_node


def register_flat_dataclass_as_pytree(cls):
  def flatten(obj):
    children, aux_data = (), None
    for field in cls.__dataclass_fields__:
      children += (getattr(obj, field),)
    return (children, aux_data)

  def unflatten(aux_data, children):
    return cls(*children)

  register_pytree_node(cls, flatten, unflatten)
  return cls
