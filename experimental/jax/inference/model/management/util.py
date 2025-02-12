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

"""Model management utility module """
import jax
from jax import numpy as jnp
import torch

torch_jax_dtype_map = {
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    jnp.bfloat16: torch.bfloat16,
    jnp.float32: torch.float32,
}


def convert_to_jax_array_on_cpu(x: torch.Tensor):
  device = jax.devices("cpu")[0]
  return jax.device_put(
      jnp.asarray(x.float().numpy(), dtype=torch_jax_dtype_map[x.dtype]),
      device=device,
  )
