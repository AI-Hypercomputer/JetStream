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

from absl.testing import absltest
from functools import partial
import os
import numpy as np
import jax.experimental
import jax.experimental.mesh_utils
import jax
from jax import numpy as jnp
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as P
from inference import parallel


class CollectiveOperationsTest(absltest.TestCase):

  def _build_mesh(self):
    devices = jax.devices()
    device_mesh = jax.experimental.mesh_utils.create_device_mesh(
        (len(devices),), devices
    )
    axis = "x"
    mesh = jax.sharding.Mesh(device_mesh, (axis,))
    return mesh, axis

  def test_reduce_scatter(self):
    key = jax.random.key(99)
    operand = jax.random.uniform(key, shape=(16 * 32, 1024), dtype=jnp.float32)
    mesh, axis = self._build_mesh()

    expect = shard_map.shard_map(
        f=partial(
            jax.lax.psum_scatter,
            axis_name=axis,
            scatter_dimension=1,
            tiled=True,
        ),
        mesh=mesh,
        in_specs=P(axis, None),
        out_specs=P(None, axis),
    )(operand)

    got = shard_map.shard_map(
        f=partial(
            parallel.ops.reduce_scatter, axis_names=axis, scatter_dimension=1
        ),
        mesh=mesh,
        in_specs=P(axis, None),
        out_specs=P(None, axis),
    )(operand)

    np.testing.assert_allclose(got, expect, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
