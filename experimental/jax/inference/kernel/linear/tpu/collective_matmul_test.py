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
import numpy as np
import jax.experimental
import jax.experimental.mesh_utils
import jax
from jax import numpy as jnp
from jax.experimental import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from inference import kernel
from inference import parallel


class CollectiveMatmulTest(absltest.TestCase):

  def _create_device_mesh(self):
    devices = jax.devices()
    return parallel.create_device_mesh(
        devices=devices,
        shape=(len(devices), 1),
    )

  def test_all_gather_collective_matmul(self):
    key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key1, shape=(1, 32), dtype=jnp.float32)
    rhs = jax.random.normal(key2, shape=(32, 16), dtype=jnp.float32)
    expect = lhs @ rhs

    mesh = self._create_device_mesh()
    axis_names = mesh.axis_names
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, axis_names)))
    rhs = kernel.prepare_rhs_for_all_gather_collective_matmul(rhs, mesh)

    def agcm(lhs, rhs, type, axis_names):
      return kernel.build_collective_matmul(type, axis_names)(lhs, rhs)

    got = shard_map.shard_map(
        f=partial(
            agcm,
            type=parallel.CollectiveMatmulType.ALL_GATHER,
            axis_names=axis_names,
        ),
        mesh=mesh,
        in_specs=(P(None, axis_names), P(None, axis_names)),
        out_specs=P(None, axis_names),
    )(lhs, rhs)
    np.testing.assert_allclose(got, expect, rtol=1e-6)

  def test_collective_matmul_reduce_scatter(self):
    key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.uniform(key1, shape=(8, 64), dtype=jnp.float32)
    rhs = jax.random.uniform(key2, shape=(64, 64), dtype=jnp.float32)
    expect = lhs @ rhs

    mesh = self._create_device_mesh()
    axis_names = mesh.axis_names
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(axis_names, None)))

    rhs = kernel.prepare_rhs_for_collective_matmul_reduce_scatter(rhs, mesh)

    def cmrc(lhs, rhs, type, axis_names):
      return kernel.build_collective_matmul(type, axis_names)(lhs, rhs)

    got = shard_map.shard_map(
        f=partial(
            cmrc,
            type=parallel.CollectiveMatmulType.REDUCE_SCATTER,
            axis_names=axis_names,
        ),
        mesh=mesh,
        in_specs=(P(None, axis_names), P(axis_names, None)),
        out_specs=P(None, axis_names),
    )(lhs, rhs)
    np.testing.assert_allclose(got, expect, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
