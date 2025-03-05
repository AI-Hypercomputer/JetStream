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
import os
import jax.experimental.shard_map
import numpy as np
import jax.experimental
import jax.experimental.mesh_utils
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from inference import parallel
from inference import nn


class NormTest(absltest.TestCase):

  def _create_device_mesh(self):
    devices = jax.devices()
    return parallel.create_device_mesh(
        devices=devices,
        shape=(len(devices), 1),
    )

  def test_rmsnorm_per_device_forward(self):
    mesh = self._create_device_mesh()
    hidden_state_size = 128
    eps = 1e-6
    rmsnorm_layer = nn.RMSNorm(
        dim=hidden_state_size,
        eps=eps,
        parallel_config=parallel.RMSNormParallelConfig(
            mesh=mesh,
            activation_sharded=False,
        ),
    )
    distributed_rmsnorm_layer = nn.RMSNorm(
        dim=hidden_state_size,
        eps=eps,
        parallel_config=parallel.RMSNormParallelConfig(
            mesh=mesh,
            activation_sharded=True,
        ),
    )

    key = jax.random.PRNGKey(0)
    input = jax.random.uniform(key, (96, hidden_state_size))
    weight = jax.random.uniform(key, (hidden_state_size,))
    sharded_weight = jnp.copy(weight)

    rmsnorm_layer.load_weights_dict({"weight": weight})
    dis_weight = distributed_rmsnorm_layer.load_weights_dict(
        {"weight": sharded_weight}
    )
    expect = rmsnorm_layer(input)

    sharded_input = jax.device_put(
        input, NamedSharding(mesh, P(None, parallel.tp_axis_names()))
    )

    def distributed_rms_ag(weight, input):
      output = distributed_rmsnorm_layer.jittable_call(weight, input)
      return parallel.ops.all_gather(output, 1, parallel.tp_axis_names())

    got = jax.experimental.shard_map.shard_map(
        distributed_rms_ag,
        mesh,
        in_specs=(
            P(parallel.tp_axis_names()),
            P(None, parallel.tp_axis_names()),
        ),
        out_specs=P(None, None),
        check_rep=False,
    )(dis_weight, sharded_input)

    np.testing.assert_allclose(got, expect, atol=1e-6, rtol=1e-7)


if __name__ == "__main__":
  absltest.main()
