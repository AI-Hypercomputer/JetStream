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
import numpy as np
import jax.experimental.mesh_utils
import jax
from inference import parallel
from inference import nn


class EmbeddingTest(absltest.TestCase):

  def _create_device_mesh(self):
    devices = jax.devices()
    return parallel.create_device_mesh(
        devices=devices,
        shape=(len(devices), 1),
    )

  def test_embedding(self):
    mesh = self._create_device_mesh()
    vocal_size, emb_dim = 2048, 8192
    embedding_layer = nn.Embedding(
        vocal_size,
        emb_dim,
        parallel_config=parallel.EmbeddingParallelConfig(
            mesh=mesh,
            parallel_type=parallel.EmbeddingParallelType.COLUMN,
        ),
    )
    key = jax.random.key(0)
    emb_table = jax.random.uniform(key, (vocal_size, emb_dim))
    input = jax.random.randint(key, (96,), 0, 2048)
    expect = emb_table[input]

    embedding_layer.load_weights_dict({"weight": emb_table})
    got = embedding_layer(input)
    np.testing.assert_allclose(got, expect)


if __name__ == "__main__":
  absltest.main()
