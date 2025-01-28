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
import numpy as np
from jax import numpy as jnp
from inference.nn import Module, Parameter


class ModuleTest(absltest.TestCase):

  def test_random_code_initialize(self):
    w0, w1, w2, w3 = (
        jnp.ones((1,)),
        jnp.ones((2,)),
        jnp.ones((3,)),
        jnp.ones((4,)),
    )
    parent_module = Module()
    parent_module.w0 = Parameter(w0)

    h1_child_0_module = Module()
    h1_child_0_module.w1 = Parameter(w1)

    h1_child_1_module = Module()
    h1_child_1_module.w2 = Parameter(w2)

    h2_child_0_module = Module()
    h2_child_0_module.w3 = Parameter(w3)

    parent_module.child0 = h1_child_0_module
    parent_module.child1 = h1_child_1_module
    h1_child_0_module.child0 = h2_child_0_module

    parent_module.init_weights()

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        w0,
        parent_module.w0.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        w1,
        h1_child_0_module.w1.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        w2,
        h1_child_1_module.w2.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        w3,
        h2_child_0_module.w3.value,
    )

  def test_load_weights_dict(self):
    w0, w1, w2, w3 = (
        jnp.ones((1,)),
        jnp.ones((2,)),
        jnp.ones((3,)),
        jnp.ones((4,)),
    )
    parent_module = Module()
    parent_module.w0 = Parameter(w0)

    h1_child_0_module = Module()
    h1_child_0_module.w1 = Parameter(w1)

    h1_child_1_module = Module()
    h1_child_1_module.w2 = Parameter(w2)

    h2_child_0_module = Module()
    h2_child_0_module.w3 = Parameter(w3)

    parent_module.child0 = h1_child_0_module
    parent_module.child1 = h1_child_1_module
    h1_child_0_module.child0 = h2_child_0_module
    print(parent_module)

    partial_parent_weight_dict = {
        "w0": jnp.zeros((1,)),
        "child0": {
            "w1": jnp.zeros((2,)),
            "child0": {
                "w3": jnp.zeros((4,)),
            },
        },
    }

    child1_weight_dict = {
        "w2": jnp.zeros((2,)),
        "wrong_weight_not_load": jnp.zeros((2,)),
    }

    parent_module.load_weights_dict(partial_parent_weight_dict)
    h1_child_1_module.load_weights_dict(child1_weight_dict)

    np.testing.assert_array_equal(parent_module.w0, 0)
    np.testing.assert_array_equal(h1_child_0_module.w1, 0)
    np.testing.assert_array_equal(h1_child_1_module.w2, 0)
    np.testing.assert_array_equal(h2_child_0_module.w3, 0)

    assert not h1_child_1_module.wrong_weight_not_load


if __name__ == "__main__":
  absltest.main()
