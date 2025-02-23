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

  def setUp(self):
    super().setUp()
    self.w0, self.w1, self.w2, self.w3 = (
        jnp.ones((1,)),
        jnp.ones((2,)),
        jnp.ones((3,)),
        jnp.ones((4,)),
    )
    self.parent_module = Module()
    self.parent_module.w0 = Parameter(self.w0)
    self.h1_child_0_module = Module()
    self.h1_child_0_module.w1 = Parameter(self.w1)
    self.h1_child_1_module = Module()
    self.h1_child_1_module.w2 = Parameter(self.w2)
    self.h2_child_0_module = Module()
    self.h2_child_0_module.w3 = Parameter(self.w3)

    self.parent_module.child0 = self.h1_child_0_module
    self.parent_module.child1 = self.h1_child_1_module
    self.h1_child_0_module.child0 = self.h2_child_0_module

  def test_random_code_initialize(self):
    self.parent_module.init_weights()
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        self.w0,
        self.parent_module.w0.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        self.w1,
        self.h1_child_0_module.w1.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        self.w2,
        self.h1_child_1_module.w2.value,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        self.w3,
        self.h2_child_0_module.w3.value,
    )

  def test_load_weights_dict(self):
    parent_weight_dict = {
        "w0": jnp.ones((1,)),
        "child0": {
            "w1": jnp.ones((2,)),
            "child0": {
                "w3": jnp.ones((4,)),
            },
        },
        "child1": {
            "w2": jnp.ones((3,)),
        },
    }

    self.parent_module.load_weights_dict(parent_weight_dict)

    np.testing.assert_array_equal(self.parent_module.w0, self.w0)
    np.testing.assert_array_equal(self.h1_child_0_module.w1, self.w1)
    np.testing.assert_array_equal(self.h1_child_1_module.w2, self.w2)
    np.testing.assert_array_equal(self.h2_child_0_module.w3, self.w3)

  def test_load_weights_dict_error(self):
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

    self.parent_module.load_weights_dict(partial_parent_weight_dict)
    self.h1_child_1_module.load_weights_dict(child1_weight_dict)

    np.testing.assert_array_equal(self.parent_module.w0, 0)
    np.testing.assert_array_equal(self.h1_child_0_module.w1, 0)
    np.testing.assert_array_equal(self.h1_child_1_module.w2, 0)
    np.testing.assert_array_equal(self.h2_child_0_module.w3, 0)

    assert not self.h1_child_1_module.wrong_weight_not_load


if __name__ == "__main__":
  absltest.main()
