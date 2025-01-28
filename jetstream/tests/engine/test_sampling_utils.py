# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests functionality of inference sampling utils."""

import jax
import jax.numpy as jnp
import unittest
from jetstream.engine import sampling_utils


class SamplingUtilsTest(unittest.TestCase):

  def setUp(self):
    self.rng = jax.random.PRNGKey(0)
    self.logits = jnp.array([[-0.5, 1.2, 0.8], [-1.0, 0.3, 0.7]])

  def test_greedy_sampling(self):
    token = sampling_utils.sampling(self.logits, self.rng, "greedy")
    expected_token = jnp.array([1, 2])
    self.assertTrue(jnp.array_equal(token, expected_token))

  def test_weighted_sampling(self):
    # Multiple samples to increase the chance of catching errors
    for _ in range(10):
      result = sampling_utils.sampling(self.logits, self.rng, "weighted")
      self.assertTrue(
          jnp.all(jnp.isin(result, jnp.array([0, 1, 2])))
      )  # Check if sampled from valid indices

  def test_nucleus_sampling(self):
    for _ in range(10):
      result = sampling_utils.sampling(
          self.logits, self.rng, "nucleus", nucleus_topp=0.8
      )
      self.assertTrue(jnp.all(jnp.isin(result, jnp.array([0, 1, 2]))))
    invalid_topp = -0.1
    with self.assertRaises(ValueError) as context:
      sampling_utils.sampling(
          self.logits, self.rng, "nucleus", nucleus_topp=invalid_topp
      )
      self.assertIn(
          f"Can't apply nucleus with parameter {invalid_topp=} less zero",
          str(context.exception),
      )

  def test_topk_sampling(self):
    for _ in range(10):
      result = sampling_utils.sampling(self.logits, self.rng, "topk", topk=2)
      self.assertTrue(
          jnp.all(jnp.isin(result, jnp.array([1, 2])))
      )  # Only top 2 logits should be sampled
    invalid_topk = 0
    with self.assertRaises(ValueError) as context:
      sampling_utils.sampling(self.logits, self.rng, "topk", topk=invalid_topk)
      self.assertIn(
          f"Can't apply algorithm topk with parameter {invalid_topk=} <= 0",
          str(context.exception),
      )

  def test_unsupported_algorithm(self):
    with self.assertRaises(ValueError):
      sampling_utils.sampling(self.logits, self.rng, "unsupported_algorithm")
