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

"""Defines the JetStream Tokenizer API."""

import abc
from typing import Any, Tuple, Union

import numpy as np
import jax

# Class to store token ids.
ResultTokens = Any


class Tokenizer(abc.ABC):
  """Tokenizer to convert strings to token ids and vice-versa."""

  @abc.abstractmethod
  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments.
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """

  @abc.abstractmethod
  def decode(self, token_ids: list[int], **kwargs) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
      **kwargs: Additional keyword arguments.
    Returns:
      str: String generated from the token ids.
    """

  @property
  @abc.abstractmethod
  def pad_id(self) -> int:
    """ID of the pad token."""

  @property
  @abc.abstractmethod
  def eos_id(self) -> int:
    """ID of EOS token."""

  @property
  @abc.abstractmethod
  def bos_id(self) -> int:
    """ID of BOS token."""

  @property
  def stop_tokens(self) -> set[int]:
    """ID of the stop token."""
    return {self.eos_id, self.pad_id}
