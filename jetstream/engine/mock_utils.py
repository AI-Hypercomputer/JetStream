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

"""Utils for engine operation."""

from typing import List, Sequence
from flax import struct
import numpy as np
from seqio.vocabularies import Vocabulary


@struct.dataclass
class TestTokenizer:
  """Tokenizer used for testing purposes."""

  def IdToPiece(self, integer: int) -> str:  # pylint: disable=invalid-name
    """In the real version, unlike encode_tf/decode_tf, doesn't strip trailing

    whitespace.
    """
    return chr(integer)

  def decode(self, tokens: np.ndarray):  # pylint: disable=invalid-name
    """Converts a numpy array into a string.

    Uses tokens[0] as we are doing streaming decode now
    """
    return chr(tokens[0])


@struct.dataclass
class TestVocab(Vocabulary):
  """Mock vocabulary used for tests.

  These methods are duplicative on the test vocab, but required to fit
  the seqio.Vocabulary interface.
  """

  pad_id = 0
  eos_id = 1
  bos_id = 2
  unk_id = 3
  stop_tokens = {pad_id, eos_id}
  _base_vocab_size = 2**16
  tokenizer: TestTokenizer = TestTokenizer()

  def _encode(self, s: str) -> Sequence[int]:
    """Converts a string into a integer sequenc."""
    # 'We use array methods, not python iterables so we don't
    # implement this method in the mock vocab.
    raise NotImplementedError

  def _decode(self, ids: np.ndarray):
    """Converts a numpy array into a string."""
    return "".join([chr(r) for r in list(ids) if r not in self.stop_tokens])

  def _encode_tf(self, s: str) -> np.ndarray:
    """Converts a string into a numpy array."""
    # We mock using numpy to avoid propagating tf dependencies.
    chars = np.array([ord(c) for c in s]).astype(np.int32)
    return chars

  def _decode_tf(self, ids: np.ndarray) -> List[str]:
    """Converts a numpy array into a string."""
    # We mock using numpy to avoid propagating tf dependencies.
    results = np.split(ids, ids.shape[0])
    return ["".join([chr(r) for r in list(line[0])]) for line in results]

  def decode(self, ids: np.ndarray, is_streaming=True):
    """Converts a numpy array into a string."""
    return is_streaming and self._decode(ids)

  def encode_tf(self, s: str) -> np.ndarray:
    """Converts a string into a numpy array."""
    return self._encode_tf(s)

  def decode_tf(self, ids: np.ndarray) -> List[str]:
    """Converts a numpy array into a string."""
    return self._decode_tf(ids)
