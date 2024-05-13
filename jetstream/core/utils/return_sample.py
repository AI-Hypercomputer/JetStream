"""ReturnSample is a data structure utility.

It is a data structure that stores the return samples.
"""

import dataclasses


@dataclasses.dataclass
class ReturnSample:
  """Both the token ids, their string representation, and other data.

  Attributes:
    text: Text piece(s) detokenized from token id(s).
    token_ids: Raw result token id(s).
  """

  text: list[str]
  token_ids: list[int]
