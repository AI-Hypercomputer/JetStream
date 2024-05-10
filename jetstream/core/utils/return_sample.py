"""ReturnSample is a data structure utility.

It is a data structure that stores the return samples.
"""

import dataclasses


@dataclasses.dataclass
class ReturnSample:
  """Both the token ids, their string representation, and other data.

  Used so that the client knows when special token_ids have been returned
  (i.e.<image_start>) and displays other modalities instead of text, but does
  not burden the user with maintaining a vocab.

  Attributes:
    text: Text piece(s) detokenized from token id(s).
    token_ids: Raw result token id(s).
  """

  text: list[str]
  token_ids: list[int]
