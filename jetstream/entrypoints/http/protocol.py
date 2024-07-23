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

"""Http API server protocol."""

from pydantic import BaseModel  # type: ignore


class TextContent(BaseModel):
  text: str


class TokenContent(BaseModel):
  token_ids: list[int]


class DecodeRequest(BaseModel):
  max_tokens: int
  text_content: TextContent | None = None
  token_content: TokenContent | None = None

  # Config to enforce the oneof behavior at runtime.
  class Config:
    extra = "forbid"  # Prevent extra fields.
    anystr_strip_whitespace = True
