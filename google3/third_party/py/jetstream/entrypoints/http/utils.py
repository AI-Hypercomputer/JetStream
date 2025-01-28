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

"""Http API server utilities."""

from google.protobuf.json_format import MessageToJson


async def proto_to_json_generator(proto_generator):
  """Wraps a generator yielding Protocol Buffer messages into a generator

  yielding JSON messages.
  """
  async for proto_message in proto_generator:
    json_string = MessageToJson(proto_message)
    yield json_string
