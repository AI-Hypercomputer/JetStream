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

"""A test request."""

from typing import Sequence

from absl import app
from absl import flags
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc


_SERVER = flags.DEFINE_string('server', 'dns:///[::1]', 'server address')
_PORT = flags.DEFINE_string('port', '9000', 'port to ping')
_SESSION_CACHE = flags.DEFINE_string(
    'session_cache', '', 'Location of any pre-cached results'
)
_TEXT = flags.DEFINE_string('text', 'Today is a good day', 'The message')
_PRIORITY = flags.DEFINE_integer('priority', 0, 'Message priority')
_MAX_TOKENS = flags.DEFINE_integer('max_tokens', 3, 'Maximum number of tokens')


def _GetResponseAsync(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    request: jetstream_pb2.DecodeRequest,
) -> None:
  """Gets an async response."""

  response = stub.Decode(request)
  output = ""
  for token_list in response:
    output += token_list.response[0]
  print(f'Prompt: {_TEXT.value}')
  print(f'Response: {output}')


def main(argv: Sequence[str]) -> None:
  del argv
  # Note: Uses insecure_channel only for local testing. Please add grpc credentials for Production.
  address = f'{_SERVER.value}:{_PORT.value}'
  with grpc.insecure_channel(
      address
  ) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print(f'Sending request to: {address}')
    request = jetstream_pb2.DecodeRequest(
        session_cache=_SESSION_CACHE.value,
        additional_text=_TEXT.value,
        priority=_PRIORITY.value,
        max_tokens=_MAX_TOKENS.value,
    )
    return _GetResponseAsync(stub, request)


if __name__ == '__main__':
  app.run(main)
