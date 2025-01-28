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

"""Miniature load test of the mock server."""

import concurrent.futures
import functools
import time
from typing import Iterator, Sequence

from absl import app
from absl import flags
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc


_SERVER = flags.DEFINE_string("server", "0.0.0.0", "server address")
_PORT = flags.DEFINE_string("port", "9000", "port to ping")
_TEXT = flags.DEFINE_string("text", "AB", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 100, "Maximum number of output/decode tokens of a sequence"
)


def collect_tokens(
    response: Iterator[jetstream_pb2.DecodeResponse], print_interim: bool
) -> list[str]:
  tokens = []
  for resp in response:
    text_pieces = resp.stream_content.samples[0].text
    if print_interim:
      print(text_pieces, end="", flush=True)
    tokens.extend(text_pieces)
  return tokens


def api_call(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    text: str,
    max_tokens: int,
    print_interim: bool = True,
) -> str:
  """Sends a request to server and returns text."""
  request = jetstream_pb2.DecodeRequest(
      text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
      max_tokens=max_tokens,
  )
  response = stub.Decode(request)
  print("---------------------- Sent!!!----------------------")
  tokens = collect_tokens(response, print_interim=print_interim)

  return "".join(tokens)


def ping(
    stub: jetstream_pb2_grpc.OrchestratorStub, text: str, number: int
) -> str:
  response = api_call(stub, text, _MAX_TOKENS.value, print_interim=False)
  print(f"Completed {number}")
  return response


def load_test(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    text: list[str],
    queries: int = 64,
) -> list[str]:
  """Sends many queries to the server."""
  assert queries % len(text) == 0
  # repeat out
  text = text * (queries // len(text))
  number = list(range(len(text)))
  start = time.time()
  ping_partial = functools.partial(ping, stub)
  with concurrent.futures.ThreadPoolExecutor(max_workers=queries) as executor:
    responses = list(executor.map(ping_partial, text, number))
  time_taken = time.time() - start
  print(f"Time taken: {time_taken}")
  print(f"QPS: {queries/time_taken}")
  return responses


def main(argv: Sequence[str]):
  del argv
  address = f"{_SERVER.value}:{_PORT.value}"
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    _ = load_test(stub, text=[_TEXT.value], queries=64)


if __name__ == "__main__":
  app.run(main)
