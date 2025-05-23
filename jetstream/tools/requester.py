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
from jetstream.engine.token_utils import load_vocab


_SERVER = flags.DEFINE_string("server", "0.0.0.0", "server address")
_PORT = flags.DEFINE_string("port", "9000", "port to ping")
_TEXT = flags.DEFINE_string("text", "My dog is cute", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 3, "Maximum number of output/decode tokens of a sequence"
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 1, "Number of responses to generate per request"
)
_TOKENIZER = flags.DEFINE_string(
    "tokenizer",
    None,
    "Name or path of the tokenizer (matched to the model)",
    required=True,
)
_CLIENT_SIDE_TOKENIZATION = flags.DEFINE_bool(
    "client_side_tokenization",
    False,
    "Enable client side tokenization with tokenizer.",
)
_LORA_ADAPTER_ID = flags.DEFINE_string(
    "lora_adapter_id",
    "",
    "ID of the adapter for this decode request.",
    required=False,
)


def _GetResponseAsync(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    request: jetstream_pb2.DecodeRequest,
) -> None:
  """Gets an async response."""

  response = stub.Decode(request)
  outputs = {i: [] for i in range(request.num_samples)}
  for resp in response:
    # print(len(resp.stream_content.samples))
    for i, sample in enumerate(resp.stream_content.samples):
      if _CLIENT_SIDE_TOKENIZATION.value:
        outputs[i].extend(sample.token_ids)
      else:
        outputs[i].extend(sample.text)
  print(f"Prompt: {_TEXT.value}")
  for i, output in outputs.items():
    if _CLIENT_SIDE_TOKENIZATION.value:
      vocab = load_vocab(_TOKENIZER.value)
      text_output = vocab.tokenizer.decode(output)
    else:
      text_output = "".join(output)
    print(f"The {i}th Response: {text_output}")


def main(argv: Sequence[str]) -> None:
  del argv
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  address = f"{_SERVER.value}:{_PORT.value}"
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print(f"Sending request to: {address}")
    if _CLIENT_SIDE_TOKENIZATION.value:
      vocab = load_vocab(_TOKENIZER.value)
      token_ids = vocab.tokenizer.encode(_TEXT.value)
      request = jetstream_pb2.DecodeRequest(
          token_content=jetstream_pb2.DecodeRequest.TokenContent(
              token_ids=token_ids
          ),
          max_tokens=_MAX_TOKENS.value,
          num_samples=_NUM_SAMPLES.value,
          lora_adapter_id=_LORA_ADAPTER_ID.value,
      )
    else:
      request = jetstream_pb2.DecodeRequest(
          text_content=jetstream_pb2.DecodeRequest.TextContent(
              text=_TEXT.value
          ),
          max_tokens=_MAX_TOKENS.value,
          num_samples=_NUM_SAMPLES.value,
          lora_adapter_id=_LORA_ADAPTER_ID.value,
      )
    return _GetResponseAsync(stub, request)


if __name__ == "__main__":
  app.run(main)
