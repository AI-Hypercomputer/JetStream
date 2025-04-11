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

"""A gRPC client to interact with JetStream Server."""

from typing import Sequence

from absl import app
from absl import flags
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core.proto import multi_lora_decoding_pb2
from jetstream.core.proto import multi_lora_decoding_pb2_grpc
from jetstream.engine.token_utils import load_vocab


_SERVER = flags.DEFINE_string("server", "0.0.0.0", "server address")
_PORT = flags.DEFINE_string("port", "9000", "port to ping")
# _TEXT = flags.DEFINE_string("text", "My dog is cute", "The message")
_TEXT = flags.DEFINE_string("text", "22 year old", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 3, "Maximum number of output/decode tokens of a sequence"
)

_LORA_ADAPTER_ID = flags.DEFINE_string(
    "lora_adapter_id",
    None,
    "Id of the fine-tuned adapter to be loaded on top of the base model.",
    required=False,
)

_LORA_ADAPTER_PATH = flags.DEFINE_string(
    "lora_adapter_path",
    None,
    "Path of the fine-tuned adapter to be loaded from.",
    required=False,
)

_TEST_API_NAME = flags.DEFINE_string(
    "test_api_name",
    None,
    "Name of the JetStream API to call.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  """
  Main function for a gRPC client that interacts with a JetStream server.

  This client can:
    - Load a LoRA adapter.
    - Unload a LoRA adapter.
    - List loaded adapters and their metadata.
    - Generate text completions (using LoRA adapters if specified).

  The client uses command-line flags to specify the server address, port,
  text input, maximum number of tokens, adapter ID, adapter path, and the
  API to call.  It uses insecure gRPC channels (suitable for local testing).

  Args:
    argv: Command-line arguments (not used directly, flags are used instead).

  Raises:
    ValueError: For invalid configurations, like missing required parameters
      for specific API calls.
  """

  del argv  # Unused
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  address = f"{_SERVER.value}:{_PORT.value}"
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = multi_lora_decoding_pb2_grpc.v1Stub(channel)
    print(f"Sending request to: {address}")

    if _TEST_API_NAME.value == "load_lora_adapter":
      print(f"Calling the /v1/load_lora_adapter.")

      adapter_id = _LORA_ADAPTER_ID.value
      adapter_path = _LORA_ADAPTER_PATH.value

      if adapter_id == None or adapter_path == None:
        print(
            f"For `load_lora_adapter` API call, `adapter_id` and `adapter_path` must be passed."
        )
        return

      request = multi_lora_decoding_pb2.LoadAdapterRequest(
          adapter_id=adapter_id, adapter_path=adapter_path
      )

      response = stub.load_lora_adapter(request)

      if response.success is True:
        print(f"Adapter={adapter_id} is loaded successfully.")
      else:
        print(
            f"Adapter={adapter_id} loading failed with error={response.error_message}"
        )

    elif _TEST_API_NAME.value == "unload_lora_adapter":
      print(f"Calling the /v1/unload_lora_adapter.")

      adapter_id = _LORA_ADAPTER_ID.value

      if adapter_id == None:
        print(
            f"For `unload_lora_adapter` API call, `adapter_id` must be passed."
        )
        return

      request = multi_lora_decoding_pb2.UnloadAdapterRequest(
          adapter_id=adapter_id,
      )

      response = stub.unload_lora_adapter(request)

      if response.success is True:
        print(f"Adapter={adapter_id} is unloaded successfully.")
      else:
        print(
            f"Adapter={adapter_id} unloading failed with error={response.error_message}"
        )

    elif _TEST_API_NAME.value == "models":
      print(f"Calling the /v1/models.")

      request = multi_lora_decoding_pb2.ListAdaptersRequest()

      response = stub.models(request)

      if response.success is True:
        print(f"`models` call responded successfully.")
        if response.adapter_infos:
          print(f"Here is the list of adapters loaded on server:")
        else:
          print(f"No adapters are loaded on the server.")

        for adapter_info in response.adapter_infos:
          print(
              f"adapter_id={adapter_info.adapter_id}, loading_cost={adapter_info.loading_cost}, size_hbm={adapter_info.size_hbm} bytes, size_cpu={adapter_info.size_cpu} Bytes, last_accessed={adapter_info.last_accessed}, status={adapter_info.status}"
          )
      else:
        print(f"`models` call failed with error={response.error_message}")

    elif _TEST_API_NAME.value == "completions":
      print(f"Calling the /v1/completions.")

      request = jetstream_pb2.DecodeRequest(
          text_content=jetstream_pb2.DecodeRequest.TextContent(
              text=_TEXT.value,
          ),
          max_tokens=_MAX_TOKENS.value,
          lora_adapter_id=_LORA_ADAPTER_ID.value,
      )
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)

      response = stub.Decode(request)

      output = []
      for resp in response:
        output.extend(resp.stream_content.samples[0].text)

        text_output = "".join(output)

      print(f"Prompt: {_TEXT.value}")
      print(f"Response: {text_output}")

    elif _TEST_API_NAME.value == None:
      print(f"`test_api_name` flag is not set. So exiting.")
      return

    else:
      print(f"API={_TEST_API_NAME.value} is not implemented yet. So exiting.")
      return


if __name__ == "__main__":
  app.run(main)
