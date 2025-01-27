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
#_TEXT = flags.DEFINE_string("text", "My dog is cute", "The message")
_TEXT = flags.DEFINE_string("text", "22 year old", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 3, "Maximum number of output/decode tokens of a sequence"
)

_ADAPTER_ID = flags.DEFINE_string(
    "adapter_id",
    None,
    "Id of the fine-tuned adapter to be loaded on top of the base model.",
    required=False,
)

_ADAPTER_CONFIG_PATH = flags.DEFINE_string(
    "adapter_config_path",
    None,
    "Path of the fine-tuned adapter to be loaded from.",
    required=False,
)

_ADAPTER_WEIGHTS_PATH = flags.DEFINE_string(
    "adapter_weights_path",
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
  del argv
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  address = f"{_SERVER.value}:{_PORT.value}"
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.MultiAdapterManagerStub(channel)
    print(f"Sending request to: {address}")

    if _TEST_API_NAME.value == "load_adapter":
      print(f"Calling the JetStream/MultiAdapterManager/LoadAdapter.")

      adapter_id=_ADAPTER_ID.value
      adapter_config_path=_ADAPTER_CONFIG_PATH.value
      adapter_weights_path=_ADAPTER_WEIGHTS_PATH.value

      if adapter_id == None or adapter_weights_path == None or adapter_config_path == None:
        print(f"For `load_adapter` API call, `adapter_id`, `adapter_config_path` and `adapter_weights_path` must be passed.")
        return

      request = jetstream_pb2.LoadAdapterRequest(
            adapter_id=adapter_id,
            adapter_config_path=adapter_config_path,
            adapter_weights_path=adapter_weights_path
      )

      response = stub.LoadAdapter(request)

      if response.success is True:
        print(f"Adapter={adapter_id} is loaded successfully.")
      else:
        print(f"Adapter={adapter_id} loading failed with error={response.error_message}")
    
    elif _TEST_API_NAME.value == "unload_adapter":
      print(f"Calling the JetStream/MultiAdapterManager/UnloadAdapter.")

      adapter_id=_ADAPTER_ID.value

      if adapter_id == None:
        print(f"For `unload_adapter` API call, `adapter_id` must be passed.")
        return

      request = jetstream_pb2.UnloadAdapterRequest(
            adapter_id=adapter_id,
      )

      response = stub.UnloadAdapter(request)

      if response.success is True:
        print(f"Adapter={adapter_id} is unloaded successfully.")
      else:
        print(f"Adapter={adapter_id} unloading failed with error={response.error_message}")
    
    elif _TEST_API_NAME.value == "list_adapters":
      print(f"Calling the JetStream/MultiAdapterManager/ListAdapters.")

      request = jetstream_pb2.ListAdaptersRequest()

      response = stub.ListAdapters(request)

      if response.success is True:
        print(f"`ListAdapter` call responded successfully. Here is the list of adapters loaded on server:")
        for adapter_info in response.adapter_infos:
          print(f"adapter_id={adapter_info.adapter_id}, loading_cost={adapter_info.loading_cost}.")
      else:
        print(f"`ListAdapter` call failed with error={error_message}")
    
    elif _TEST_API_NAME.value == None:
      print(f"`test_api_name` flag is not set. So exiting.")
      return

    else:
      print(f"API={_TEST_API_NAME.value} is not implemented yet. So exiting.")
      return


    print(f"API calls ended.")


if __name__ == "__main__":
  app.run(main)
