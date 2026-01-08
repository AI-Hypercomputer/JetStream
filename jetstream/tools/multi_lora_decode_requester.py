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

"""Decoding multiple LoRA requests via JetStream online serving.
"""


import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import time
from typing import Any, AsyncGenerator, Optional
import os


import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine.token_utils import load_vocab
from jetstream.external_tokenizers.llama3 import llama3_tokenizer
import numpy as np


@dataclass
class InputRequest:
  prompt: str = ""
  output: str = ""
  output_len: int = 0
  sample_idx: int = -1
  adapter_id: str = ""


@dataclass
class RequestFuncOutput:
  input_request: Optional[InputRequest] = None
  generated_token_list: list[str] = field(default_factory=list)
  generated_text: str = ""
  success: bool = False
  latency: float = 0
  ttft: float = 0

  # Flatten the structure and return only the necessary results
  def to_dict(self):
    return {
        "prompt": self.input_request.prompt,
        "original_output": self.input_request.output,
        "generated_text": self.generated_text,
        "success": self.success,
        "latency": self.latency,
        "sample_idx": self.input_request.sample_idx,
    }


def get_tokenizer(
    model_id: str,
    tokenizer_name: str,
) -> Any:
  """Return a tokenizer or a tokenizer placeholder."""
  if tokenizer_name == "test":
    print("Using test tokenizer")
    return "test"
  elif model_id == "llama-3":
    # Llama 3 uses a tiktoken tokenizer.
    print(f"Using llama-3 tokenizer: {tokenizer_name}")
    return llama3_tokenizer.Tokenizer(tokenizer_name)
  else:
    # Use JetStream tokenizer util. It's using the sentencepiece wrapper in
    # seqio library.
    print(f"Using tokenizer: {tokenizer_name}")
    vocab = load_vocab(tokenizer_name)
    return vocab.tokenizer


async def grpc_async_request(
    api_url: str, request: Any
) -> tuple[list[str], float, float]:
  """Send grpc synchronous request since the current grpc server is sync."""
  options = [("grpc.keepalive_timeout_ms", 10000)]
  async with grpc.aio.insecure_channel(api_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print("Making request")
    ttft = 0
    token_list = []
    request_start_time = time.perf_counter()
    response = stub.Decode(request)
    async for resp in response:
      if ttft == 0:
        ttft = time.perf_counter() - request_start_time
      token_list.extend(resp.stream_content.samples[0].token_ids)
    latency = time.perf_counter() - request_start_time
    return token_list, ttft, latency


async def send_request(
    api_url: str,
    tokenizer: Any,
    input_request: InputRequest,
) -> RequestFuncOutput:
  """Send the request to JetStream server."""
  # Tokenization on client side following MLPerf standard.
  token_ids = tokenizer.encode(input_request.prompt)
  request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(
          token_ids=token_ids
      ),
      max_tokens=input_request.output_len,
      lora_adapter_id=input_request.adapter_id,
  )
  output = RequestFuncOutput()
  output.input_request = input_request
  generated_token_list, ttft, latency = await grpc_async_request(
      api_url, request
  )
  output.ttft = ttft
  output.latency = latency
  output.generated_token_list = generated_token_list
  # generated_token_list is a list of token ids, decode it to generated_text.
  output.generated_text = tokenizer.decode(generated_token_list)
  output.success = True
  return output


async def get_request(
    input_requests: list[InputRequest],
) -> AsyncGenerator[InputRequest, None]:
  input_requests = iter(input_requests)

  for request in input_requests:
    yield request


async def send_multi_request(
    api_url: str,
    tokenizer: Any,
    input_requests: list[InputRequest],
):
  """Send multiple LoRA adapter requests."""
  tasks = []
  async for request in get_request(input_requests):
    tasks.append(
        asyncio.create_task(
            send_request(
                api_url=api_url,
                tokenizer=tokenizer,
                input_request=request,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)

  return outputs


def mock_adapter_requests(total_mock_requests: int):
  """Generates a list of mock requests containing mock data."""
  data = []
  for index in range(total_mock_requests):
    request = InputRequest()
    request.prompt = f"22 year old"
    if index == 0:
      request.adapter_id = ""
    else:
      i = (index % 10) + 1
      request.adapter_id = f"test_lora_{i}"
    request.output_len = 200
    data.append(request)
  return data


def main(args: argparse.Namespace):
  print(args)

  model_id = args.model
  tokenizer_id = args.tokenizer

  api_url = f"{args.server}:{args.port}"

  tokenizer = get_tokenizer(model_id, tokenizer_id)
  input_requests = mock_adapter_requests(
      args.total_mock_requests
  )  # e.g. [("AB", 2, "AB", 3)]

  request_outputs = asyncio.run(
      send_multi_request(
          api_url=api_url,
          tokenizer=tokenizer,
          input_requests=input_requests,
      )
  )

  output = [output.to_dict() for output in request_outputs]

  # Process output
  for index, output in enumerate(output):
    print(f"Prompt: {input_requests[index].prompt}")
    print(f"AdapterId: {input_requests[index].adapter_id}")
    print(f"Output: {output}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Sending multiple serving requests to JetStream Server"
  )
  parser.add_argument(
      "--server",
      type=str,
      default="0.0.0.0",
      help="Server address.",
  )
  parser.add_argument("--port", type=str, default=9000)
  parser.add_argument(
      "--model",
      type=str,
      default="no_model",
      help=(
          "Name of the model like llama-2, llama-3, gemma. (it's just used to"
          " label the benchmark, pick the tokenizer, the model config is"
          " defined in config_lib, and passed as the server config flag when"
          " we run the JetStream server)"
      ),
  )
  parser.add_argument(
      "--total-mock-requests",
      type=int,
      default=3,
      help="The maximum number of mock requests to send for benchmark testing.",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      default="test",
      help=(
          "Name or path of the tokenizer. (For mock model testing, use the"
          " default value)"
      ),
  )

  parsed_args = parser.parse_args()
  main(parsed_args)
