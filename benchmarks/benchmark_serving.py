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

"""Benchmark JetStream online serving.

On the server side, run one of the following commands:
    * For real server, you need to pass correct server config (include the
      model config that being passed into your engine impl) to the command
      below. Refer to config_lib.py and implementations/mock/config.py for
      config impl detail.

    (run with real server)
    python -m jetstream.core.implementations.<your_impl>.server \
        --config <your_server_config>

    (run with mock server)
    python -m jetstream.core.implementations.mock.server

On the client side, run:
    * For real server and shareGPT dataset, you need to pass the tokenizer,
      server config, and dataset flags to the command below, and make some
      changes to the tokenizer logic in the benchmark script (get_tokenizer
      and sample_requests func) to use your tokenizer correctly.
    * Add `--save-result` flag to save the benchmark result to a json file in
      current folder.
    * You can also add `--run_eval true` if you want to calculate ROUGE score
      on the predicted outputs.

    (run with real model and engines)
    python -m benchmarks.benchmark_serving \
        --tokenizer <your_tokenizer> \
        --dataset <target_dataset_name> \
        --dataset-path <target_dataset_path> \
        --request-rate <request_rate>

    (run with mock)
    python -m benchmarks.benchmark_serving \
        --request-rate 1

e2e example:
python3 benchmark_serving.py \
    --tokenizer /home/{username}/maxtext/assets/tokenizer \
    --num-prompts 100 \
    --dataset sharegpt \
    --dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json

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
from jetstream.third_party.llama3 import llama3_tokenizer
import numpy as np
from tqdm.asyncio import tqdm  # pytype: disable=pyi-error
import pandas

from eval_accuracy import eval_accuracy


def str2bool(v: str) -> bool:
  """Convert a string of truth to True or False.

  Args:
    - v (str):
      - True values are 'y', 'yes', 't', 'true', and '1';
      - False values are 'n', 'no', 'f', 'false', and '0'.

  Returns:
    bool: True or False

  Raises:
    ValueError if v is anything else.
  """
  v = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v in true_values:
    return True
  elif v in false_values:
    return False
  else:
    raise ValueError(f"Invalid value '{v}'!")


@dataclass
class BenchmarkMetrics:
  """Data class to store benchmark metrics."""

  completed: int
  total_input: int
  total_output: int
  request_throughput: float
  input_throughput: float
  output_throughput: float
  mean_ttft_ms: float
  median_ttft_ms: float
  p99_ttft_ms: float
  mean_tpot_ms: float
  median_tpot_ms: float
  p99_tpot_ms: float


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_len: int = 0
  output: str = ""
  output_len: int = 0
  sample_idx: int = -1


@dataclass
class RequestFuncOutput:
  input_request: Optional[InputRequest] = None
  generated_token_list: list[str] = field(default_factory=list)
  generated_text: str = ""
  success: bool = False
  latency: float = 0
  ttft: float = 0
  prompt_len: int = 0

  # Flatten the structure and return only the necessary results
  def to_dict(self):
    return {
        "prompt": self.input_request.prompt,
        "original_output": self.input_request.output,
        "generated_text": self.generated_text,
        "success": self.success,
        "latency": self.latency,
        "prompt_len": self.prompt_len,
        "sample_idx": self.input_request.sample_idx,
    }


def get_tokenizer(model_id: str, tokenizer_name: str) -> Any:
  """Return a tokenizer or a tokenizer placholder."""
  if tokenizer_name == "test":
    return "test"
  elif model_id == "llama-3":
    # Llama 3 uses a tiktoken tokenizer.
    return llama3_tokenizer.Tokenizer(tokenizer_name)
  else:
    # Use JetStream tokenizer util. It's using the sentencepiece wrapper in
    # seqio library.
    vocab = load_vocab(tokenizer_name)
    return vocab.tokenizer


def load_sharegpt_dataset(
    dataset_path: str,
    conversation_starter: str,
) -> list[tuple[Any, Any]]:
  # Load the dataset.
  with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
  # Filter out the conversations with less than 2 turns.
  dataset = [data for data in dataset if len(data["conversations"]) >= 2]

  # Filter based on conversation starter
  if conversation_starter != "both":
    dataset = [
        data
        for data in dataset
        if data["conversations"][0]["from"] == conversation_starter
    ]
  # Only keep the first two turns of each conversation.
  dataset = [
      (data["conversations"][0]["value"], data["conversations"][1]["value"])
      for data in dataset
  ]

  return dataset


def load_openorca_dataset_pkl():
  # read pickle file
  samples = pandas.read_pickle(
      os.path.join(
          os.path.dirname(os.path.relpath(__file__)),
          "open_orca_gpt4_tokenized_llama.calibration_1000.pkl",
      )
  )

  prompts = []
  outputs = []
  for _, row in samples.iterrows():
    prompts.append(row["input"])
    outputs.append(row["output"])

  return [(prompt, output) for prompt, output in zip(prompts, outputs)]


def tokenize_dataset(
    dataset: list[tuple[Any, Any, Any]],
    tokenizer: Any,
) -> list[tuple[str, Any, str, int, int, int]]:

  n = len(dataset)

  prompts = []
  outputs = []
  indices = []
  prompt_token_ids = []
  outputs_token_ids = []
  for prompt, output, idx in dataset:
    prompts.append(prompt)
    outputs.append(output)
    indices.append(idx)
    prompt_token_ids.append(tokenizer.encode(prompt))
    outputs_token_ids.append(tokenizer.encode(output))

  tokenized_dataset = []
  for i in range(n):
    prompt_len = len(prompt_token_ids[i])
    output_len = len(outputs_token_ids[i])
    tokenized_data = (
        prompts[i],
        prompt_token_ids[i],
        outputs[i],
        prompt_len,
        output_len,
        indices[i],
    )
    tokenized_dataset.append(tokenized_data)
  return tokenized_dataset


def filter_dataset(
    tokenized_dataset: list[tuple[str, Any, str, int, int, int]],
    max_output_length: int = 0,
) -> list[InputRequest]:
  if max_output_length != 0:
    print("In InputRequest, pass in actual output_length for each sample")
  else:
    print(
        f"In InputRequest, pass in max_output_length: {max_output_length} for"
        " each sample"
    )

  # Filter out too long sequences.
  filtered_dataset: list[InputRequest] = []
  for (
      prompt,
      _,
      output,
      prompt_len,
      output_len,
      sample_idx,
  ) in tokenized_dataset:
    if prompt_len < 4 or output_len < 4:
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      continue
    if prompt_len > 1024 or prompt_len + output_len > 2048:
      # Prune too long sequences.
      continue
    request = InputRequest(
        prompt, prompt_len, output, max_output_length or output_len, sample_idx
    )
    filtered_dataset.append(request)

  print(f"The dataset contains {len(tokenized_dataset)} samples.")
  print(f"The filtered dataset contains {len(filtered_dataset)} samples.")

  return filtered_dataset


def sample_requests(
    dataset: list[tuple[Any, Any]],
    tokenizer: Any,
    num_requests: int,
    max_output_length: int = 0,
    oversample_multiplier: float = 1.2,
) -> list[InputRequest]:

  # Original dataset size
  n = len(dataset)
  dataset_indices = range(n)

  # Create necessary number of requests even if bigger than dataset size
  sampled_indices = random.sample(
      dataset_indices, min(int(num_requests * oversample_multiplier), n)
  )

  if num_requests > len(sampled_indices):
    print(
        f"Number of requests {num_requests} is larger than size of dataset"
        f" {n}.\n",
        "Repeating data to meet number of requests.\n",
    )
    sampled_indices = sampled_indices * int(
        np.ceil(num_requests / len(sampled_indices))
    )

  print(f"{len(sampled_indices)=}")
  # some of these will be filtered out, so sample more than we need

  sampled_dataset = []
  for i in sampled_indices:
    sampled_data = dataset[i] + (dataset_indices[i],)
    sampled_dataset.append(sampled_data)

  tokenized_dataset = tokenize_dataset(sampled_dataset, tokenizer)

  input_requests = filter_dataset(tokenized_dataset, max_output_length)

  # Sample the requests.
  if len(input_requests) > num_requests:
    input_requests = random.sample(input_requests, num_requests)

  return input_requests


async def get_request(
    input_requests: list[InputRequest],
    request_rate: float,
) -> AsyncGenerator[InputRequest, None]:
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request

    if request_rate == 0.0:
      # If the request rate is infinity, then we don't need to wait.
      continue
    # Sample the request interval from the exponential distribution.
    interval = np.random.exponential(1.0 / request_rate)
    # The next request will be sent after the interval.
    await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[InputRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: Any,
) -> BenchmarkMetrics:
  total_output = 0
  total_input = 0
  completed = 0
  per_token_latencies = []
  ttfts = []
  for i in range(len(outputs)):
    if outputs[i].success:
      output_len = len(
          outputs[i].generated_token_list
          if tokenizer != "test"
          else ["Ċ", "Ō", "Ɵ"]
      )
      total_output += output_len
      total_input += input_requests[i].prompt_len
      if output_len == 0:
        print(
            f"""-------- output_len is zero for {i}th request:,
             output: {outputs[i]}"""
        )
        continue
      per_token_latencies.append(outputs[i].latency / output_len)
      ttfts.append(outputs[i].ttft)
      completed += 1

  metrics = BenchmarkMetrics(
      completed=completed,
      total_input=total_input,
      total_output=total_output,
      request_throughput=completed / dur_s,
      input_throughput=total_input / dur_s,
      output_throughput=total_output / dur_s,
      mean_ttft_ms=float(np.mean(ttfts) * 1000),
      median_ttft_ms=float(np.median(ttfts) * 1000),
      p99_ttft_ms=float(np.percentile(ttfts, 99) * 1000),
      mean_tpot_ms=float(np.mean(per_token_latencies) * 1000),
      median_tpot_ms=float(np.median(per_token_latencies) * 1000),
      p99_tpot_ms=float(np.percentile(per_token_latencies, 99) * 1000),
  )

  return metrics


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
    pbar: tqdm,
    session_cache: str,
    priority: int,
) -> RequestFuncOutput:
  """Send the request to JetStream server."""
  # Tokenization on client side following MLPerf standard.
  token_ids = tokenizer.encode(input_request.prompt)
  request = jetstream_pb2.DecodeRequest(
      session_cache=session_cache,
      token_content=jetstream_pb2.DecodeRequest.TokenContent(
          token_ids=token_ids
      ),
      priority=priority,
      max_tokens=input_request.output_len,
  )
  output = RequestFuncOutput()
  output.input_request = input_request
  output.prompt_len = input_request.prompt_len
  generated_token_list, ttft, latency = await grpc_async_request(
      api_url, request
  )
  output.ttft = ttft
  output.latency = latency
  output.generated_token_list = generated_token_list
  # generated_token_list is a list of token ids, decode it to generated_text.
  output.generated_text = tokenizer.decode(generated_token_list)
  output.success = True
  if pbar:
    pbar.update(1)
  return output


async def benchmark(
    api_url: str,
    tokenizer: Any,
    input_requests: list[InputRequest],
    request_rate: float,
    disable_tqdm: bool,
    session_cache: str,
    priority: int,
):
  """Benchmark the online serving performance."""
  pbar = None if disable_tqdm else tqdm(total=len(input_requests))

  print(f"Traffic request rate: {request_rate}")

  benchmark_start_time = time.perf_counter()
  tasks = []
  async for request in get_request(input_requests, request_rate):
    tasks.append(
        asyncio.create_task(
            send_request(
                api_url=api_url,
                tokenizer=tokenizer,
                input_request=request,
                pbar=pbar,
                session_cache=session_cache,
                priority=priority,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)

  if not disable_tqdm and pbar:
    pbar.close()

  benchmark_duration = time.perf_counter() - benchmark_start_time

  metrics = calculate_metrics(
      input_requests=input_requests,
      outputs=outputs,
      dur_s=benchmark_duration,
      tokenizer=tokenizer,
  )

  print(f"Successful requests: {metrics.completed}")
  print(f"Benchmark duration: {benchmark_duration:2f} s")
  print(f"Total input tokens: {metrics.total_input}")
  print(f"Total generated tokens: {metrics.total_output}")
  print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
  print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
  print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
  print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
  print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
  print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
  print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
  print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
  print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

  result = {
      "duration": benchmark_duration,
      "completed": metrics.completed,
      "total_input_tokens": metrics.total_input,
      "total_output_tokens": metrics.total_output,
      "request_throughput": metrics.request_throughput,
      "input_throughput": metrics.input_throughput,
      "output_throughput": metrics.output_throughput,
      "mean_ttft_ms": metrics.mean_ttft_ms,
      "median_ttft_ms": metrics.median_ttft_ms,
      "p99_ttft_ms": metrics.p99_ttft_ms,
      "mean_tpot_ms": metrics.mean_tpot_ms,
      "median_tpot_ms": metrics.median_tpot_ms,
      "p99_tpot_ms": metrics.p99_tpot_ms,
  }
  return result, outputs


def mock_requests(total_mock_requests: int):
  """Generates a list of mock requests containing mock data."""
  data = []
  for _ in range(total_mock_requests):
    reqeust = InputRequest()
    reqeust.prompt = f"Prompt {random.randint(1, 1000)}"
    reqeust.prompt_len = random.randint(10, 100)
    reqeust.out = f"Output {random.randint(1, 1000)}"
    reqeust.output_len = random.randint(1, 10)
    data.append(reqeust)
  return data


def sample_warmup_requests(requests):
  interesting_buckets = [
      0,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
  ]

  for start, end in zip(interesting_buckets[:-1], interesting_buckets[1:]):
    for request in requests:
      if start < request.prompt_len <= end:
        yield request
        break


def main(args: argparse.Namespace):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)

  model_id = args.model
  tokenizer_id = args.tokenizer

  api_url = f"{args.server}:{args.port}"

  tokenizer = get_tokenizer(model_id, tokenizer_id)
  if tokenizer == "test" or args.dataset == "test":
    input_requests = mock_requests(
        args.total_mock_requests
    )  # e.g. [("AB", 2, "AB", 3)]
  else:
    dataset = []
    if args.dataset == "openorca":
      dataset = load_openorca_dataset_pkl()
    elif args.dataset == "sharegpt":
      dataset = load_sharegpt_dataset(
          args.dataset_path,
          args.conversation_starter,
      )

    # A given args.max_output_length value is the max generation step,
    # when the args.max_output_length is default to None, the sample's golden
    # output length will be used to decide the generation step.
    input_requests = sample_requests(
        dataset=dataset,
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        max_output_length=args.max_output_length,
    )

  warmup_requests = None
  if args.warmup_mode == "full":
    warmup_requests = input_requests
  elif args.warmup_mode == "sampled":
    warmup_requests = list(sample_warmup_requests(input_requests)) * 2

  if warmup_requests:
    print(f"Starting {args.warmup_mode} warmup:")
    benchmark_result, request_outputs = asyncio.run(
        benchmark(
            api_url=api_url,
            tokenizer=tokenizer,
            input_requests=warmup_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            session_cache=args.session_cache,
            priority=args.priority,
        )
    )
    print(f"{args.warmup_mode} warmup completed.")

  # TODO: Replace this with warmup complete signal once supported.
  # Wait for server completely warmup before running the benchmark.
  time.sleep(5)

  benchmark_result, request_outputs = asyncio.run(
      benchmark(
          api_url=api_url,
          tokenizer=tokenizer,
          input_requests=input_requests,
          request_rate=args.request_rate,
          disable_tqdm=args.disable_tqdm,
          session_cache=args.session_cache,
          priority=args.priority,
      )
  )

  # Process output
  output = [output.to_dict() for output in request_outputs]
  if args.run_eval:
    eval_json = eval_accuracy(output)

  # Save config and results to json
  if args.save_result:
    # dimensions values are strings
    dimensions_json = {}
    # metrics values are numerical
    metrics_json = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    dimensions_json["date"] = current_dt
    dimensions_json["model_id"] = model_id
    dimensions_json["tokenizer_id"] = tokenizer_id
    dimensions_json = {
        **dimensions_json,
        **json.loads(args.additional_metadata_metrics_to_save),
    }
    metrics_json["num_prompts"] = args.num_prompts

    # Traffic
    metrics_json["request_rate"] = args.request_rate
    metrics_json = {**metrics_json, **benchmark_result}
    if args.run_eval:
      metrics_json = {**metrics_json, **eval_json}

    final_json = {}
    final_json["metrics"] = metrics_json
    final_json["dimensions"] = dimensions_json

    # Save to file
    base_model_id = model_id.split("/")[-1]
    file_name = (
        f"JetStream-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
    )
    with open(file_name, "w", encoding="utf-8") as outfile:
      json.dump(final_json, outfile)

  if args.save_request_outputs:
    file_path = args.request_outputs_file_path
    with open(file_path, "w", encoding="utf-8") as output_file:
      json.dump(
          output,
          output_file,
          indent=4,
      )


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--server",
      type=str,
      default="0.0.0.0",
      help="Server address.",
  )
  parser.add_argument("--port", type=str, default=9000)
  parser.add_argument(
      "--dataset",
      type=str,
      default="test",
      choices=["test", "sharegpt", "openorca"],
      help="The dataset name.",
  )
  parser.add_argument("--dataset-path", type=str, help="Path to the dataset.")
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
      "--tokenizer",
      type=str,
      default="test",
      help=(
          "Name or path of the tokenizer. (For mock model testing, use the"
          " default value)"
      ),
  )
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=1000,
      help=(
          "Number of prompts to process. (number of sample requests we randomly"
          " collect from dataset)"
      ),
  )
  parser.add_argument(
      "--request-rate",
      type=float,
      default=0.0,
      help=(
          "Number of requests per second. If this is 0., "
          "then all the requests are sent at time 0. "
          "Otherwise, we use Poisson process to synthesize "
          "the request arrival times."
      ),
  )
  parser.add_argument(
      "--total-mock-requests",
      type=int,
      default=150,
      help="The maximum number of mock requests to send for benchmark testing.",
  )

  parser.add_argument(
      "--max-output-length",
      type=int,
      default=0,
      help=(
          "The maximum output length for reference request. It would be passed"
          " to `max_tokens` parameter of the JetStream's DecodeRequest proto,"
          " and used in JetStream to control the output/decode length of a"
          " sequence. It would not be used in the engine. We should always set"
          " max_tokens <= (max_target_length - max_prefill_predict_length)."
          " max_target_length is the maximum length of a sequence;"
          " max_prefill_predict_length is the maximum length of the"
          " input/prefill of a sequence. Default to 0, in this case, "
          "the output length of the golden dataset would be passed."
      ),
  )

  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--disable-tqdm",
      action="store_true",
      help="Specify to disable tqdm progress bar.",
  )
  parser.add_argument(
      "--save-result",
      action="store_true",
      help="Specify to save benchmark results to a json file",
  )
  parser.add_argument(
      "--additional-metadata-metrics-to-save",
      type=str,
      help=(
          "Additional metadata about the workload. Should be a dictionary in"
          " the form of a string."
      ),
  )
  parser.add_argument(
      "--priority",
      type=int,
      default=0,
      help=(
          "Message priority. (currently no business logic implemented, use"
          " default 0)"
      ),
  )
  parser.add_argument(
      "--session-cache",
      type=str,
      default="",
      help=(
          "Location of any pre-cached results. (currently _load_cache_history"
          " not implemented, use default empty str)"
      ),
  )
  parser.add_argument(
      "--save-request-outputs",
      action="store_true",
      help="Specify to store request outputs into a json file",
  )
  parser.add_argument(
      "--request-outputs-file-path",
      type=str,
      default="/tmp/request-outputs.json",
      help="File path to store request outputs",
  )
  parser.add_argument(
      "--run-eval",
      type=str2bool,
      default=False,
      help="Whether to run evaluation script on the saved outputs",
  )
  parser.add_argument(
      "--warmup-mode",
      type=str,
      default="none",
      choices=["none", "sampled", "full"],
      help="Whether to warmup first, and set the warmup mode",
  )
  parser.add_argument(
      "--conversation-starter",
      type=str,
      default="human",
      choices=["human", "gpt", "both"],
      help="What entity should be the one starting the conversations.",
  )

  parsed_args = parser.parse_args()
  main(parsed_args)
