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
import gc
import json
import os
import random
import time
from typing import Any, AsyncGenerator, Optional

from benchmarks.eval_accuracy import eval_accuracy
from benchmarks.eval_accuracy_mmlu import eval_accuracy_mmlu
from benchmarks.metrics import CounterMetric, EventMetric
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine.token_utils import load_vocab
from jetstream.external_tokenizers.llama3 import llama3_tokenizer
import numpy as np
import pandas
from tqdm.asyncio import tqdm  # pytype: disable=pyi-error
from transformers import AutoTokenizer


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


class AsyncCounter:
  """An counter class for counting and quota management with asycio,
  not thread safe. It's safe with asyncio as value changes are done
  outside of await statements.
  """

  def __init__(self, init_value: int, block_on_zero_seconds=0.002):
    """
    Args:
      init_value: Initial value for the counter.
      block_on_zero_seconds: if greater than 0, the counter will spin when
        value hits 0, hence can be used for quota management.
    """
    self._init_value = init_value
    self._value = init_value
    self._block_on_zero_seconds = block_on_zero_seconds

  async def inc(self):
    self._value += 1

  async def dec(self):
    while True:
      if self._value > 0 or self._block_on_zero_seconds <= 0.0:
        self._value -= 1
        return
      await asyncio.sleep(self._block_on_zero_seconds)

  def value(self):
    return self._value

  def delta(self):
    return self._init_value - self._value


@dataclass
class BenchmarkMetrics:
  """Data class to store benchmark metrics."""

  completed: int
  total_input: int
  total_output: int
  request_throughput: float
  input_throughput: float
  output_throughput: float
  overall_throughput: float

  ttft: EventMetric  # Time-to-first-token
  ttst: EventMetric  # Time-to-second-token
  tpot: EventMetric  # Time-per-output-token


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_len: int = 0
  output: str = ""
  output_len: int = 0
  sample_idx: int = -1


@dataclass
class RequestFuncOutput:
  """Data class to store the response of a request."""

  input_request: Optional[InputRequest] = None
  generated_token_list: list[int] = field(default_factory=list)
  generated_text: str = ""
  success: bool = False
  latency_sec: float = 0
  ttft_sec: float = 0
  ttst_sec: float = 0
  prompt_len: int = 0

  # Flatten the structure and return only the necessary results
  def to_dict(self):
    if self.input_request:
      prompt = self.input_request.prompt
      original_output = self.input_request.output
      sample_idx = self.input_request.sample_idx
    else:
      prompt = None
      original_output = None
      sample_idx = None
    return {
        "prompt": prompt,
        "original_output": original_output,
        "generated_text": self.generated_text,
        "success": self.success,
        "latency_sec": self.latency_sec,
        "ttft_sec": self.ttft_sec,
        "ttst_sec": self.ttst_sec,
        "prompt_len": self.prompt_len,
        "sample_idx": sample_idx,
    }


def get_tokenizer(
    model_id: str,
    tokenizer_name: str,
    use_hf_tokenizer: bool,
) -> Any:
  """Return a tokenizer or a tokenizer placholder."""
  if tokenizer_name == "test":
    print("Using test tokenizer")
    return "test"
  elif use_hf_tokenizer:
    # Please accept agreement to access private/gated models in HF, and
    # follow up instructions below to set up access token
    # https://huggingface.co/docs/transformers.js/en/guides/private
    print(f"Using HuggingFace tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name)
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


def load_openorca_dataset_pkl(
    dataset_path: str,
) -> list[tuple[Any, Any]]:
  if not dataset_path:
    dataset_path = "open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
  # read pickle file
  samples = pandas.read_pickle(
      os.path.join(
          os.path.dirname(os.path.relpath(__file__)),
          dataset_path,
      )
  )

  prompts = []
  outputs = []
  for _, row in samples.iterrows():
    prompts.append(row["input"])
    outputs.append(row["output"])

  return [(prompt, output) for prompt, output in zip(prompts, outputs)]


def load_longcontext_dataset_pkl(
    dataset_path: str,
) -> list[tuple[Any, Any]]:
  assert os.path.isfile(dataset_path)

  # read pickle file
  data = pandas.read_pickle(dataset_path)

  samples = []
  for _, row in data.iterrows():
    samples.append((row["input"], row["ref_output"]))

  return samples


def load_mmlu_dataset_csv(dataset_path: str) -> tuple[Any, dict[str, str]]:
  assert dataset_path != ""
  dataset = []
  prompts_per_subject = dict()
  for cvs_file in os.listdir(dataset_path):
    if cvs_file.endswith(".csv"):
      subject = " ".join(cvs_file.split("_")[:-1])
      if subject not in prompts_per_subject:
        prompts_per_subject[subject] = ""
      filepath = os.path.join(dataset_path, cvs_file)
      data = pandas.read_csv(filepath, header=None)
      data["subject"] = subject
      dataset.append(data)

  combined_dataset = pandas.concat(dataset, ignore_index=True)
  header_dict = {
      0: "question",
      1: "A",
      2: "B",
      3: "C",
      4: "D",
      5: "answer",
  }
  combined_dataset.rename(columns=header_dict, inplace=True)
  return combined_dataset, prompts_per_subject


def gen_mmlu_qa(data: Any, mmlu_method: str = "") -> str:

  output = ""
  for _, row in data.iterrows():
    output += (
        f"Question: {row['question']}\n"
        f"Choices:\n"
        f"(A) {row['A']}\n"
        f"(B) {row['B']}\n"
        f"(C) {row['C']}\n"
        f"(D) {row['D']}\n"
    )

    output += "\nCorrect answer: "

    if mmlu_method == "HELM":
      output += f"({row['answer']})\n\n"
    elif mmlu_method == "Harness":
      content = row[row["answer"].upper()]
      output += f"({row['answer']}) {content}\n\n"

  return output


def gen_mmlu_prompt(
    dataset_path: str, num_shots: int = 1, mmlu_method: str = "HELM"
) -> list[tuple[Any, Any]]:
  combined_dataset, prompts_per_subject = load_mmlu_dataset_csv(dataset_path)
  num_rows, _ = combined_dataset.shape
  print(f"Loaded {num_rows} data from mmlu dataset")

  for subject in prompts_per_subject:
    header = (
        f"The following are multiple choice questions (with answers) "
        f"about {subject}:\n"
    )
    shots_data = combined_dataset[combined_dataset["subject"] == subject].head(
        num_shots
    )
    prompts_per_subject[subject] = header + gen_mmlu_qa(
        shots_data, mmlu_method=mmlu_method
    )

  mmlu_data = []
  for _, row in combined_dataset.iloc[num_shots:].iterrows():
    question_prompt = gen_mmlu_qa(pandas.DataFrame([row]))
    output = row["answer"]
    prompt = prompts_per_subject[row["subject"]] + question_prompt
    mmlu_data.append((prompt, output))

  return mmlu_data


def load_math500_dataset(dataset_path: str) -> list[tuple[Any, Any]]:
  if not dataset_path:
    dataset_path = "benchmarks/huggingfaceh4_math500.json"
  abs_path = os.path.abspath(dataset_path)
  with open(abs_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
  return [
      (data["problem"], (data["solution"], data["answer"])) for data in dataset
  ]


def tokenize_dataset(
    dataset: list[tuple[Any, Any, Any]],
    tokenizer: Any,
) -> list[tuple[str, Any, str, int, int, int]]:
  tokenized_dataset = []

  for prompt, output, idx in dataset:
    if isinstance(output, tuple):
      output_len = len(tokenizer.encode(output[0]))
      output_tokens = output[1]
    else:
      output_len = len(tokenizer.encode(output))
      output_tokens = output

    prompt_tokens = tokenizer.encode(prompt)

    tokenized_data = (
        prompt,
        prompt_tokens,
        output_tokens,
        len(prompt_tokens),
        output_len,
        idx,
    )
    tokenized_dataset.append(tokenized_data)
  return tokenized_dataset


def filter_dataset(
    tokenized_dataset: list[tuple[str, Any, str, int, int, int]],
    dataset_type: str,
    max_output_length: int = 0,
    run_mmlu_dataset: bool = False,
    min_input_length: int = 4,
    max_input_length: int = 0,
    max_target_length: int = 0,
    max_output_multiplier: int = 0,
) -> list[InputRequest]:
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
    if prompt_len < min_input_length or (
        not (run_mmlu_dataset or dataset_type == "math500") and output_len < 4
    ):
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      # Math results could be really short though.
      continue
    if (
        prompt_len > max_input_length
        or prompt_len + output_len > max_target_length
    ):
      # Prune too long sequences.
      continue
    if dataset_type == "math500":
      max_output_len = max_output_length or output_len * max_output_multiplier
    else:
      max_output_len = max_output_length or output_len
    request = InputRequest(
        prompt, prompt_len, output, max_output_len, sample_idx
    )
    filtered_dataset.append(request)

  print(f"The dataset contains {len(tokenized_dataset)} samples.")
  print(f"The filtered dataset contains {len(filtered_dataset)} samples.")

  return filtered_dataset


def sample_requests(
    dataset: list[tuple[Any, Any]],
    tokenizer: Any,
    num_requests: int,
    dataset_type: str,
    max_output_length: int = 0,
    oversample_multiplier: float = 1.2,
    run_mmlu_dataset: bool = False,
    min_input_length: int = 4,
    max_input_length: int = 0,
    max_target_length: int = 0,
    max_output_multiplier: int = 0,
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

  input_requests = filter_dataset(
      tokenized_dataset,
      dataset_type,
      max_output_length,
      run_mmlu_dataset,
      min_input_length,
      max_input_length,
      max_target_length,
      max_output_multiplier,
  )

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
  ttft = EventMetric("ttft", "Time-to-first-token", "ms")
  ttst = EventMetric("ttst", "Time-to-second-token", "ms")
  per_out_token_lat = EventMetric("TPOT", "Time-per-output-token", "ms")
  output_sizes = []
  for i in range(len(outputs)):
    if outputs[i].success:
      completed += 1
      output_len = len(
          outputs[i].generated_token_list
          if tokenizer != "test"
          else ["Ċ", "Ō", "Ɵ"]
      )
      output_sizes.append(output_len)
      total_output += output_len
      total_input += input_requests[i].prompt_len
      if output_len == 0:
        print(
            f"""-------- output_len is zero for {i}th request:,
             output: {outputs[i]}"""
        )
        continue
      ttft.record(outputs[i].ttft_sec * 1000)
      ttst.record(outputs[i].ttst_sec * 1000)
      per_out_token_lat.record(outputs[i].latency_sec / output_len * 1000)

  print("Mean output size:", float(np.mean(output_sizes)))
  print("Median output size:", float(np.median(output_sizes)))
  print("P99 output size:", float(np.percentile(output_sizes, 99)))

  metrics = BenchmarkMetrics(
      completed=completed,
      total_input=total_input,
      total_output=total_output,
      request_throughput=completed / dur_s,
      input_throughput=total_input / dur_s,
      output_throughput=total_output / dur_s,
      overall_throughput=(total_input + total_output) / dur_s,
      ttft=ttft,
      ttst=ttst,
      tpot=per_out_token_lat,
  )

  return metrics


async def grpc_async_request(
    api_url: str,
    request: Any,
    prefill_quota: AsyncCounter,
    active_req_quota: AsyncCounter,
    out_token_cnt: CounterMetric,
) -> tuple[list[int], float, float, float]:
  """Send grpc synchronous request since the current grpc server is sync."""
  options = [("grpc.keepalive_timeout_ms", 10000)]
  async with grpc.aio.insecure_channel(api_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    request_start_time = time.perf_counter()
    response = stub.Decode(request)
    token_list = []
    ttft = 0
    ttst = 0
    stream_resp_cnt = 0
    async for resp in response:
      stream_resp_cnt += 1
      if stream_resp_cnt == 1:
        await prefill_quota.inc()
        ttft = time.perf_counter() - request_start_time
        if ttft > 2.0:
          print(datetime.now(), f"slow TTFT {ttft:.2f}", prefill_quota.value())
      elif stream_resp_cnt == 2:
        ttst = time.perf_counter() - request_start_time
      resp_tokens = resp.stream_content.samples[0].token_ids
      token_list.extend(resp_tokens)
      out_token_cnt.increment(len(resp_tokens))
    await active_req_quota.inc()
    req_latency = time.perf_counter() - request_start_time
    return token_list, ttft, ttst, req_latency


async def send_request(
    api_url: str,
    tokenizer: Any,
    input_request: InputRequest,
    prefill_quota: AsyncCounter,
    active_req_quota: AsyncCounter,
    req_complete_cnt: CounterMetric,
    out_token_cnt: CounterMetric,
    pbar: tqdm,
) -> RequestFuncOutput:
  """Send the request to JetStream server."""
  # Tokenize on client side following MLPerf standard.
  token_ids = tokenizer.encode(input_request.prompt)

  # Send the request
  request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(
          token_ids=token_ids
      ),
      max_tokens=input_request.output_len,
      metadata=jetstream_pb2.DecodeRequest.Metadata(
          start_time=time.perf_counter()
      ),
  )
  out_tokens, ttft_sec, ttst_sec, latency_sec = await grpc_async_request(
      api_url,
      request,
      prefill_quota,
      active_req_quota,
      out_token_cnt,
  )
  req_complete_cnt.increment()

  # Collect per-request output and metrics.
  output = RequestFuncOutput()
  output.input_request = input_request
  output.prompt_len = input_request.prompt_len
  output.ttft_sec = ttft_sec
  output.ttst_sec = ttst_sec
  output.latency_sec = latency_sec
  output.generated_token_list = out_tokens
  # generated_token_list is a list of token ids, decode it to generated_text.
  output.generated_text = tokenizer.decode(out_tokens)
  output.success = True
  if pbar:
    pbar.postfix = (
        f"#reqs: {active_req_quota.delta()}/"
        f"{active_req_quota.value()}; "
        f"#prefill: {prefill_quota.delta()}/"
        f"{prefill_quota.value()}"
    )
    pbar.update(1)
  return output


async def benchmark(
    api_url: str,
    tokenizer: Any,
    input_requests: list[InputRequest],
    request_rate: float,
    disable_tqdm: bool,
    prefill_quota: AsyncCounter,
    active_req_quota: AsyncCounter,
    is_warmup: bool = False,
) -> tuple[dict[str, float | int], list[RequestFuncOutput]]:
  """Benchmark the online serving performance.

  Args:
    api_url: URL (e.g. host:port) of the JetStream server to send requests to.
    tokenizer: The tokenizer used to convert texts into tokens that will be set
      in requests.
    input_requests: A list of requests to send.
    request_rate: The number of requests to send per second.
    disable_tqdm: Whether progress bar should be disabled or not.
    prefill_quota: Quota for limiting pending prefill operations.
    active_req_quota: Quota for limiting inflight requests.
    is_warmup: Whether this run is to warm up the server.

  Return:
    A tuple containing the performance statistics for all requests and a list
    of responses from the executed requests.
  """
  print(f"Benchmarking with a total number of {len(input_requests)} requests")
  print(f"Benchmarking with request rate of {request_rate}")
  pbar = None if disable_tqdm else tqdm(total=len(input_requests))
  req_complete_cnt = CounterMetric(
      "ReqCompleteCount", "Request Completion Counter"
  )
  out_token_cnt = CounterMetric("OutTokenCount", "OutToken Counter")

  # Run benchmarking
  tasks = []
  benchmark_start_time = time.perf_counter()
  async for request in get_request(input_requests, request_rate):
    await prefill_quota.dec()
    await active_req_quota.dec()
    tasks.append(
        asyncio.create_task(
            send_request(
                api_url=api_url,
                tokenizer=tokenizer,
                input_request=request,
                prefill_quota=prefill_quota,
                active_req_quota=active_req_quota,
                req_complete_cnt=req_complete_cnt,
                out_token_cnt=out_token_cnt,
                pbar=pbar,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)
  if pbar is not None:
    pbar.close()

  # Compute metrics
  output_metrics = {}
  if not is_warmup:
    # No need to calculate metrics when executing warmup requests
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
    print(
        f"Overall token throughput: {metrics.overall_throughput:.2f} tokens/s"
    )

    print(f"{metrics.ttft.distribution_summary_str()}")
    print(f"{metrics.ttst.distribution_summary_str()}")
    print(f"{metrics.tpot.distribution_summary_str()}")

    # Calculate one rate for each 10 sec window. Adjusts the window size if
    # needed to use csv output below for plotting the rate over time.
    window_size_sec = 10
    print(
        f"----- Request complete rate time series "
        f"(window_size = {window_size_sec} sec) -----"
    )
    print(f"{req_complete_cnt.rate_over_window_to_csv(window_size_sec)}")
    print(
        f"----- Output token rate time series "
        f"(window_size = {window_size_sec} sec) -----"
    )
    print(f"{out_token_cnt.rate_over_window_to_csv(window_size_sec)}")

    output_metrics = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
    }
    output_metrics = {
        **output_metrics,
        **metrics.ttft.distribution_summary_dict(),
        **metrics.ttst.distribution_summary_dict(),
        **metrics.tpot.distribution_summary_dict(),
    }
  return output_metrics, outputs


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


def parse_args() -> argparse.Namespace:
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
      choices=[
          "test",
          "sharegpt",
          "openorca",
          "mmlu",
          "math500",
          "longcontext",
      ],
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
      "--use-hf-tokenizer",
      type=str2bool,
      default=False,
      help=(
          "Whether to use tokenizer from HuggingFace. If so, set this flag"
          " to True, and provide name of the tokenizer in the tokenizer flag."
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
  parser.add_argument(
      "--max-target-length",
      type=int,
      default=2048,
      help=(
          "The maximum prompt length plus the output length for reference "
          " request. It would be used to filter the requests "
      ),
  )
  parser.add_argument(
      "--max-output-multiplier",
      type=int,
      default=2,
      help=(
          "The multiplier applied to the reference output length. The generated"
          " output might be longer than the reference outputs. We apply this "
          " multiplier for finer grained control. "
      ),
  )
  parser.add_argument(
      "--min-input-length",
      type=int,
      default=4,
      help=(
          "The minium input length for reference request. It would be used"
          " to filter the input requests. "
      ),
  )
  parser.add_argument(
      "--max-input-length",
      type=int,
      default=1024,
      help=(
          "The maximum input length for reference request. It would be used"
          " to filter the input requests. "
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
  parser.add_argument(
      "--num-shots",
      type=int,
      default=1,
      help="Num shorts to give for mmlu data set",
  )
  parser.add_argument(
      "--mmlu-method",
      type=str,
      default="HELM",
      choices=["HELM", "Harness", ""],
      help="mmlu method/format to generate shots",
  )
  parser.add_argument(
      "--run-mmlu-dataset",
      action="store_true",
      help="specify if it's for mmlu dataset",
  )
  return parser.parse_args()


def main(args: argparse.Namespace):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)

  model_id = args.model
  tokenizer_id = args.tokenizer
  use_hf_tokenizer = args.use_hf_tokenizer

  prefill_quota = AsyncCounter(init_value=3)
  active_req_quota = AsyncCounter(init_value=450)

  api_url = f"{args.server}:{args.port}"

  tokenizer = get_tokenizer(model_id, tokenizer_id, use_hf_tokenizer)
  if tokenizer == "test" or args.dataset == "test":
    input_requests = mock_requests(
        args.total_mock_requests
    )  # e.g. [("AB", 2, "AB", 3)]
  else:
    dataset = []
    if args.dataset == "openorca":
      dataset = load_openorca_dataset_pkl(args.dataset_path)
    elif args.dataset == "sharegpt":
      dataset = load_sharegpt_dataset(
          args.dataset_path,
          args.conversation_starter,
      )
    elif args.dataset == "mmlu":
      dataset = gen_mmlu_prompt(
          args.dataset_path, args.num_shots, args.mmlu_method
      )
    elif args.dataset == "math500":
      dataset = load_math500_dataset(
          args.dataset_path,
      )
    elif args.dataset == "longcontext":
      dataset = load_longcontext_dataset_pkl(
          args.dataset_path,
      )
    else:
      raise ValueError(
          f"Fatal Error: Uncognized input parameters: {args.dataset}"
      )

    # A given args.max_output_length value is the max generation step,
    # when the args.max_output_length is default to None, the sample's golden
    # output length will be used to decide the generation step.
    input_requests = sample_requests(
        dataset=dataset,
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        dataset_type=args.dataset,
        max_output_length=args.max_output_length,
        run_mmlu_dataset=args.run_mmlu_dataset,
        min_input_length=args.min_input_length,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_output_multiplier=args.max_output_multiplier,
    )

  warmup_requests = None
  if args.warmup_mode == "full":
    warmup_requests = input_requests
  elif args.warmup_mode == "sampled":
    warmup_requests = list(sample_warmup_requests(input_requests)) * 2

  if warmup_requests:
    print(f"Warmup (mode: {args.warmup_mode}) is starting.")
    _, _ = asyncio.run(
        benchmark(
            api_url=api_url,
            tokenizer=tokenizer,
            input_requests=warmup_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            prefill_quota=prefill_quota,
            active_req_quota=active_req_quota,
            is_warmup=True,
        )
    )
    print(f"Warmup (mode: {args.warmup_mode}) has completed.")

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
          prefill_quota=prefill_quota,
          active_req_quota=active_req_quota,
      )
  )

  # Process output
  output = [output.to_dict() for output in request_outputs]
  if args.run_eval:
    if args.run_mmlu_dataset:
      eval_json = eval_accuracy_mmlu(output)
    else:
      eval_json = eval_accuracy(output, args.dataset[:4])

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
    if args.additional_metadata_metrics_to_save is not None:
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
  gc.disable()
  main(parse_args())
