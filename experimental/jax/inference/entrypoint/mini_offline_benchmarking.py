"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import time
import pandas
from inference.runtime.request_type import *
from inference.runtime import offline_inference


def load_openorca_dataset_pkl():
  # Read pickle file
  current_dir = os.path.dirname(__file__)
  samples = pandas.read_pickle(
      f"{current_dir}/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
  )

  prompts = []
  outputs = []
  for _, row in samples.iterrows():
    prompts.append(row["input"])
    outputs.append(row["output"])

  return [(prompt, output) for prompt, output in zip(prompts, outputs)]


def benchmarking():
  dataset = load_openorca_dataset_pkl()

  ds = dataset[:1000]
  ds = [d[0] for d in ds]

  inference_instance = offline_inference.OfflineInference()

  start_time = time.perf_counter()
  res_list: list[Response] = inference_instance(ds)
  end_time = time.perf_counter()
  duration = end_time - start_time

  input_tokens = []
  for res in res_list:
    input_tokens = input_tokens + res.input_tokens

  output_tokens = []
  for res in res_list:
    output_tokens = output_tokens + res.generated_tokens

  num_input_tokens = len(input_tokens)
  num_output_tokens = len(output_tokens)

  print("Benchmarking result: ")
  # Hardcode the number of requests as 1000 based on the test
  # dataset.
  print("  Total requests: 1000")
  print("  Total input tokens:", num_input_tokens)
  print("  Total output tokens:", num_output_tokens)
  print(f"  Input token throughput: {num_input_tokens/duration} tokens/sec")
  print(f"  Output token throughput: {num_output_tokens/duration} tokens/sec")


if __name__ == "__main__":
  benchmarking()
