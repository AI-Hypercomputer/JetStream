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
import math
import random
import time
import pandas
from inference.config.config import ModelId
from inference.runtime.request_type import *
from inference.runtime import offline_inference


def _load_openorca_dataset(size: int, shuffle: bool) -> list[str]:
  # Read pickle file
  current_dir = os.path.dirname(__file__)
  samples = pandas.read_pickle(
      # f"{current_dir}/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
      f"{current_dir}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
  )
  data = [row["input"] for _, row in samples.iterrows()]

  # Repeat data if necessary
  n = len(data)
  if n < size:
    data = data * math.ceil(size / n)
  assert len(data) >= size

  # Shuffle data if requested
  if shuffle:
    return random.sample(data, size)
  else:
    return data[:size]


def benchmark():
  size = 24_000
  dataset = _load_openorca_dataset(size=size, shuffle=True)
  assert len(dataset) == size

  inference = offline_inference.OfflineInference(
      model_id=ModelId.llama_2_7b_chat_hf,
      num_engines=1,
      enable_multiprocessing=False,
  )

  start_time = time.perf_counter()
  res_list: list[Response] = inference(dataset)
  duration = time.perf_counter() - start_time

  num_input_tokens = sum(map(lambda r: len(r.input_tokens), res_list))
  num_output_tokens = sum(map(lambda r: len(r.generated_tokens), res_list))

  print("Benchmarking result: ")
  print("  Total requests:", len(dataset))
  print("  Total input tokens:", num_input_tokens)
  print("  Total output tokens:", num_output_tokens)
  print(f"  Input token thruput: {num_input_tokens/duration: .2f} tokens/sec")
  print(f"  Output token thruput: {num_output_tokens/duration: .2f} tokens/sec")


if __name__ == "__main__":
  benchmark()
