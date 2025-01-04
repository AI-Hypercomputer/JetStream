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
from typing import Any
import json
import pandas
from inference.runtime.request_type import *
from inference.runtime import offline_inference

# def load_sharegpt_dataset(
#     dataset_path: str,
# ) -> list[tuple[Any, Any]]:
#     # Load the dataset.
#     with open(dataset_path, "r", encoding="utf-8") as f:
#         dataset = json.load(f)

#     # Filter out the conversations with less than 2 turns.
#     dataset = [data for data in dataset if len(data["conversations"]) >= 2]

#     dataset = [
#         data
#         for data in dataset
#         if data["conversations"][0]["from"] == "human"
#     ]

#     # Only keep the first two turns of each conversation.
#     dataset = [
#       (data["conversations"][0]["value"], data["conversations"][1]["value"])
#       for data in dataset
#     ]

#     return dataset

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

dataset = load_openorca_dataset_pkl()

ds = dataset[:1000]
ds = [d[0] for d in ds]

inference_instance = offline_inference.OfflineInference()
res_list: list[Response] =  inference_instance(ds)

# print(res)

input_tokens = []
for res in res_list:
    input_tokens = input_tokens + res.input_tokens

out_tokens = []
for res in res_list:
    out_tokens = out_tokens + res.generated_tokens

print("sum of the number of input tokens: ", len(input_tokens))
print("sum of the number of output tokens: ", len(out_tokens))
