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
    # read pickle file
    samples = pandas.read_pickle(
        "/home/zhihaoshan/JetStream/benchmarks/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
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
print(len(ds))

inference_instance = offline_inference.OfflineInference()
res_list: list[Response] =  inference_instance(ds)

# print(res)

input_tokens = []
for res in res_list:
    input_tokens = input_tokens + res.input_tokens

out_tokens = []
for res in res_list:
    out_tokens = out_tokens + res.generated_tokens

print("sum_input_tokens: ", len(input_tokens))
print("sum_output_tokens: ", len(out_tokens))
