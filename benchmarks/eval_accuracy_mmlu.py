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

"""Evaluate accuracy of JetStream online serving."""

import argparse
import nltk
import evaluate
import json
import re

import numpy as np


def postprocess_text(preds, targets):
  choices = ["A", "B", "C", "D", None]

  def _parse_answer(output):
    match = re.search(r"\s*\(([A-D])\)\s*\w*", output, re.IGNORECASE)
    predicted_answer = match.group(1).upper() if match else None
    return predicted_answer

  preds = [choices.index(_parse_answer(pred.strip())) for pred in preds]
  targets = [choices.index(target.strip().upper()) for target in targets]
  return preds, targets


def eval_accuracy_mmlu(request_outputs_dict):
  metric = evaluate.load("accuracy")
  nltk.download("punkt")
  preds = []
  targets = []

  for output in request_outputs_dict:
    preds.append(output["generated_text"])
    targets.append(output["original_output"])
  preds, targets = postprocess_text(preds, targets)
  result = metric.compute(
      predictions=preds,
      references=targets,
  )
  result = {k: float(round(np.mean(v), 4)) for k, v in result.items()}
  result["gen_num"] = len(preds)
  print("\nResults\n")
  print(result)
  return result


def main(args):
  with open(args.output_path, "r", encoding="utf-8") as f:
    request_outputs_dict = json.load(f)

  eval_accuracy_mmlu(request_outputs_dict)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_path",
      type=str,
      default="/tmp/request-outputs.json",
      help="File path which has original_output and inference generated_text.",
  )

  parsed_args = parser.parse_args()

  main(parsed_args)
