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

import numpy as np


def extract_boxed_answers(text):
  pieces = text.split("boxed{")
  if len(pieces) == 1:
    return ""
  piece = pieces[1]
  n = 0
  for i in range(len(piece)):
    if piece[i] == "{":
      n += 1
    elif piece[i] == "}":
      n -= 1
      if n < 0:
        if i + 1 < len(piece) and piece[i + 1] == "%":
          return piece[: i + 1]
        else:
          return piece[:i]
  return ""


def replace_space_answers(text):
  return text.replace(" ", "")


def special_handling(text):
  return text.replace("\\dfrac", "\\frac")


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]

  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def eval_accuracy(request_outputs_dict, match_type):
  preds = []
  targets = []
  for output in request_outputs_dict:
    preds.append(output["generated_text"])
    targets.append(output["original_output"])

  if match_type == "math":
    correct_ans = 0
    wrong_ans = 0
    for p, t in zip(preds, targets):
      ans = extract_boxed_answers(p)
      ans = replace_space_answers(ans)
      ans = special_handling(ans)
      tt = replace_space_answers(t)
      if tt == ans:
        correct_ans += 1
        continue
      wrong_ans += 1
    total_ans = correct_ans + wrong_ans
    result = {}
    result["literal"] = correct_ans / total_ans if total_ans > 0 else 0.0
    result["gen_len"] = total_ans
    result["gen_num"] = total_ans
  else:
    metric = evaluate.load("rouge")
    nltk.download("punkt_tab")
    preds, targets = postprocess_text(preds, targets)
    result = metric.compute(
        predictions=preds,
        references=targets,
        use_stemmer=True,
        use_aggregator=False,
    )
    result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = int(np.sum(prediction_lens))
    result["gen_num"] = len(preds)

  print("\nResults\n")
  print(result)
  return result


def main(args):
  with open(args.output_path, "r", encoding="utf-8") as f:
    request_outputs_dict = json.load(f)

  eval_accuracy(request_outputs_dict, args.match_type)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_path",
      type=str,
      default="/tmp/request-outputs.json",
      help="File path which has original_output and inference generated_text.",
  )
  parser.add_argument(
      "--match_type",
      type=str,
      default="rouge",
      nargs="?",
      help="Optional, values are 'rouge' or 'math'. The way to measure the "
      "accuracy of the results. ",
  )

  parsed_args = parser.parse_args()

  main(parsed_args)