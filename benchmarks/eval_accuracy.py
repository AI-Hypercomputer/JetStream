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
from benchmarks.math_utils import extract_numbers, post_processing_math_ans, sympify_set


def extract_boxed_answers(text):
  pieces = text.split("boxed{")
  if len(pieces) == 1:
    return [""]
  piece = pieces[1]
  ans = []
  for piece in pieces[1:]:
    n = 0
    for i in range(len(piece)):
      if piece[i] == "{":
        n += 1
      elif piece[i] == "}":
        n -= 1
        if n < 0:
          if i + 1 < len(piece) and piece[i + 1] == "%":
            ans.append(piece[: i + 1])
            break
          else:
            ans.append(piece[:i])
            break
  if ans:
    return ans
  else:
    return [""]


def extract_answer(pred_str, exhaust=False):
  pred = []
  if "boxed{" in pred_str:
    pred = extract_boxed_answers(pred_str)
  elif "Answer:" in pred_str:
    matches = re.findall(r"Answer:[\*]*\s+(\S*.*)", pred_str)
    if matches:
      pred = [extract_numbers(matches[-1])]
  elif "the answer is" in pred_str:
    pred = [extract_numbers(pred_str.split("the answer is")[-1].strip())]
  elif "final answer is $" in pred_str and "$. I hope" in pred_str:
    tmp = pred_str.split("final answer is $", 1)[1]
    pred = [tmp.split("$. I hope", 1)[0].strip()]
  else:  # use the last number
    pattern = r"-?\d*\.?\d+"
    ans = re.findall(pattern, pred_str.replace(",", ""))
    if len(ans) >= 1:
      ans = ans[-1]
    else:
      ans = ""
    if ans:
      pred.append(ans)
  # multiple line
  pred_list = []
  for ans in pred:
    ans = ans.replace("<|end_of_text|>", "")
    ans = ans.strip().split("\n")[0]
    ans = ans.lstrip(":")
    ans = ans.lstrip("$")
    ans = ans.rstrip("$")
    ans = ans.rstrip(".")
    ans = ans.rstrip("/")
    pred_list.append(ans)
  if exhaust:
    return pred_list
  else:
    return pred_list[-1] if pred_list else ""


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

      p = extract_answer(p)
      ans_set = post_processing_math_ans(p)
      sympified_ans_set = sympify_set(ans_set)

      target_set = post_processing_math_ans(t)
      sympified_target_set = sympify_set(target_set)

      if sympified_target_set == sympified_ans_set:
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
