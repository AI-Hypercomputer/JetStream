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

"""Evaluate accuracy of JetStream online serving only for long context dataset."""

import argparse
import nltk
import evaluate
from tqdm import tqdm
import pandas as pd
import json
import re
from multiprocessing import Pool, cpu_count
import numpy as np
from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge(label, pred):
  score = scorer.score(label, pred)
  return {
      "rougeL": 100 * score["rougeL"].fmeasure,
  }


def niah_em(label, pred):
  label_uuids = re.findall(r"[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}", label)
  pred_uuids = re.findall(r"[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}", pred)

  if len(pred_uuids) == 0:
    return {"exact_match": 0.0}

  # https://github.com/hsiehjackson/RULER/blob/main/scripts/eval/synthetic/constants.py#L28
  score = (
      sum(
          [
              sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
              / len(ref)
              for pred, ref in zip(pred_uuids, label_uuids)
          ]
      )
      / len(pred_uuids)
      * 100
  )

  return {"exact_match": round(score, 2)}


def qa_em(label, pred):
  answer_substring = pred

  if "Answer: " in pred:
    last_answer_index = pred.rfind("Answer: ")
    if last_answer_index == -1:
      return {"exact_match": 0.0}

    answer_substring = pred[last_answer_index + len("Answer: ") :]

  if answer_substring in label:
    return {"exact_match": 100.0}

  normalized_answer = re.sub(r"\s+", "", answer_substring).lower()
  label_entries = [
      re.sub(r"\s+", "", entry).lower() for entry in label.split("|")
  ]

  match_found = any(entry in normalized_answer for entry in label_entries)
  return {"exact_match": 100.0 if match_found else 0.0}


metrics = {fn.__name__: fn for fn in [rouge, niah_em, qa_em]}


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]

  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def process_item(item):
  pred, target, metric = item
  metric_fn = metrics[metric]
  metric_eval = metric_fn(target, pred)
  return metric_eval


def run_evaluation(preds, targets, metrics, n_process=None):
  n_process = cpu_count() if n_process is None else n_process
  with Pool(n_process) as pool:
    accuracies = list(
        tqdm(
            pool.imap(process_item, zip(preds, targets, metrics)),
            total=len(preds),
        )
    )
  df = pd.DataFrame({"accuracy": accuracies, "metric": metrics})
  return df.accuracy.apply(pd.Series).describe().loc["mean"].to_dict()


def eval_accuracy_longcontext(request_outputs_dict):
  nltk.download("punkt")
  preds = []
  targets = []
  metrics = []
  for output in request_outputs_dict:
    preds.append(output["generated_text"])
    targets.append(output["original_output"])
    metrics.append(output["metric"])
  preds, targets = postprocess_text(preds, targets)
  result = run_evaluation(preds, targets, metrics)
  result = dict(result)
  prediction_lens = [len(pred) for pred in preds]
  result["gen_len"] = int(np.sum(prediction_lens))
  result["gen_num"] = len(preds)
  print("\nResults\n")
  print(result)
  return result


def main(args):
  with open(args.output_path, "r", encoding="utf-8") as f:
    request_outputs_dict = json.load(f)

  eval_accuracy_longcontext(request_outputs_dict)


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
