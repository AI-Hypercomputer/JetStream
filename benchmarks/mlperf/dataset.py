# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset.py")


class Dataset:

  def __init__(
      self,
      dataset_path: str,
      input_mode: str,
      total_sample_count: int = 24576,
      perf_count_override: int = 0,
      dataset_rename_cols: str = "",
  ):
    if not os.path.isfile(dataset_path):
      log.warn(
          "Processed pickle file {} not found. Please check that the path is correct".format(
              dataset_path
          )
      )
    self.dataset_path = dataset_path

    self._input_mode = validate_sample_mode(input_mode)
    self.dataset_rename_cols = dataset_rename_cols
    self.load_processed_dataset()

    self.total_sample_count = min(len(self.input_ids_strs), total_sample_count)
    self.perf_count = perf_count_override or self.total_sample_count

  @property
  def input_ids_strs(self):
    return self._input_ids_strs

  @property
  def input_texts(self):
    return self._input_texts

  @property
  def input_token_lengths(self):
    return self._input_token_lengths

  @property
  def inputs(self):
    return self._inputs

  @property
  def inputs_with_token_lengths(self):
    return self._inputs_with_token_lengths

  def load_processed_dataset(self):
    processed_data = pd.read_pickle(self.dataset_path)
    if self.dataset_rename_cols:
      rename_dict = json.loads(self.dataset_rename_cols)
      processed_data.rename(columns=rename_dict, inplace=True)
      log.info(f"Renaming columns of dataset with mapping: {rename_dict}")

    self._input_ids_strs = []
    for input_ids in processed_data["tok_input"]:
      input_ids_str = ",".join([str(input_id) for input_id in input_ids])
      self._input_ids_strs.append(input_ids_str)

    self._input_texts = []
    for input_text in processed_data["input"]:
      self._input_texts.append(input_text)

    self._input_token_lengths = []
    for token_length in processed_data["tok_input_length"]:
      self._input_token_lengths.append(token_length)

    log.info(f"input_mode is {self._input_mode}")
    self._inputs = (
        self._input_ids_strs
        if self._input_mode == "tokenized"
        else self._input_texts
    )
    log.info(f"example sample input is {self._inputs[0]}")
    self._inputs_with_token_lengths = [
        (input_ids_str_or_input_text, token_length)
        for input_ids_str_or_input_text, token_length in zip(
            self._inputs, self._input_token_lengths
        )
    ]

  def LoadSamplesToRam(self, sample_list):
    pass

  def UnloadSamplesFromRam(self, sample_list):
    pass

  def __del__(self):
    pass


SAMPLE_MODE_CHOICES = ["tokenized", "text"]


def validate_sample_mode(sample_mode: str) -> str:
  if sample_mode not in SAMPLE_MODE_CHOICES:
    raise ValueError(
        "The sample_mode should be set to either `tokenized` or `text`."
    )
  return sample_mode
