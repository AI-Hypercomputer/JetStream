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

import dataclasses


class ModelId:
  llama_2_7b_chat_hf = "meta-llama/Llama-2-7b-chat-hf"
  llama_2_70b_chat_hf = "meta-llama/Llama-2-70b-chat-hf"


@dataclasses.dataclass
class InferenceParams:
  model_id: str
  batch_size: int
  max_seq_length: int
  max_input_length: int
  prefill_chunk_sizes: list[int]
  page_size: int
  hbm_utilization: float


class Config:
  _configs = {
      ModelId.llama_2_7b_chat_hf: InferenceParams(
          model_id=ModelId.llama_2_7b_chat_hf,
          batch_size=320,
          max_seq_length=2048,
          max_input_length=1024,
          prefill_chunk_sizes=[128, 256, 512, 1024],
          page_size=128,
          hbm_utilization=0.875,
      ),
      ModelId.llama_2_70b_chat_hf: InferenceParams(
          model_id=ModelId.llama_2_70b_chat_hf,
          batch_size=100,
          max_seq_length=2048,
          max_input_length=1024,
          prefill_chunk_sizes=[128, 256, 512, 1024],
          page_size=128,
          hbm_utilization=0.875,
      ),
  }

  @classmethod
  def get(cls, model_id: str) -> InferenceParams:
    return cls._configs[model_id]
