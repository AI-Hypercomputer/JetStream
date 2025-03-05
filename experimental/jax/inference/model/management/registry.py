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

"""Model registry"""

import enum
from typing import Any
from jax.sharding import Mesh
from jax import numpy as jnp
from transformers import (
    logging,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from inference import nn
from .hf_llama_ckpt_conversion import convert_hf_llama
from .util import torch_jax_dtype_map

logging.set_verbosity_error()


@enum.unique
class ModelSource(enum.Enum):
  NATIVE = enum.auto()
  HUGGINGFACE = enum.auto()


class ModelRegistry:

  def load_tokenizer(self, model_id: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_id)

  def load_model_config(self, model_id: str) -> PretrainedConfig:
    return AutoConfig.from_pretrained(model_id)

  def load_weights_to_host(
      self,
      model_id: str,
      num_devices: int,
      model_config: PretrainedConfig,
      dtype: jnp.dtype,
  ) -> Any:
    """Load the ckpt to the host DRAM."""
    hg_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_jax_dtype_map[dtype],
        config=model_config,
    )
    state_dict = hg_model.state_dict()
    del hg_model

    weights = convert_hf_llama(state_dict, num_devices, model_config)
    return weights
