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
    logging as hf_logging,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from inference import nn
from inference.model.llama import LlamaForCausalLM
from .hf_llama_ckpt_conversion import convert_hf_llama
from .util import torch_jax_dtype_map

hf_logging.set_verbosity_error()

supported_model = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "maxtext/Llama2-7b",
]

maxtext_config_map = {
    "maxtext/Llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
}

hf_ckpt_conversion = {
    "meta-llama/Llama-2-7b-chat-hf": convert_hf_llama,
    "meta-llama/Llama-2-7b-hf": convert_hf_llama,
}

hf_model_class_map: dict[str, nn.CausalLM] = {
    "meta-llama/Llama-2-7b-chat-hf": LlamaForCausalLM,
    "meta-llama/Llama-2-7b-hf": LlamaForCausalLM,
}


@enum.unique
class ModelSource(enum.Enum):
  NATIVE = enum.auto()
  HUGGINGFACE = enum.auto()
  MAXTEXT = enum.auto()


class ModelRegistry:

  def load_model_config(
      self,
      model_id: str,
      source: ModelSource = ModelSource.HUGGINGFACE,
  ) -> PretrainedConfig:
    if model_id not in supported_model:
      raise ValueError(f"{model_id} is not supported")

    if source == ModelSource.MAXTEXT:
      model_id = maxtext_config_map[model_id]

    config = AutoConfig.from_pretrained(model_id)
    return config

  def load_tokenizer(
      self,
      model_id: str,
      source: ModelSource = ModelSource.HUGGINGFACE,
      path: str | None = None,
  ) -> PreTrainedTokenizer:
    if model_id not in supported_model:
      raise ValueError(f"{model_id} is not supported")

    if path:
      raise ValueError(
          f"Load tokenizer from given path {path} is not supported"
      )

    if source == ModelSource.MAXTEXT:
      model_id = maxtext_config_map[model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

  def model_cls(self, model_id: str):
    model_cls = hf_model_class_map[model_id]
    return model_cls

  def load_model(
      self,
      mesh: Mesh,
      model_id: str,
      model_config: PretrainedConfig | None = None,
      path: str | None = None,
      source: ModelSource = ModelSource.HUGGINGFACE,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> tuple[nn.Module, dict]:
    if model_config:
      config = model_config
    else:
      config = self.load_model_config(model_id)

    weights_on_host = self.load_weights_to_host(
        model_id,
        mesh.devices.size,
        path,
        source,
        dtype,
        config,
    )
    print("loaded to host")
    if model not in hf_model_class_map:
      raise ValueError(f"cannot find class for model {model}")
    model_cls = hf_model_class_map[model]
    model: nn.Module = model_cls(config, mesh)
    print("loading to device")
    weight_dict = model.load_weights_dict(weights_on_host)
    print("loaded to device")
    return model, weight_dict

  def load_weights_to_host(
      self,
      model_id: str,
      num_devices: int,
      model_config: PretrainedConfig | None = None,
      path: str | None = None,
      source: ModelSource = ModelSource.HUGGINGFACE,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> Any:
    """Load the ckpt to the host DRAM."""

    if model_id not in supported_model:
      raise ValueError(f"{model_id} is not supported")
    if dtype not in torch_jax_dtype_map:
      raise ValueError(f"Unknown jax dtype for weight to load {dtype}")

    if source == ModelSource.HUGGINGFACE:
      if model_id not in hf_ckpt_conversion:
        raise ValueError(
            f"No weight conversion function for HF model {model_id}"
        )

      if model_config:
        config = model_config
      else:
        config = AutoConfig.from_pretrained(model_id)

      if not path:
        hg_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_jax_dtype_map[dtype],
            config=config,
        )
        ckpt_conversion_func = hf_ckpt_conversion[model_id]
        state_dict = hg_model.state_dict()
        del hg_model
        weights = ckpt_conversion_func(state_dict, num_devices, config)
        return weights
      else:
        raise NotImplemented(
            "Loading from path for HF model is not supported yet"
        )

    raise NotImplemented(f"Loading from {source} is not supported yet")
