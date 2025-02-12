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

"""Llama model family."""

import jax
from transformers import LlamaConfig
from inference import nn
from inference import parallel
from .sampling.sampler import Sampler, SamplingParams
from inference.model.postprocess import *


class _LlamaFeedForward(nn.Module):

  def __init__(
      self,
      config: LlamaConfig,
      parallel_config: parallel.FeedForwardParallelConfig,
  ):
    super().__init__()
    self.config = config
    self.parallel_config = parallel_config

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    gate_up_proj_parallel = parallel.LinearParallelConfig(
        mesh=parallel_config.mesh,
        parallel_type=parallel.LinearParallelType.COLUMN,
    )

    if parallel_config.enable_collective_matmul:
      gate_up_proj_parallel.collective_matmul_type = (
          parallel.CollectiveMatmulType.ALL_GATHER
      )

    self.gate_up_proj = nn.Linear(
        in_features=hidden_size,
        out_features=[intermediate_size] * 2,
        parallel_config=gate_up_proj_parallel,
    )

    down_proj_parallel = parallel.LinearParallelConfig(
        mesh=parallel_config.mesh,
        parallel_type=parallel.LinearParallelType.ROW,
    )

    if parallel_config.enable_collective_matmul:
      down_proj_parallel.collective_matmul_type = (
          parallel.CollectiveMatmulType.REDUCE_SCATTER
      )
    else:
      down_proj_parallel.reduce_output = True

    self.down_proj = nn.Linear(
        in_features=intermediate_size,
        out_features=hidden_size,
        parallel_config=down_proj_parallel,
    )

    self.silu = jax.nn.silu

  def __call__(self, x: jax.Array) -> jax.Array:
    gate, up = self.gate_up_proj(x)
    x = self.silu(gate) * up
    x = self.down_proj(x)
    return x


class _LlamaAttention(nn.Module):

  def __init__(
      self,
      config: LlamaConfig,
      parallel_config: parallel.AttentionParallelConfig,
  ):
    super().__init__()
    self.config = config
    self.parallel_config = parallel_config
    self.hidden_size = config.hidden_size
    self.num_attn_heads = config.num_attention_heads
    self.head_dim = getattr(
        config, "head_dim", self.hidden_size // self.num_attn_heads
    )
    self.num_kv_heads = config.num_key_value_heads
    self.num_kv_groups = self.num_attn_heads // self.num_kv_heads
    self.rope_theta = config.rope_theta
    self.parallel_config = parallel_config

    column_parallel_config = parallel.config.LinearParallelConfig(
        mesh=parallel_config.mesh,
        parallel_type=parallel.LinearParallelType.COLUMN,
    )

    self.q_proj = nn.Linear(
        self.hidden_size,
        self.num_attn_heads * self.head_dim,
        parallel_config=column_parallel_config,
    )

    self.k_proj = nn.Linear(
        self.hidden_size,
        self.num_kv_heads * self.head_dim,
        parallel_config=column_parallel_config,
    )

    self.v_proj = nn.Linear(
        self.hidden_size,
        self.num_kv_heads * self.head_dim,
        parallel_config=column_parallel_config,
    )

    out_proj_parallel = parallel.config.LinearParallelConfig(
        mesh=parallel_config.mesh,
        parallel_type=parallel.LinearParallelType.ROW,
    )

    if parallel_config.reduce_output:
      out_proj_parallel.reduce_output = True
    else:
      out_proj_parallel.reduce_scatter_output = True

    self.o_proj = nn.Linear(
        self.num_attn_heads * self.head_dim,
        self.hidden_size,
        out_proj_parallel,
    )

    self.rotary_emb = nn.apply_rope_embedding
    self.attn = nn.AttentionOps(
        self.num_attn_heads,
        self.num_kv_heads,
        self.head_dim,
    )

  def __call__(
      self,
      hidden_states,
      positions,
      kv_cache: nn.KVCache,
      attn_metadata: nn.AttentionMetadata,
  ) -> tuple[jax.Array, nn.KVCache]:
    if self.parallel_config.gather_input:
      hidden_states = parallel.ops.all_gather(
          hidden_states,
          len(hidden_states.shape) - 1,
          parallel.tp_major_axis_names(),
      )

    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # reshape as (num_tokens, num_heads, head_dim)
    q = q.reshape((q.shape[0], -1, self.head_dim))
    k = k.reshape((k.shape[0], -1, self.head_dim))
    v = v.reshape((v.shape[0], -1, self.head_dim))

    q = self.rotary_emb(q, positions, self.rope_theta)
    k = self.rotary_emb(k, positions, self.rope_theta)

    output, kv_cache = self.attn(
        q,
        k,
        v,
        kv_cache,
        attn_metadata,
    )

    output = self.o_proj(output)
    return output, kv_cache


class _LlamaDecoderLayer(nn.Module):

  def __init__(
      self,
      config: LlamaConfig,
      parallel_config: parallel.DecoderLayerParallelConfig,
  ):
    super().__init__()
    self.config = config
    self.parallel_config = parallel_config
    mesh = parallel_config.mesh

    if parallel.platform() == "tpu":
      enable_collective_matmul = True
    else:
      enable_collective_matmul = False

    self.self_attn = _LlamaAttention(
        config,
        parallel.AttentionParallelConfig(
            mesh=mesh,
            gather_input=enable_collective_matmul,
            reduce_output=(not enable_collective_matmul),
        ),
    )

    self.ffw = _LlamaFeedForward(
        config,
        parallel_config=parallel.FeedForwardParallelConfig(
            mesh, enable_collective_matmul=enable_collective_matmul
        ),
    )

    self.input_layernorm = nn.RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        parallel_config=parallel.RMSNormParallelConfig(
            mesh=mesh,
            activation_sharded=enable_collective_matmul,
        ),
    )

    self.post_attention_layernorm = nn.RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        parallel_config=parallel.RMSNormParallelConfig(
            mesh=mesh,
            activation_sharded=enable_collective_matmul,
        ),
    )

  def __call__(
      self,
      hidden_states,
      positions,
      kv_cache,
      attn_metadata: nn.AttentionMetadata,
  ) -> tuple[jax.Array, nn.KVCache]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache = self.self_attn(
        hidden_states, positions, kv_cache, attn_metadata
    )
    hidden_states += residual

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    hidden_states = self.ffw(hidden_states)
    hidden_states += residual

    return hidden_states, kv_cache


class LlamaModel(nn.Model):

  def __init__(
      self,
      config: LlamaConfig,
      parallel_config: parallel.ModelParallelConfig,
  ):
    super().__init__()
    self.config = config
    self.parallel_config = parallel_config
    self.vocab_size = config.vocab_size
    mesh = parallel_config.mesh

    self.embed_tokens = nn.Embedding(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        parallel_config=parallel.EmbeddingParallelConfig(
            mesh=mesh,
            parallel_type=parallel.EmbeddingParallelType.COLUMN,
        ),
    )

    self.layers = nn.ModuleList(
        [
            _LlamaDecoderLayer(
                config=config,
                parallel_config=parallel.DecoderLayerParallelConfig(
                    mesh=parallel_config.mesh,
                ),
            )
            for _ in range(config.num_hidden_layers)
        ]
    )

    if parallel.platform() == "tpu":
      enable_collective_matmul = True
    else:
      enable_collective_matmul = False

    self.norm = nn.RMSNorm(
        config.hidden_size,
        config.rms_norm_eps,
        parallel_config=parallel.RMSNormParallelConfig(
            mesh=mesh,
            activation_sharded=enable_collective_matmul,
        ),
    )
    lm_head_parallel = parallel.LinearParallelConfig(
        mesh=mesh,
        parallel_type=parallel.LinearParallelType.COLUMN,
    )

    if enable_collective_matmul:
      lm_head_parallel.collective_matmul_type = (
          parallel.CollectiveMatmulType.ALL_GATHER
      )

    self.lm_head = nn.Linear(
        config.hidden_size,
        config.vocab_size,
        parallel_config=lm_head_parallel,
    )

  def __call__(
      self,
      input_ids,
      positions,
      kv_caches,
      attn_metadata,
  ) -> tuple[jax.Array, list[nn.KVCache]]:
    hidden_states = self.embed_tokens(input_ids)
    for i in range(self.config.num_hidden_layers):
      hidden_states, kv_cache = self.layers[i](
          hidden_states,
          positions,
          kv_caches[i],
          attn_metadata,
      )
      kv_caches[i] = kv_cache

    hidden_states = self.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    logits = parallel.ops.all_gather(
        logits, axis=len(logits.shape) - 1, axis_names=parallel.tp_axis_names()
    )

    return logits, kv_caches


class LlamaForCausalLM(nn.CausalLM):

  def __init__(
      self,
      config: LlamaConfig,
      parallel_config: parallel.ModelParallelConfig,
      eos: int | None = None,
      max_length: int | None = None,
  ):
    super().__init__()
    self.config = config
    self.model = LlamaModel(config=config, parallel_config=parallel_config)
    self.sampler = Sampler(eos, max_length)

  def __call__(
      self,
      input_ids: jax.Array,
      positions: jax.Array,
      kv_caches: list[nn.KVCache],
      attn_metadata: nn.AttentionMetadata,
      sampling_params: SamplingParams,
  ) -> tuple[ModelOutput, list[nn.KVCache]]:
    logits, kv_caches = self.model(
        input_ids, positions, kv_caches, attn_metadata
    )
    tokens, done = self.sampler.sample(
        logits, positions, attn_metadata, sampling_params
    )

    return postprocess(tokens, done, attn_metadata), kv_caches
