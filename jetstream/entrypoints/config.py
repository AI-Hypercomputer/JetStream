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

"""Config for JetStream Server (including engine init)."""

import functools
import os
from jetstream.engine.implementations.maxtext.MaxText.maxengine_config import create_maxengine
import pyconfig
from typing import Sequence, Type

import jax


from jetstream.core import config_lib
from jetstream_pt import config


def get_server_config(
    config_str: str, argv: Sequence[str]
) -> config_lib.ServerConfig | Type[config_lib.ServerConfig]:
  match config_str:
    case "MaxtextInterleavedServer":
      jax.config.update("jax_default_prng_impl", "unsafe_rbg")
      os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
      pyconfig.initialize(argv)
      server_config = config_lib.ServerConfig(
          prefill_slices=(),
          generate_slices=(),
          interleaved_slices=("tpu=" + str(jax.device_count()),),
          prefill_engine_create_fns=(),
          generate_engine_create_fns=(),
          interleaved_engine_create_fns=(
              functools.partial(create_maxengine, config=pyconfig.config),
          ),
      )
    case "PyTorchInterleavedServer":
      os.environ["XLA_FLAGS"] = (
          "--xla_dump_to=/tmp/xla_logs --xla_dump_hlo_as_text"
      )
      engine = config.create_engine_from_config_flags()
      server_config = config_lib.ServerConfig(
          prefill_slices=(),
          generate_slices=(),
          interleaved_slices=("tpu=" + str(jax.device_count()),),
          prefill_engine_create_fns=(),
          generate_engine_create_fns=(),
          interleaved_engine_create_fns=(lambda a: engine,),
      )
    case "InterleavedCPUTestServer":
      server_config = config_lib.InterleavedCPUTestServer
    case "CPUTestServer":
      server_config = config_lib.CPUTestServer
    case _:
      raise NotImplementedError
  return server_config
