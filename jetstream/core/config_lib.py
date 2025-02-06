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

"""Configs of engines for the orchestrator to load."""

import dataclasses
import functools
from typing import Any, Callable, List, Optional, Tuple, Type
from numpy import uint16

from jetstream.engine import engine_api
from jetstream.engine import mock_engine


Devices = Any

CreateEngineFn = Callable[[Devices], engine_api.Engine]


@dataclasses.dataclass
class ServerConfig:
  """Configs for slices to put engines on."""

  prefill_slices: Tuple[str, ...] = ()
  generate_slices: Tuple[str, ...] = ()
  interleaved_slices: Tuple[str, ...] = ()
  prefill_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  generate_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  interleaved_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  is_ray_backend: bool = False
  # Parameters for customized gc config, increase the numbers here will
  # potentially increase memory usage.
  gc_gen0_allocs: int = 60000  # default is 700, too frequent sometimes.
  gc_gen1_multipler: int = 2  # Make gen1 gc runs less frequent
  gc_gen2_multipler: int = 3  # Make gen2 gc runs less frequent


@dataclasses.dataclass
class InstantiatedEngines:
  prefill_engines: List[engine_api.Engine]
  generate_engines: List[engine_api.Engine]
  interleaved_engines: List[engine_api.Engine]


@dataclasses.dataclass
class MetricsServerConfig:
  port: uint16
  model_name: Optional[str] = None


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#


def get_test_engine(devices: Devices, weight: int) -> engine_api.Engine:
  del devices
  return mock_engine.TestEngine(
      batch_size=8,
      cache_length=32,
      weight=weight,
  )


@dataclasses.dataclass
class CPUTestServer(ServerConfig):
  prefill_slices = ("cpu=1",)
  generate_slices = ("cpu=1",)
  prefill_engine_create_fns = (functools.partial(get_test_engine, weight=2),)
  generate_engine_create_fns = (functools.partial(get_test_engine, weight=4),)


@dataclasses.dataclass
class InterleavedCPUTestServer(ServerConfig):
  interleaved_slices = ("cpu=1",)
  interleaved_engine_create_fns = (
      functools.partial(get_test_engine, weight=2),
  )


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#


def slice_to_num_chips(s: str) -> int:
  """Converts a TPU spec like v5e-8 or v5e=8 to the number of chips, 8."""
  # Account for the case where it is written 'tpu=8' for compatibility.
  delim = "-" if "-" in s else "="
  # TODO: Support more accelerator type check.
  accelerator_type, num_devices = s.split(delim)
  return int(num_devices) if accelerator_type != "v4" else int(num_devices) // 2


def _split_devices_by_slices(
    devices: list[Devices], slices: list[int]
) -> List[List[Devices]]:
  """Converts an ordered list of devices into slices."""
  assert sum(slices) == len(devices), f"{sum(slices)} != {len(devices)}"
  cumsum = 0
  slice_split_devices = []
  for sl in slices:
    slice_split_devices.append(devices[cumsum : cumsum + sl])
    cumsum += sl
  return slice_split_devices


def get_engines(
    server_config: Type[ServerConfig], devices: List[Devices]
) -> InstantiatedEngines:
  """Processes config to get the appropriate engines.

  Args:
    server_config: ServerConfig.
    devices: Device objects.

  Returns:
    Instantiated engines!

  Devices are popped in order!
  """
  # Now, we need to split devices by slice due to TPU backend config.
  slices: list[int] = [
      slice_to_num_chips(s)
      for s in list(server_config.prefill_slices)
      + list(server_config.generate_slices)
      + list(server_config.interleaved_slices)
  ]
  if sum(slices) != len(devices):
    raise ValueError(
        f"The number of available devices ({len(devices)}) do not match the "
        f"expected number of devices on all the slices ({sum(slices)}) "
        "specified in the server_config:\n"
        f"{server_config.prefill_slices=}\n"
        f"{server_config.generate_slices=}\n"
        f"{server_config.interleaved_slices=}\n"
    )
  # e.g. [[tpu_0], [tpu_1]] corresponding to prefill: v5e=1x1
  # generate = v5e=1x1; or [[tpu_0, tpu_1, tpu_2, tpu_3]] corresponding to
  # interleaved: v5e=2x2
  split_devices = _split_devices_by_slices(devices, slices)
  prefill_engines = [
      e(split_devices.pop(0)) for e in server_config.prefill_engine_create_fns
  ]
  generate_engines = [
      e(split_devices.pop(0)) for e in server_config.generate_engine_create_fns
  ]
  # These share chips and weights for prefill and generation.
  interleaved_engines = [
      e(split_devices.pop(0))
      for e in server_config.interleaved_engine_create_fns
  ]
  return InstantiatedEngines(
      prefill_engines, generate_engines, interleaved_engines
  )
