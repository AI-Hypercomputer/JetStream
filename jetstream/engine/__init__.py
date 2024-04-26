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

"""Initialization for any Engine implementation."""

import jax


def register_proxy_backend():
  """Try to register IFRT Proxy backend if it's needed."""
  # TODO: find a more elegant way to do it.
  if jax.config.jax_platforms and "proxy" in jax.config.jax_platforms:
    try:
      jax.lib.xla_bridge.get_backend("proxy")
    except RuntimeError:
      try:
        from jaxlib.xla_extension import ifrt_proxy  # pylint: disable=import-outside-toplevel

        jax_backend_target = jax.config.read("jax_backend_target")
        jax._src.xla_bridge.register_backend_factory(  # pylint: disable=protected-access
            "proxy",
            lambda: ifrt_proxy.get_client(
                jax_backend_target,
                ifrt_proxy.ClientConnectionOptions(),
            ),
            priority=-1,
        )
        print(f"Registered IFRT Proxy with address {jax_backend_target}")
      except ImportError as e:
        print(f"Failed to register IFRT Proxy, exception: {e}")
        pass


register_proxy_backend()
