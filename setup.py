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

from setuptools import find_packages, setup


def parse_requirements(filename):
  """load requirements from a pip requirements file."""
  with open(filename) as f:
    lineiter = (line.strip() for line in f)
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="google-jetstream",
    version="0.2.2",
    description=(
        "JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (and GPUs in future -- PRs welcome)."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Google LLC",
    url="https://github.com/google/JetStream",
    packages=find_packages(exclude="benchmarks"),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.in"),
)
