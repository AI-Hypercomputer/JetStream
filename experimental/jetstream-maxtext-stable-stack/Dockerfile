# Copyright 2025 Google LLC
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


FROM alpine/git:2.47.2 AS maxtext_cloner

ARG MAXTEXT_COMMIT_HASH

WORKDIR /src

RUN \
  git clone --depth=1 https://github.com/AI-Hypercomputer/maxtext.git && \
  if [ -n "${MAXTEXT_COMMIT_HASH}" ]; then \
  cd maxtext && \
  git fetch origin ${MAXTEXT_COMMIT_HASH} && \
  git switch --detach ${MAXTEXT_COMMIT_HASH}; \
  fi


FROM alpine/git:2.47.2 AS jetstream_cloner

ARG JETSTREAM_COMMIT_HASH

WORKDIR /src
RUN \
  git clone --depth=1 https://github.com/AI-Hypercomputer/JetStream.git && \
  if [ -n "${JETSTREAM_COMMIT_HASH}" ]; then \
  cd JetStream && \
  git fetch origin ${JETSTREAM_COMMIT_HASH} && \
  git switch --detach ${JETSTREAM_COMMIT_HASH}; \
  fi

FROM python:3.10-slim-bullseye AS runner

WORKDIR /jetstream_maxtext_stable_stack

# Environment variable for no-cache-dir and pip root user warning
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PYTHON_VERSION=3.10
ENV CLOUD_SDK_VERSION=latest

# Set DEBIAN_FRONTEND to noninteractive to avoid frontend errors
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && \
  apt-get install -y --no-install-recommends git git-lfs \
  && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set environment variables for Google Cloud SDK
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Install MaxText package
COPY --from=maxtext_cloner /src .
RUN cd maxtext && bash setup.sh

# MaxText install jetstream from the main. Need overwrite it.
# Install JetStream requirements
COPY --from=jetstream_cloner /src .
RUN python3 -m pip install ./JetStream
RUN python3 -m pip install -r ./JetStream/benchmarks/requirements.in

COPY generate_manifest.sh .
RUN \
  bash ./generate_manifest.sh \
  PREFIX=jetstream_maxtext \
  MAXTEXT_COMMIT_HASH=$(git -C ./maxtext rev-parse HEAD) \
  JETSTREAM_COMMIT_HASH=$(git -C ./JetStream rev-parse HEAD)
