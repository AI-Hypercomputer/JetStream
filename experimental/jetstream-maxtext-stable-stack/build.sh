#!/bin/bash
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

set -xe

export LOCAL_IMAGE_TAG=${LOCAL_IMAGE_TAG:-jetstream-maxtext-stable-stack:latest}
export MAXTEXT_COMMIT_HASH=${MAXTEXT_COMMIT_HASH}
export JETSTREAM_COMMIT_HASH=${JETSTREAM_COMMIT_HASH}

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done


if [[ -z "$LOCAL_IMAGE_TAG" ]]; then
    echo -e "\n\nError: You must specify an LOCAL_IMAGE_TAG.\n\n"
    exit 1
fi

docker build --no-cache \
    --build-arg MAXTEXT_COMMIT_HASH=${MAXTEXT_COMMIT_HASH} \
    --build-arg JETSTREAM_COMMIT_HASH="${JETSTREAM_COMMIT_HASH}" \
    -t ${LOCAL_IMAGE_TAG} \
    -f ./Dockerfile .

echo "********* Sucessfully built Stable Stack Image with tag $LOCAL_IMAGE_TAG *********"
