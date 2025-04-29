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

export LOCAL_IMAGE_TAG="jetstream-maxtext-stable-stack:nightly"
export MAXTEXT_COMMIT_HASH=""
export JETSTREAM_COMMIT_HASH=""
export UPLOAD_IMAGE_TAG=""

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ -z "$UPLOAD_IMAGE_TAG" ]]; then
    echo -e "\n\nError: You must specify an UPLOAD_IMAGE_TAG.\n\n"
    exit 1
fi


docker_image_upload()
{
  local nightly_tag=${UPLOAD_IMAGE_TAG%:*}:nightly
  docker tag ${LOCAL_IMAGE_TAG} ${UPLOAD_IMAGE_TAG}
  docker tag ${LOCAL_IMAGE_TAG} ${nightly_tag}
  docker push ${UPLOAD_IMAGE_TAG}
  docker push ${nightly_tag}
  echo "All done, check out your artifacts at: ${UPLOAD_IMAGE_TAG}"
}

gcloud auth configure-docker us-docker.pkg.dev --quiet
./build.sh 
./test.sh
docker_image_upload
