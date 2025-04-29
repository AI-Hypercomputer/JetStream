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

# Docker image name to use for executing test scripts
export LOCAL_IMAGE_TAG=${LOCAL_IMAGE_TAG}

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

echo "--- Starting test execution ---"

shopt -s nullglob
test_script_files=(test_script/*.sh)
shopt -u nullglob

echo "Found the following test scripts:"
printf " - %s\n" "${test_script_files[@]}"

declare -a failed_scripts
overall_exit_status=0

for script_path in "${test_script_files[@]}"; do
  if [[ -f "$script_path" ]]; then
    echo ">>> Running test script: $script_path"
    
    docker run --net=host --privileged --rm -i ${LOCAL_IMAGE_TAG} bash < "$script_path"
    script_exit_status=$? # Capture the exit code of the docker run command

    if [[ $script_exit_status -ne 0 ]]; then
      echo "<<< FAILED test script: $script_path (Exit Code: $script_exit_status)"
      failed_scripts+=("$script_path")
      overall_exit_status=1
    else
      echo "<<< Finished test script successfully: $script_path"
    fi
    echo
  else
    echo "--- Skipping non-file entry: $script_path ---"
  fi
done

echo

if [[ $overall_exit_status -ne 0 ]]; then
  echo "--- Test Execution Summary: FAILURES DETECTED ---"
  echo "The following scripts failed:"
  printf " - %s\n" "${failed_scripts[@]}"
  exit 1
else
  echo "--- Test Execution Summary: All tests passed successfully ---"
  exit 0
fi
