#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

NUM_PROMPTS=${NUM_PROMPTS:-100}
MAX_OUTPUT_LENGTH=${MAX_OUTPUT_LENGTH:-50}

# Test combination from lengths and common prefix lengths.
# The length should be shorter than max_input_length minus 1 for bos.
BENCHMARK_PROMPT_LENGTHS=${BENCHMARK_PROMPT_LENGTHS:-8000,16000}
BENCHMARK_PROMPT_COMMON_PREFIX_LENGTHS=${BENCHMARK_PROMPT_COMMON_PREFIX_LENGTHS:-4000,6000,8000,10000,12000,14000,16000}

benchmark_serving_with_prefix_cache() {
  echo "Starting prefix cache benchmark..."
  echo "Benchmark serving script: ${BENCHMARK_SERVING_SCRIPT_PATH}"
  echo "Prompt lengths to test: ${BENCHMARK_PROMPT_LENGTHS}"
  echo "Common prefix lengths to test: ${BENCHMARK_PROMPT_COMMON_PREFIX_LENGTHS}"
  echo "Number of prompts per run: ${NUM_PROMPTS}"
  echo "Max output length per prompt: ${MAX_OUTPUT_LENGTH}"
  echo "Base output directory for results: ${OUTPUTS_DIR_BASE}"
  echo "Warmup mode: ${WARMUP_MODE}"

  # Convert comma-separated strings to arrays for iteration
  IFS=',' read -r -a prompt_lengths_arr <<< "$BENCHMARK_PROMPT_LENGTHS"
  IFS=',' read -r -a common_prefix_lengths_arr <<< "$BENCHMARK_PROMPT_COMMON_PREFIX_LENGTHS"

  for prompt_len in "${prompt_lengths_arr[@]}"; do
    for common_len in "${common_prefix_lengths_arr[@]}"; do
      if [ "${common_len}" -gt "${prompt_len}" ]; then
        echo "Skipping: Common prefix length ${common_len} is greater than prompt length ${prompt_len}."
        continue
      fi

      echo "----------------------------------------------------------------------"
      echo "Running benchmark: Prompt Length=${prompt_len}, Common Prefix Length=${common_len}"
      echo "----------------------------------------------------------------------"
      echo "Warm up twice"
      echo "----------------------------------------------------------------------"
      
      # With warmup-mode full, it will run twice
      python3 ./benchmark_serving.py \
        --tokenizer "prefix_cache_test" \
        --dataset "prefix_cache_test" \
        --num-prompts 10 \
        --max-output-length "${MAX_OUTPUT_LENGTH}" \
        --warmup-mode "full" \
        --max-input-length "${prompt_len}" \
        --prefix-cache-test-common-len "${common_len}"

      echo "Warm up done"
      echo "----------------------------------------------------------------------"

      python3 ./benchmark_serving.py \
        --tokenizer "prefix_cache_test" \
        --dataset "prefix_cache_test" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-output-length "${MAX_OUTPUT_LENGTH}" \
        --warmup-mode "none" \
        --max-input-length "${prompt_len}" \
        --prefix-cache-test-common-len "${common_len}"

      echo "Benchmark finished for Prompt Length=${prompt_len}, Common Prefix Length=${common_len}"
      echo "----------------------------------------------------------------------"
      echo
    done
  done
  echo "All benchmark runs completed."
}

main() {
  benchmark_serving_with_prefix_cache
  echo "Script finished."
  exit 0
}

main "$@"
