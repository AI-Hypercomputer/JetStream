# Copyright 2024 Google LLC
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


# Tokenizer
# export TOKENIZER_PATH=meta-llama/Llama-2-70b-chat-hf
export DATASET_PREFIX="mixtral-"
export TOKENIZER_PATH=mistralai/Mixtral-8x7B-Instruct-v0.1
export NUM_CLIENT_THREADS=${NUM_CLIENT_THREADS:=600}

# Loadgen
export LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)

get_dataset_name() {
  dataset_type=$1
	if [ ${dataset_type} = "full" ]
		then echo "${DATASET_PREFIX}processed-data"
	elif [ ${dataset_type} = "calibration" ]
		then echo "${DATASET_PREFIX}-processed-calibration-data"
	fi
}
