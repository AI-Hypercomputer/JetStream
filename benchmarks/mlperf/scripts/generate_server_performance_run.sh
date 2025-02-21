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

set -u  # Treat unset variables as an error
set -e  # Exit script if any command fails

function get_dataset_name() {
  local dataset_type=$1
	if [ ${dataset_type} = "full" ]
		then echo "processed-data"
	elif [ ${dataset_type} = "calibration" ]
		then echo "processed-calibration-data"
	fi
}

export MODEL_ID="llama2-70b"

export DATA_DISK_DIR=/home/$USER/loadgen_run_data
export DATASET_TYPE=full # for calibration run, DATASET_TYPE=calibration
export DATASET_NAME=$(get_dataset_name ${DATASET_TYPE})
export DATASET_PATH=${DATA_DISK_DIR}/${DATASET_NAME}.pkl

export API_URL=0.0.0.0:9000
export USER_CONFIG=user.conf
export TOTAL_SAMPLE_COUNT=24576 # for calibration run, TOTAL_SAMPLE_COUNT=1000
export BATCH_SIZE_EXP=8
export TOKENIZER_PATH=meta-llama/Llama-2-70b-chat-hf
export LOG_INTERVAL=1000
export NUM_CLIENT_THREADS=600
export RENAME_DATASET_COLS="{\"tok_input_len\": \"tok_input_length\", \"tok_ref_output_len\": \"tok_output_length\"}"

export OUTPUT_LOG_ID=${MODEL_ID}-${DATASET_TYPE}-performance-$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
export OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}

mkdir -p  ${OUTPUT_LOG_DIR}

pushd ../
python3 main.py \
	--api-url ${API_URL} \
	--is-stream \
	--log-pred-outputs \
	--scenario Server \
	--input-mode tokenized \
	--output-mode tokenized \
	--max-output-len 1024 \
	--mlperf-conf mlperf.conf \
	--user-conf ${USER_CONFIG} \
	--audit-conf no-audit \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--batch-size-exp ${BATCH_SIZE_EXP} \
	--dataset-path ${DATASET_PATH} \
	--tokenizer-path ${TOKENIZER_PATH} \
	--log-interval ${LOG_INTERVAL} \
	--num-client-threads ${NUM_CLIENT_THREADS} \
	--rename-dataset-cols "${RENAME_DATASET_COLS}" \
	--mlperf-conf-id "${MODEL_ID}" \
	--output-log-dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/server_performance_log.log
popd
