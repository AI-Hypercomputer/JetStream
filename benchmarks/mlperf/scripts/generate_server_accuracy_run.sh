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

source run_utils.sh

export TOKENIZER_PATH=meta-llama/Llama-2-70b-chat-hf
export DATASET_PREFIX=""
export MODEL_ID="llama2-70b"
DATASET_NAME=$(get_dataset_name ${DATASET_TYPE})
export DATASET_PATH=${DATA_DISK_DIR}/${DATASET_NAME}.pkl
export API_URL=${API_URL}
export LOADGEN_RUN_TYPE=server-accuracy
export OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}
export OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
export OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json

echo "LOADGEN_RUN_TYPE: ${LOADGEN_RUN_TYPE}"
echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "API_URL: ${API_URL}"
echo "BATCH_SIZE_EXP: ${BATCH_SIZE_EXP}"
echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
echo "OUTPUT_ACCURACY_JSON_PATH: ${OUTPUT_ACCURACY_JSON_PATH}"
echo "USER_CONFIG: ${USER_CONFIG}"

mkdir -p ${OUTPUT_LOG_DIR} && cp ../${USER_CONFIG} ${OUTPUT_LOG_DIR}
MIXTRAL_COLS_RENAME="{\"tok_input_len\": \"tok_input_length\", \"tok_ref_output_len\": \"tok_output_length\"}"

# Accuracy Run
cd ../ && python3 main.py \
	--api-url ${API_URL} \
	--is-stream \
	--accuracy \
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
	--mlperf-conf-id "${MODEL_ID}" \
        --rename-dataset-cols "${MIXTRAL_COLS_RENAME}" \
	--output-log-dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/server_accuracy_log.log

# Eval Run
if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
	python3 evaluate-accuracy.py \
		--checkpoint-path meta-llama/Llama-2-70b-chat-hf \
		--mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
		--dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_server_accuracy_log.log
fi
