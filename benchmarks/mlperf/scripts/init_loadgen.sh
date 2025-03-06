cd /inference_mlperf4.1/language/llama2-70b/tpu/scripts/
export API_URL=0.0.0.0:9000

export DATA_DISK_DIR=/loadgen_run_data
export MODEL_NAME=llama70b
export LOG_INTERVAL=1000
export BATCH_SIZE_EXP=10
export USER_CONFIG=user.conf

export DATASET_TYPE=full
export TOTAL_SAMPLE_COUNT=24576
export NUM_CLIENT_THREADS=600

