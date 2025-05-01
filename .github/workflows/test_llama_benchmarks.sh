#!/bin/bash

OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/test_dir}

pip install nltk==3.8.1
python -c "import nltk; nltk.download('punkt')"

cd maxtext

export TOKENIZER_PATH=assets/tokenizer.llama2
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-70b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=54
export LOAD_PARAMETERS_PATH=gs://jetstream-runner/llama-70B-int8/int8_

python -m MaxText.maxengine_server MaxText/configs/base.yml   tokenizer_path=${TOKENIZER_PATH}   load_parameters_path=${LOAD_PARAMETERS_PATH}   max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH}   max_target_length=${MAX_TARGET_LENGTH}   model_name=${MODEL_NAME}   ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}   ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM}   ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM}   scan_layers=${SCAN_LAYERS}   weight_dtype=${WEIGHT_DTYPE}   per_device_batch_size=${PER_DEVICE_BATCH_SIZE} checkpoint_is_quantized=True quantization=int8 quantize_kvcache=True  enable_jax_profiler=True &

sleep 800

cd ..

python JetStream/benchmarks/benchmark_serving.py   --tokenizer maxtext/assets/tokenizer.llama2 --save-result   --save-request-outputs   --request-outputs-file-path outputs.json   --num-prompts 1200   --max-output-length 1024  --dataset openorca --run-eval True > ${OUTPUT_DIR}/llama_70b_jetstream.txt
#tail -n25 ${OUTPUT_DIR}/llama_70b_jetstream.txt > ${OUTPUT_DIR}/llama_70b_jetstream.tmp && mv ${OUTPUT_DIR}/llama_70b_jetstream.tmp ${OUTPUT_DIR}/llama_70b_jetstream.txt

# kill Jetstream server
kill -9 %%
tail -n25 ${OUTPUT_DIR}/llama_70b_jetstream.txt > ${OUTPUT_DIR}/llama_70b_jetstream.tmp
echo "\n8x7b Maxtext Jetstream Run throughput and accuracy for llama 70b" >> ${OUTPUT_DIR}/result_comparison.txt
grep "\nthroughput" ${OUTPUT_DIR}/llama_70b_jetstream.tmp >> ${OUTPUT_DIR}/result_comparison.txt
grep "\nrouge1" ${OUTPUT_DIR}/llama_70b_jetstream.tmp >> ${OUTPUT_DIR}/result_comparison.txt
mv ${OUTPUT_DIR}/llama_70b_jetstream.tmp ${OUTPUT_DIR}/llama_70b_jetstream.txt
