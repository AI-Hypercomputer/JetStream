#!/bin/bash

SERVER_PID=""
CLIENT_PID=""

python -c "import nltk; nltk.download('punkt')"

pushd maxtext
LIBTPU_INIT_ARGS="--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000" \
python -m MaxText.maxengine_server \
  MaxText/configs/inference.yml \
  tokenizer_path=assets/tokenizer.mistral-v1 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  model_name=mixtral-8x7b \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=8 \
  scan_layers=false \
  weight_dtype=bfloat16 \
  per_device_batch_size=8 \
  megablox=False \
  quantization=int8 \
  quantize_kvcache=False \
  checkpoint_is_quantized=True \
  capacity_factor=1 \
  attention=dot_product \
  model_call_mode=inference \
  sparse_matmul=False \
  use_chunked_prefill=true \
  prefill_chunk_size=256 \
  load_parameters_path=gs://jetstream-runner/8-7B-int8 &

SERVER_PID=$!

popd

# mixtral-8x7b
python ./JetStream/benchmarks/benchmark_serving.py \
  --tokenizer ./maxtext/assets/tokenizer.mistral-v1 \
  --warmup-mode sampled \
  --save-result \
  --save-request-outputs \
  --request-outputs-file-path outputs.json \
  --num-prompts 100 \
  --max-output-length 2048 \
  --dataset openorca \
  --run-eval True &

CLIENT_PID=$!

while true; do
  # If server is not running, it is crash. Terminate the script.
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    exit 1
  fi

  # If client is done
  if ! kill -0 "${CLIENT_PID}" 2>/dev/null; then
    wait $CLIENT_PID
    exit $?
  fi

  sleep 1
done
