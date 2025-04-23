# TODO: need a public path
export PARAM_PATH=${PARAM_PATH}

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
  prefill_chunk_size=64 \
  load_parameters_path=${PARAM_PATH} &

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
  --run-eval True
