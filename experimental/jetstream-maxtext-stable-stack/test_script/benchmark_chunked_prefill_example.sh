cd maxtext

LIBTPU_INIT_ARGS="--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000" \
python -m MaxText.benchmark_chunked_prefill \
  MaxText/configs/inference.yml \
  tokenizer_path=assets/tokenizer.mistral-v1 \
  max_prefill_predict_length=8192 \
  max_target_length=8704 \
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
  prefill_chunk_size=2048

