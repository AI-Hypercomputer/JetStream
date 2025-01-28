
source config_utils.sh

export checkpoint_path=gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8w_

export quant_mode=w-i8w-kv-i8
export checkpoint_is_quantized=True
export quantization=int8w
export quantize_kvcache=True
export kv_quant_axis=heads_and_dkv
export kv_quant_dtype=int8
export per_device_batch_size=${per_device_batch_size:=28}
