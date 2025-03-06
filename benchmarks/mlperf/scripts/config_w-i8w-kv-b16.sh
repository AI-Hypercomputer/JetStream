
source config_utils.sh

export checkpoint_path=gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8w_

export quant_mode=w-i8w-kv-b16
export checkpoint_is_quantized=True
export quantization=int8w
export quantize_kvcache=False
export kv_quant_axis=
export kv_quant_dtype=
export per_device_batch_size=${per_device_batch_size:=14}
