
source config_utils.sh

export checkpoint_path=gs://morgandu-tpu/checkpoints/quantized/aqt/llama2-70b-chat

export quant_mode=w-i8-kv-i8
export checkpoint_is_quantized=True
export quantization=int8
export quantize_kvcache=True
export kv_quant_axis=heads_and_dkv
export kv_quant_dtype=int8
export per_device_batch_size=${per_device_batch_size:=28}
