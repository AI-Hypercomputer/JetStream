
source config_utils.sh

export checkpoint_path=gs://morgandu-tpu/checkpoints/quantized/aqt/llama2-70b-chat

export quant_mode=w-i8-kv-b16
export checkpoint_is_quantized=True
export quantization=int8
export quantize_kvcache=False
export kv_quant_axis=
export kv_quant_dtype=
export per_device_batch_size=${per_device_batch_size:=14}
