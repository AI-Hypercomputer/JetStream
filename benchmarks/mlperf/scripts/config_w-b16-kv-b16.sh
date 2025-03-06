
source config_utils.sh

export checkpoint_path=gs://runner-maxtext-logs/2024-05-07-23-34/unscanned_chkpt/checkpoints/0/items

export quant_mode=w-b16-kv-b16
export checkpoint_is_quantized=False
export quantization=
export quantize_kvcache=False
export kv_quant_axis=
export kv_quant_dtype=
export per_device_batch_size=${per_device_batch_size:=6}
