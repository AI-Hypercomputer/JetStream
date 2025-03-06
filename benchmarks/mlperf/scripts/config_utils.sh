export base_output_dir=gs://${USER}-tpu/mlperf-4.1
export experiment_time=$(date +%Y-%m-%d-%H-%M)

export tpu=v5e-16
export model_name=llama2-70b
export attention=dot_product
export reshape_q=${reshape_q:=False}
export compute_axis_order=${compute_axis_order:=0,2,1,3}
export prefill_cache_axis_order=${prefill_cache_axis_order:=0,2,1,3}
export ar_cache_axis_order=${ar_cache_axis_order:=0,2,1,3}

export config_file_path=MaxText/configs/v5e/inference/llama2_70b_v5e-16.yml
export ici_fsdp_parallelism=1
export ici_autoregressive_parallelism=${ici_autoregressive_parallelism:=2}
export ici_tensor_parallelism=${ici_tensor_parallelism:=8}
export allow_split_physical_axes=True
