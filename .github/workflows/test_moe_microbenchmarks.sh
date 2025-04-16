#!/bin/bash
mkdir ~/test_dir
cd ~/test_dir
git clone https://github.com/google/maxtext.git

cd ~/test_dir
git clone https://github.com/google/JetStream.git
cd ~/test_dir
sudo apt-get -y update
sudo apt-get -y install python3.10-venv
sudo apt-get -y install jq
python -m venv .env
source .env/bin/activate

cd ~/test_dir
cd JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in

cd ~/test_dir
cd maxtext/
pip3 install wheel
bash setup.sh MODE=stable DEVICE=tpu

pip install nltk==3.8.1


# moe 8x7b microbenchmark
LIBTPU_INIT_ARGS="--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000" python -m MaxText.inference_microbenchmark MaxText/configs/inference.yml tokenizer_path=assets/tokenizer.mistral-v1 max_prefill_predict_length=1024 max_target_length=2048 model_name=mixtral-8x7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=1 ici_context_autoregressive_parallelism=8 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=8 megablox=False quantization=int8 quantize_kvcache=False checkpoint_is_quantized=True load_parameters_path=gs://jetstream-runner/8-7B-int8 capacity_factor=1 attention=dot_product model_call_mode=inference sparse_matmul=False weight_dtype=bfloat16 > ~/test_dir/moe_8x7b.txt
tail -n5 ~/test_dir/moe_8x7b.txt > ~/test_dir/moe_8x7b.tmp && mv ~/test_dir/moe_8x7b.tmp ~/test_dir/moe_8x7b.txt

# moe 8x22B microbenchmark
LIBTPU_INIT_ARGS="--xla_tpu_enable_windowed_einsum_for_reduce_scatter=false --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000" python -m MaxText.inference_microbenchmark MaxText/configs/inference.yml load_parameters_path=gs://jetstream-runner/8-22B-int8  max_prefill_predict_length=1024 max_target_length=2048 model_name=mixtral-8x22b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=1 ici_context_autoregressive_parallelism=8 scan_layers=false per_device_batch_size=24 attention=dot_product megablox=False quantization=int8 checkpoint_is_quantized=True quantize_kvcache=True capacity_factor=1 tokenizer_path=assets/tokenizer.mistral-v3 inference_microbenchmark_prefill_lengths="128,1024" sparse_matmul=False model_call_mode=inference > ~/test_dir/moe_8x22b.txt
tail -n5 ~/test_dir/moe_8x22b.txt > ~/test_dir/moe_8x22b.tmp && mv ~/test_dir/moe_8x22b.tmp ~/test_dir/moe_8x22b.txt