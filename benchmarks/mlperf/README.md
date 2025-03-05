
## Create TPU VM.
Follow these [instructions](https://cloud.google.com/tpu/docs/v5e-inference#tpu-vm) to create TPU v5e-8 VM and ssh into the VM

## Install python and create an virtual environment
```
sudo apt-get install python3-dev python3-venv -y
sudo apt-get install build-essential -y
python -m venv ~/venv/jetstream
source ~/venv/jetstream/bin/activate
```

## Install JAX on Cloud TPU VM
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Verify that JAX can access the TPU and can run basic operations:
```
$ python
>>> import jax
>>> jax.device_count()
```

##  Download MLPerf Inference Benchmark Suite and install loadgen
```
cd ~
git clone https://github.com/mlcommons/inference.git
pushd inference/loadgen
pip install . 
```

## Install eval dependencies
```
pip install \
transformers==4.31.0 \
nltk==3.8.1 \
evaluate==0.4.0 \
absl-py==1.4.0 \
rouge-score==0.1.2 \
sentencepiece==0.1.99 \
accelerate==0.21.0
```

## Download llama2-70B data file
```
export DATA_DISK_DIR=~/loadgen_run_data
mkdir -p ${DATA_DISK_DIR}
gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl ${DATA_DISK_DIR}/processed-calibration-data.pkl
gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATA_DISK_DIR}/processed-data.pkl
```

## Download Maxtext and Jetstream
```
cd ~
git clone git@github.com:AI-Hypercomputer/maxtext.git
git clone git@github.com:AI-Hypercomputer/JetStream.git
```

## Checkpoint generation

Steps to get a quantized llama2-70B checkpoint for v5e-8

Note llama2-70B model takes about 140G of memory and will not fit into a v5e-8. It must be downloaded onto a large machine (such as v5p-8) and quantized to a smaller quantized checkpoint to be loaded onto a v5e-8 machine.

* Obtain a llama2-70b checkpoint and convert it to a maxtext inference checkpoint. Please follow maxtext instructions specified here: https://github.com/google/maxtext/blob/main/getting_started/Run_Llama2.md

* Convert the checkpoint into a quantized checkpoint

To create an int8 DRQ checkpoint run the following step:

1. Define paths to load maxtext checkpoint from and save quantized checkpoint to.

```
export LOAD_PARAMS_PATH=gs://${USER}-bkt/llama2-70b-chat/param-only-decode-ckpt-maxtext/checkpoints/0/items

export SAVE_QUANT_PARAMS_PATH=gs://${USER}-bkt/quantized/llama2-70b-chat
```

2. Run the following maxtext script to generate and save an in8 quantized checkpoint

```
export TOKENIZER_PATH=maxtext/assets/tokenizer.llama2
cd maxtext && \
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${LOAD_PARAMS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-70b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11 attention=dot_product quantization=int8 save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH}
```

Your checkpoint is generated at `$SAVE_QUANT_PARAMS_PATH`. This is used to set `load_parameters_path` param below in `MAXENGINE_ARGS` env variable. 

## HF login
```
huggingface-cli login
```

## Start Jetstream server
Start Jetstream server in a terminal.
```
cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=assets/tokenizer.llama2 \
  load_parameters_path="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8_" \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  model_name=llama2-70b \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 \
  scan_layers=false \
  weight_dtype=bfloat16 \
  checkpoint_is_quantized=True \
  quantization=int8 \
  quantize_kvcache=True \
  compute_axis_order=0,2,1,3 \
  ar_cache_axis_order=0,2,1,3 \
  enable_jax_profiler=True \
  per_device_batch_size=50 \
  optimize_mesh_for_tpu_v6e=True
```

Wait until you see these server logs to indicate server is ready to process requests:
```
Memstats: After load_params:
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_0(process=0,(0,0,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_1(process=0,(1,0,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_2(process=0,(0,1,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_3(process=0,(1,1,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_4(process=0,(0,2,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_5(process=0,(1,2,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_6(process=0,(0,3,0,0))
        Using (GB) 8.1 / 31.25 (25.920000%) on TPU_7(process=0,(1,3,0,0))
WARNING:root:Initialising driver with 1 prefill engines and 1 generate engines.
2025-02-10 22:10:34,122 - root - WARNING - Initialising driver with 1 prefill engines and 1 generate engines.
WARNING:absl:T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
2025-02-10 22:10:34,152 - absl - WARNING - T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
WARNING:absl:T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
2025-02-10 22:10:34,260 - absl - WARNING - T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
WARNING:absl:T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
2025-02-10 22:10:34,326 - absl - WARNING - T5 library uses PAD_ID=0, which is different from the sentencepiece vocabulary, which defines pad_id=-1
GC tweaked (allocs, gen1, gen2):  60000 20 30
2025-02-10 22:10:36.360296: I external/xla/xla/tsl/profiler/rpc/profiler_server.cc:46] Profiler server listening on [::]:9999 selected port:9999
```

## Run server performance
```
cd ~/JetStream/benchmarks/mlperf/scripts
bash ./generate_server_performance_run.sh
```

## Run server accuracy
```
cd Google/code/llama2-70b/tpu_v5e_8_jetstream_maxtext/scripts/
bash ./generate_server_accuracy_run.sh
```

## Run server audit
```
cd Google/code/llama2-70b/tpu_v5e_8_jetstream_maxtext/scripts/
bash ./generate_server_audit_run.sh
```