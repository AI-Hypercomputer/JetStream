# JetStream MaxText Inference on v5e Cloud TPU VM User Guide

## Outline


1. Prerequisites: Prepare your GCP project and connect to Cloud TPU VM
2. Download the JetStream and MaxText github repository
3. Setup your MaxText JetStream environment
4. Convert Model Checkpoints
5. Run the JetStream MaxText server
6. Send a test request to the JetStream MaxText server
7. Run benchmarks with the JetStream MaxText server
8. Clean up


## Prerequisites: Prepare your GCP project and connect to Cloud TPU VM

Follow the steps in [Manage TPU resources | Google Cloud](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm) to create a Cloud TPU VM (Recommend TPU type: `v5litepod-8`) and connect to the Cloud TPU VM.


## Step 1: Download JetStream and the MaxText github repository

```bash
git clone -b jetstream-v0.2.2 https://github.com/google/maxtext.git
git clone -b v0.2.2 https://github.com/google/JetStream.git
```

## Step 2: Setup MaxText

```bash
# Create a python virtual environment for the demo.
sudo apt install python3.10-venv
python -m venv .env
source .env/bin/activate

# Setup MaxText.
cd maxtext/
bash setup.sh
```

## Step 3: Convert Model Checkpoints 

You can run the JetStream MaxText Server with Gemma and Llama2 models. This section describes how to run the JetStream MaxText server with various sizes of these models.

### Use a Gemma model checkpoint

*   You can download a [Gemma checkpoint from Kaggle](https://www.kaggle.com/models/google/gemma/frameworks/maxText/variations/7b). 
*   After downloading checkpoints, copy them to your GCS bucket at `$CHKPT_BUCKET`.
    *   `gsutil -m cp -r ${YOUR_CKPT_PATH} ${CHKPT_BUCKET}`
    *   Please refer to the [conversion script](https://github.com/google/JetStream/blob/main/jetstream/tools/maxtext/model_ckpt_conversion.sh) for an example of `$CHKPT_BUCKET`.
*   Then, using the following command to convert the Gemma checkpoint into a MaxText compatible unscanned checkpoint.

```bash
# bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh ${MODEL} ${MODEL_VARIATION} ${CHKPT_BUCKET}

# For gemma-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh gemma 7b ${CHKPT_BUCKET}
```

Note: For more information about the Gemma model and checkpoints, see [About Gemma](https://github.com/google/maxtext/blob/main/end_to_end/gemma/Run_Gemma.md).


### Use a Llama2 model checkpoint

*   You can use a Llama2 checkpoint you have generated or one from [the open source community](https://llama.meta.com/llama-downloads/). 
*   After downloading checkpoints, copy them to your GCS bucket at `$CHKPT_BUCKET`.
    *   `gsutil -m cp -r ${YOUR_CKPT_PATH} ${CHKPT_BUCKET}`
    *   Please refer to the [conversion script](https://github.com/google/JetStream/blob/main/jetstream/tools/maxtext/model_ckpt_conversion.sh) for an example of `$CHKPT_BUCKET`.
*   Then, using the following command to convert the Llama2 checkpoint into a MaxText compatible unscanned checkpoint.

```bash
# bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh ${MODEL} ${MODEL_VARIATION} ${CHKPT_BUCKET}

# For llama2-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 7b ${CHKPT_BUCKET}

# For llama2-13b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 13b ${CHKPT_BUCKET}
```

Note: For more information about the Llama2 model and checkpoints, see [About Llama2](https://github.com/google/maxtext/blob/main/getting_started/Run_Llama2.md).


## Step4: Run the JetStream MaxText server


### Create model config environment variables for server flags

You can export the following environment variables based on the model you used.

*   You can copy and export the `UNSCANNED_CKPT_PATH` from the model\_ckpt\_conversion.sh output.


#### Create Gemma-7b environment variables for server flags



*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.gemma
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=gemma-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
```

#### Create Llama2-7b environment variables for server flags

*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
```

#### Create Llama2-13b environment variables for server flags



*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-13b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=4
```

### Run the following command to start the JetStream MaxText server

```bash
cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

### JetStream MaxText Server flag descriptions:



*   tokenizer\_path: file path to a tokenizer (should match your model)
*   load\_parameters\_path: Loads the parameters (no optimizer states) from a specific directory
*   per\_device\_batch\_size: decoding batch size per device (1 TPU chip = 1 device)
*   max\_prefill\_predict\_length: Maximum length for the prefill when doing autoregression
*   max\_target\_length: Maximum sequence length
*   model\_name: Model name
*   ici\_fsdp\_parallelism: The number of shards for FSDP parallelism
*   ici\_autoregressive\_parallelism: The number of shards for autoregressive parallelism
*   ici\_tensor\_parallelism: The number of shards for tensor parallelism
*   weight\_dtype: Weight data type (e.g. bfloat16)
*   scan\_layers: Scan layers boolean flag (set to `false` for inference)

Note: these flags are from [MaxText config](https://github.com/google/maxtext/blob/f9e04cdc1eec74a0e648411857c09403c3358461/MaxText/configs/base.yml)


## Step 5: Send test request to JetStream MaxText server

```bash
cd ~
# For Gemma model
python JetStream/jetstream/tools/requester.py --tokenizer maxtext/assets/tokenizer.gemma
# For Llama2 model
python JetStream/jetstream/tools/requester.py --tokenizer maxtext/assets/tokenizer.llama2
```

The output will be similar to the following:

```bash
Sending request to: 0.0.0.0:9000
Prompt: Today is a good day
Response:  to be a fan
```

## Step 6: Run benchmarks with JetStream MaxText server

Note: The JetStream MaxText Server is not running with quantization optimization in Step 3. To get best benchmark results, we need to enable quantization (Please use AQT trained or fine tuned checkpoints to ensure accuracy) for both weights and KV cache, please add the quantization flags and restart the server as following:

```bash
# Enable int8 quantization for both weights and KV cache
export QUANTIZATION=int8
export QUANTIZE_KVCACHE=true

# For Gemma 7b model, change per_device_batch_size to 12 to optimize performance. 
export PER_DEVICE_BATCH_SIZE=12

cd ~/maxtext
python MaxText/maxengine_server.py \
MaxText/configs/base.yml \
tokenizer_path=${TOKENIZER_PATH} \
load_parameters_path=${LOAD_PARAMETERS_PATH} \
max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
max_target_length=${MAX_TARGET_LENGTH} \
model_name=${MODEL_NAME} \
ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
scan_layers=${SCAN_LAYERS} \
weight_dtype=${WEIGHT_DTYPE} \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
quantization=${QUANTIZATION} \
quantize_kvcache=${QUANTIZE_KVCACHE}
```

### Benchmarking Gemma-7b

Instructions
- Download the ShareGPT dataset
- Make sure to use the Gemma tokenizer (tokenizer.gemma) when running Gemma 7b.
- Add `--warmup-first` flag for your 1st run to warmup the server

```bash
# Activate the python virtual environment we created in Step 2.
cd ~
source .env/bin/activate

# download dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# run benchmark with the downloaded dataset and the tokenizer in maxtext
# You can control the qps by setting `--request-rate`, the default value is inf.
python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.gemma \
--num-prompts 1000 \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-first true
```

### Benchmarking Llama2-\*b

```bash
# Same as Gemma-7b except for the tokenizer (must use a tokenizer that matches your model, which should now be tokenizer.llama2). 

python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.llama2 \
--num-prompts 1000  \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-first true
```

## Clean Up

```bash
# Clean up gcs buckets.
gcloud storage buckets delete ${MODEL_BUCKET}
gcloud storage buckets delete ${BASE_OUTPUT_DIRECTORY}
gcloud storage buckets delete ${DATASET_PATH}
# Clean up repositories.
rm -rf maxtext
rm -rf JetStream
# Clean up python virtual environment
rm -rf .env
```
