# An experimental JAX inference framework for prototyping new ideas.

## About

 It has the following features (some of them are limited version):

```
  Performance:
    1. Paged Attention
    2. Chunked Prefill and Piggybacking Decode
    3. Collective Matmul

  Framework:
    1. Pythonic model builder
    2. JAX manual sharding
    3. Interface for different hardware supports
    4. On-the-flying HF model conversion and deployment
```

## Quick Start

The experimental code supports llama2 7b and can run on TPUs (tested with v5e-8) and NVIDIA GPUs. The setup process varies slightly depending on your hardware.

### 1. Create Cloud TPU v5e-8 on Google Cloud:

```
gcloud alpha compute tpus queued-resources create ${QR_NAME} \
    --node-id ${NODE_NAME} \
    --project ${PROJECT_ID} \
    --zone ${ZONE} \
    --accelerator-type v5litepod-8 \
    --runtime-version v2-alpha-tpuv5-lite 
```

For more [information](https://cloud.google.com/tpu/docs/queued-resources)

### 2. Set up for GPU:

Ensure you have a compatible NVIDIA GPU, CUDA Toolkit, and cuDNN installed. Then, update your Python environment with JAX compiled for CUDA. Modify `experimental/jax/requirements.txt` to change `jax[tpu]` to `jax[cuda-pip]` (or the specific CUDA version you need, e.g., `jax[cuda12_pip]`) and reinstall the requirements:

```bash
# (Activate your virtual environment first)
# Modify requirements.txt as described above by changing the jax[tpu] line to:
# jax[cuda-pip]==0.4.33 # Or your specific CUDA version, e.g., jax[cuda12_pip]
pip install -r experimental/jax/requirements.txt
```

The rest of the setup, including cloning the repository and logging into Hugging Face (see step 3 below), is the same as for TPU.

### 3. Set up the LLM Server and serve request:
SSH into your Cloud TPU VM or your GPU machine first and run the following command:

Set up a new Python env.
```
virtualenv jax-inference
source jax-inference/bin/activate
```

Clone the repo and install the dependencies.
```
git clone https://github.com/AI-Hypercomputer/JetStream.git

cd JetStream/experimental/jax

pip install -r requirements.txt
```

Log in to the Hugging Face (make sure your account has the permission to access `meta-llama/Llama-2-7b-chat-hf`)

```
huggingface-cli login
```


### 4. Offline Benchmarking:

Note: the current setup is using 8-ways TP which is just for experiment and compare with current JetStream + MaxText number.

```
export PYTHONPATH=$(pwd)
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
python inference/entrypoint/mini_offline_benchmarking.py
```

Offline Benchmarking result:

This number is around `45%` better than the current MaxText and JetStream (as of 2024/08/16) number in the same situation.


```
Benchmarking result:
  Total requests: 1000
  Total input tokens: 218743
  Total output tokens: 291740
  Input token throughput: 2980.654636529649 tokens/sec
  Output token throughput: 3975.332621666338 tokens/sec
```

Note: The online number should be even more better than the current MaxText and JetStream as the experimental framework runs the prefill and decode together in one model forward pass.

### 5. Online Serving Example:

Start server:

```
python inference/entrypoint/run_simple_server.py &
```

Send request:

```
curl --no-buffer -H 'Content-Type: application/json' \
  -d '{ "prompt": "Today is a good day" }' \
  -X POST \
  localhost:8000/generate
```