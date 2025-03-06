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

So far, the experimental code only works for llama2 7b and TPU v5e-8. The whole process only takes less than 10 mins if you have a Cloud TPU v5e-8 ready.

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


### 2. Set up the LLM Server and serve request:
SSH into your Cloud TPU VM first and run the following command:

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


### 3. Offline Benchmarking:

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

### 4. Online Serving Example:

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