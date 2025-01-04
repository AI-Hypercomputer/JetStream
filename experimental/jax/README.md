# An experimental JAX inference framework for prototyping new ideas.

## About

 It has the following characteristics (some of them are limited version):

```
  Performance:
    1. Paged Attention
    2. Chunked Prefill and Piggybacking Decode
    3. Collective Matmul
    4. Async device program schedule and postprocess

  Framework:
    1. Pythonic model builder
    2. JAX manual sharding
    3. Interface for different hardware supports
    4. On-the-flying HF model conversion and deployment
```

## Quick Start

So far, the experimental code only works for llama2 7b and TPU v5e-8.

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


### 2. Run LLM Server and Send request:



### 3. Offline Benchmarking:

Benchmarking result:
