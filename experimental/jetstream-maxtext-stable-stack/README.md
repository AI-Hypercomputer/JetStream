# Jetstream MaxText Stable Stack

This provides a stable Docker image stack for running MaxText using JetStream on Cloud TPUs for inference.

## Overview

The goal of this project is to offer a reliable and up-to-date environment for deploying and serving MaxText efficiently on TPU hardware via the JetStream inference server.

## Getting Started

### Prerequisites

- Docker installed on your machine or VM.
- Access to Google Cloud Platform and authenticated `gcloud` CLI (if pulling from GCR).
- Access to TPU resources configured for your project.

### Pulling the Image

The stable stack is available as a nightly Docker image hosted on Google Container Registry (GCR). To pull the latest nightly image, replace `YYYYMMDD` with the desired date (e.g., `20231027`):

```bash
# Replace YYYYMMDD with the specific date, e.g., 20231027
export NIGHTLY_DATE=$(date +"%Y%m%d") # Or set manually, e.g., export NIGHTLY_DATE=20231027

docker pull gcr.io/cloud-tpu-inference-test/jetstream-maxtext-stable-stack/tpu:nightly-${NIGHTLY_DATE}

# Or the last nightly build
docker pull gcr.io/cloud-tpu-inference-test/jetstream-maxtext-stable-stack/tpu:nightly
```

## Running the Container

Run on the TPU VM.

```bash
docker run --net=host --privileged --rm -it \
  # Add necessary volume mounts, TPU device access, network ports, etc.
  gcr.io/cloud-tpu-inference-test/jetstream-maxtext-stable-stack/tpu:nightly \
  bash
```

## Image Information

- Registry: Google Container Registry (GCR)
- Path: gcr.io/cloud-tpu-inference-test/jetstream-maxtext-stable-stack/tpu
- Tagging Scheme: nightly-YYYYMMDD (e.g., nightly-20231027)

A new image is built nightly, incorporating the latest updates and dependencies for the JetStream-MaxText stack on TPUs. Use the tag corresponding to the date you wish to use.

## Build the Image

- build.sh build the local docker image
- test.sh test all the .sh in test_script using the built image
- pipeline.sh build, test and upload the image if all success.

```bash
./pipeline.sh UPLOAD_IMAGE_TAG=gcr.io/cloud-tpu-inference-test/jetstream-maxtext-stable-stack/tpu:nightly-$(date +"%Y%m%d")
```
