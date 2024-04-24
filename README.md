[![Unit Tests](https://github.com/google/JetStream/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/google/JetStream/actions/workflows/unit_tests.yaml)
[![PyPI version](https://badge.fury.io/py/google-jetstream.svg)](https://badge.fury.io/py/google-jetstream)
[![PyPi downloads](https://img.shields.io/pypi/dm/google-jetstream?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/google-jetstream/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

# JetStream is a throughput and memory optimized engine for LLM inference on XLA devices.

## About

JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (and GPUs in future -- PRs welcome).

## JetStream Engine Implementation 

Currently, there are two reference engine implementations available -- one for Jax models and another for Pytorch models.

### Jax

- Git: https://github.com/google/maxtext
- README: https://github.com/google/JetStream/blob/main/docs/online-inference-with-maxtext-engine.md

### Pytorch

- Git: https://github.com/google/jetstream-pytorch 
- README: https://github.com/google/jetstream-pytorch/blob/main/README.md 

## Documentation
- [Online Inference with MaxText on v5e Cloud TPU VM](https://cloud.google.com/tpu/docs/tutorials/LLM/jetstream) [[README](#jetstream-maxtext-inference-on-v5e-cloud-tpu-vm-user-guide)]
- [Online Inference with Pytorch on v5e Cloud TPU VM](https://cloud.google.com/tpu/docs/tutorials/LLM/jetstream-pytorch) [[README](https://github.com/google/jetstream-pytorch/tree/main?tab=readme-ov-file#jetstream-pytorch)]
- [Serve Gemma using TPUs on GKE with JetStream](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-tpu-jetstream)
- [JetStream Standalone Local Setup](#jetstream-standalone-local-setup)


# JetStream Standalone Local Setup

## Getting Started

### Setup
```
pip install -r requirements.txt
```

### Run local server & Testing

Use the following commands to run a server locally:
```
# Start a server
python -m jetstream.core.implementations.mock.server

# Test local mock server
python -m jetstream.tools.requester

# Load test local mock server
python -m jetstream.tools.load_tester

```

### Test core modules
```
# Test JetStream core orchestrator
python -m jetstream.tests.core.test_orchestrator

# Test JetStream core server library
python -m jetstream.tests.core.test_server

# Test mock JetStream engine implementation
python -m jetstream.tests.engine.test_mock_engine

# Test mock JetStream token utils
python -m jetstream.tests.engine.test_utils

```
