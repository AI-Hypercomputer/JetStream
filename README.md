# JetStream - A throughput and memory optimized engine for LLM inference on TPU and GPU

## About

JetStream is a fast library for LLM inference and serving on TPU and GPU.

## Getting Started

### Run local server & Testing

Use the following commands to run a server locally:
```
# Start a server
python -m jetstream.core.implementations.mock.server

# Test local mock server
python -m jetstream.core.tools.requester

# Load test local mock server
python -m jetstream.core.tools.load_tester

```

### Test core modules
```
# Test JetStream core orchestrator
python -m jetstream.core.orchestrator_test

# Test JetStream core server library
python -m jetstream.core.server_test

# Test mock JET engine implementation
python -m jetstream.engine.mock_engine_test

# Test mock JET engine implementation
python -m jetstream.engine.utils_test

```
