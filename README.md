# JetStream - A throughput and memory optimized engine for LLM inference on TPUs

## About

JetStream is a fast library for LLM inference and serving on TPUs.

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
python -m jetstream.core.orchestrator_test

# Test JetStream core server library
python -m jetstream.core.server_test

# Test mock JetStream engine implementation
python -m jetstream.engine.mock_engine_test

# Test mock JetStream token utils
python -m jetstream.engine.utils_test

```
