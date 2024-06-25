# Observability in JetStream Server

In JetStream Server, we use [Prometheus](https://prometheus.io/docs/introduction/overview/) to collect key metrics within JetStream orchestrator and engines. We implemented a [Prometheus client server](https://prometheus.github.io/client_python/exporting/http/) in JetStream `server_lib.py` and use `MetricsServerConfig` (by passing `prometheus_port` in server entrypoint) to gaurd the metrics observability feature.

## Enable Prometheus server to observe Jetstream metrics

Metrics are not exported by default, here is an example to run JetStream MaxText server with metrics observability:

```bash
# Refer to JetStream MaxText User Guide for the following server config.
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
# Set PROMETHEUS_PORT to enable Prometheus metrics.
export PROMETHEUS_PORT=9090

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
  prometheus_port=${PROMETHEUS_PORT}
```

Now that we configured `prometheus_port=9090` above, we can observe various Jetstream metrics via HTTP requests to `0.0.0.0:9000`. Towards the end, the response should have content similar to the following:

```
# HELP jetstream_prefill_backlog_size Size of prefill queue
# TYPE jetstream_prefill_backlog_size gauge
jetstream_prefill_backlog_size{id="SOME-HOSTNAME-HERE>"} 0.0
# HELP jetstream_slots_used_percentage The percentage of decode slots currently being used
# TYPE jetstream_slots_used_percentage gauge
jetstream_slots_used_percentage{id="<SOME-HOSTNAME-HERE>",idx="0"} 0.04166666666666663
```