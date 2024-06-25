# Profiling in JetStream Server

In JetStream server, we have implemented JAX profiler server to support profiling JAX program with tensorboard.

## Profiling with JAX profiler server and tenorboard server

Following the [JAX official manual profiling approach](https://jax.readthedocs.io/en/latest/profiling.html#manual-capture-via-tensorboard), here is an example of JetStream MaxText server profiling with tensorboard:

1. Start a TensorBoard server:
```bash
tensorboard --logdir /tmp/tensorboard/
```
You should be able to load TensorBoard at http://localhost:6006/. You can specify a different port with the `--port` flag. If you are running on a remote Cloud TPU VM, the `tensorboard-plugin-profile` python package enables remote access to tensorboard endpoints (JetStream deps include this package).

When you can not access the tensorboard and the profiling code is run remotely, please run below command setup an SSH tunnel on port 6006 to work. If you run with vs code remote debug commandline, the vs code did ssh forward port for you.

```bash
 gcloud compute ssh <machine-name> -- -L 6006:127.0.0.1:6006
 ```


2. Start JetStream MaxText server:
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
# Set ENABLE_JAX_PROFILER to enable JAX profiler server at port 9999.
export ENABLE_JAX_PROFILER=true
# Set JAX_PROFILER_PORT to customize JAX profiler server port.
export JAX_PROFILER_PORT=9999

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
  enable_jax_profiler=${ENABLE_JAX_PROFILER} \
  jax_profiler_port=${JAX_PROFILER_PORT}
```

3. Open http://localhost:6006/#profile, and click the “CAPTURE PROFILE” button in the upper left. Enter “localhost:9999” as the profile service URL (this is the address of the profiler server you started in the previous step). Enter the number of milliseconds you’d like to profile for, and click “CAPTURE”.

4. After the capture finishes, TensorBoard should automatically refresh. (Not all of the TensorBoard profiling features are hooked up with JAX, so it may initially look like nothing was captured.) On the left under “Tools”, select `trace_viewer`.