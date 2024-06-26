# JetStream Benchmark And Eval

## Install Dependencies

```
cd ~/JetStream/benchmarks
pip install -r requirements.in
```

## Benchmark with shareGPT

### Prepare DataSet

```
cd ~/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

```

### Run Benchmark with maxtext tokenizer

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024

```

### Run Benchmark for Llama 3

```
python benchmark_serving.py \
--tokenizer <llama3 tokenizer path> \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--model llama-3

```

### Save request outputs in Benchmark

Please use `--save-request-outputs` flag to save predictions to a file.

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024  \
--save-request-outputs

```

### Automatically run evaluation after Benchmark

To automatically evaluate the outputs against the ROUGE evaluation metric, add the `--run-eval true` flag.
Note: If `--save-result` is used, the evaluation scores will be saved as well.

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024  \
--save-request-outputs \
--run-eval true

```

## Benchmark with openorca dataset (openorca is used by MLPerf inference for LLaMA2 models)
```
python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.llama2  \
--warmup-mode sampled   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca

```

## Benchmark warmup mode

The benchmark has better performance if it first conducts a warmup of the JetStream server. We currently support `sampled` and `full` warmup modes. `full` mode would warmup up the JetStream server with all the input requests. `sampled` mode would warmup up the JetStream server with a sampling of the input requests across different bucket sizes of input lengths.

Example to run benchmark with `full` warmup mode:
```
python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.llama2  \
--warmup-mode full   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca
```

## Standalone Evaluation Run

If you used `--save-request-outputs`, you can separately evaluate against the generated outputs.

```
python eval_accuracy.py outputs.json

```

With openorca dataset and llama2-chat models (used by MLPerf), here are the reference accuracy numbers:
```
llama2-7b-chat {'rouge1': 42.0706, 'rouge2': 19.8021, 'rougeL': 26.8474, 'rougeLsum': 39.5952, 'gen_len': 1146679, 'gen_num': 998}
llama2-70b-chat {'rouge1': 44.4312, 'rouge2': 22.0352, 'rougeL': 28.6162}
``` 