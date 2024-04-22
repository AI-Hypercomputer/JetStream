# JetStream Benchmark And Eval

## Install Dependencies

```
cd ~/JetStream/benchmarks
pip install -r requirements.in
```

## Benchmark

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

## Standalone Evaluation Run

If you used `--save-request-outputs`, you can separately evaluate against the generated outputs.

```
python eval_accuracy.py

```