# JetStream Benchmarks And Eval

## Install Dependencies 

```
cd ~/JetStream/benchmarks
pip install -r requirements.in
```

## Benchmarks 

### Prepare DataSet

```
cd ~/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

``` 

### Run Benchmarks with maxtext tokenizer

```
python benchmarks/benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset ~/data/ShareGPT_V3_unfiltered_cleaned_split.json

``` 

### Save request outputs in Benchmarks

Please use --save-request-outputs flag to enable this feature.

```
python benchmarks/benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--save-request-outputs

```

## Eval Accuracy

Evaluate inference genereted output accuracy using saved request outputs.

```
python benchmarks/eval_accuracy.py

```