
# JP Language Model Evaluation Harness

## Leaderboard

| Model  | Average | [JCommonsenseQA](#jcommonsenseqa) (acc) | [JNLI](#jnli) (acc) | [MARC-ja](#marc-ja) (acc) | [JSQuAD](#jsquad) (exact_match) | eval script | Notes|
| :--: | --: | --: | --: | --: | --: | :-- | :-- |
| [rinna-japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo) | 59.63 | 41.38 | 54.03 | 89.71 | 53.42 | [models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo) |- Use v0.4 prompt template |
| [rinna-japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2) | 56.65 | 38.43 | 53.37 | 89.48 | 45.32 | [models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2) |- Use v0.4 prompt template|
| [rinna-japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft) | 53.77 | 36.55 | 42.19 | 89.02 | 47.32 | [models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft) |- Use v0.4 prompt template |
| [cyberagent-open-calm-3b](https://huggingface.co/cyberagent/open-calm-3b) | 49 | 27.79 | 40.35 | 86.21 | 41.65 | [models/cyberagent/cyberagent-open-calm-3b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/cyberagent/cyberagent-open-calm-3b) | |
| [rinna-japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b) | 47.79 | 31.64 | 34.43 | 74.82 | 50.29 | [models/rinna/rinna-japanese-gpt-neox-3.6b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b) | |
| [rinna-japanese-gpt-1b](https://huggingface.co/rinna/japanese-gpt-1b) | 47.09 | 34.76 | 37.67 | 87.86 | 28.07 | [models/rinna/rinna-japanese-gpt-1b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-1b) | |
| [cyberagent-open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b) | 46.04 | 24.22 | 37.63 | 74.12 | 48.18 | [models/cyberagent/cyberagent-open-calm-7b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/cyberagent/cyberagent-open-calm-7b) | |
| [cyberagent-open-calm-1b](https://huggingface.co/cyberagent/open-calm-1b) | 43.88 | 26.9 | 33.57 | 77.92 | 37.12 | [models/cyberagent/cyberagent-open-calm-1b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/cyberagent/cyberagent-open-calm-1b) | |
| [abeja-gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) | 37.1 | 20.02 | 39.73 | 74.99 | 13.67 | [models/abeja-gpt-neox-japanese-2.7b](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/abeja-gpt-neox-japanese-2.7b) | |



## How to evaluate your model

1. git clone https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable
    ```bash
    git clone -b jp-stable https://github.com/Stability-AI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e ".[ja]"
    ```
2. Choose your prompt template based on [docs/prompt_templates.md]((https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/prompt_templates.md))
3. Replace `TEMPLATE` to the version and change `MODEL_PATH` . And, save the script as `harness.sh`

    ```bash
    MODEL_ARGS="pretrained=MODEL_PATH"
    TASK="jsquad-1.1-TEMPLATE,jcommonsenseqa-1.1-TEMPLATE,jnli-1.1-TEMPLATE,marc_ja-1.1-TEMPLATE"
    python main.py \
        --model hf-causal \
        --model_args $MODEL_ARGS \
        --tasks $TASK \
        --num_fewshot "2,3,3,3" \
        --device "cuda" \
        --output_path "result.json"
    ```

4. Run! 
   ```bash
   sh harness.sh
   ```

We evaluated some open-sourced Japanese LMs. Pleasae refer to `harness.sh` inside `models` folder. 


## JP Tasks
For more details, please see [docs/jptasks.md](https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/jptasks.md).

| Tasks | [Supported Prompt Templates](https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/prompt_templates.md) |
| :- | -: | 
| [JSQuAD](#jsquad) | 0.1 / 0.2 / 0.3 / 0.4 |
| [JCommonsenseQA](#jcommonsenseqa) |  0.1 / 0.2 / 0.3 / 0.4 |
| [JNLI](#jnli) | 0.2 / 0.3 / 0.4 |
| [MARC-ja](#marc-ja) | 0.2 / 0.3 / 0.4 |
| [JaQuAD](#jaquad) | 0.1 / 0.2 / 0.3 / 0.4 |
| [JBLiMP](#jblimp) | - |

-----------------
# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for the Hugging Face `transformers` library, GPT-NeoX, Megatron-DeepSpeed, and the OpenAI API, with flexible tokenization-agnostic interface.
- Support for evaluation on adapters (e.g. LoRa) supported in [Hugging Face's PEFT library](https://github.com/huggingface/peft).
- Task versioning to ensure reproducibility.

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

To evaluate a model hosted on the [Hugging Face Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on tasks with names matching the pattern `lambada_*` and `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks lambada_*,hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Hugging Face, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

To use with [PEFT](https://github.com/huggingface/peft), take the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument as shown below:
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

Our library also supports the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

To address concerns about train / test contamination, we provide utilities for comparing results on a benchmark using only the data points nto found in the model training set. Unfortunately, outside of models trained on the Pile and C4, its very rare that people who train models disclose the contents of the training data. However this utility can be useful to evaluate models you have trained on private data, provided you are willing to pre-compute the necessary indices. We provide computed indices for 13-gram exact match deduplication against the Pile, and plan to add additional precomputed dataset indices in the future (including C4 and min-hash LSH deduplication).

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
