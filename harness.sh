#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="jp-eval"
#SBATCH --partition=g40
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=11G
#SBATCH --output=/fsx/home-mkshing/slurm_outs/%x_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/%x_%j.err

micromamba activate stable-neox-env
# MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b"
# MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
# MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b,use_fast=False"
# MODEL_ARGS="pretrained=cyberagent/open-calm-1b"
# MODEL_ARGS="pretrained=cyberagent/open-calm-3b"
# MODEL_ARGS="pretrained=cyberagent/open-calm-7b"
# MODEL_ARGS="pretrained=rinna/japanese-gpt2-xsmall,use_fast=False"
MODEL_ARGS="pretrained=/fsx/jp-llm/hf_model/1b-ja-230b,tokenizer=/fsx/home-mkshing/models/novelai-tokenizer,use_fast=False"
TASK="jsquad-1.1-0.2,jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "lm_outputs/stablelm-1b-ja-230b.json"

# [Example] From gpt-neox's checkpoints
# python ./deepy.py evaluate.py \
#     -d configs /fsx/jp-llm/ckpts/1b_tok=nai_data=mc4-cc100-wiki_bs=4m_tp=1_pp=1_init=wang-small-init_dtype=int64/global_step40000/configs/stable-lm-jp-1b-nai_tok-mc4_cc100_wiki.yml \
#     --eval_tasks lambada_openai_mt_ja \
#     --eval_num_fewshot 2

# srun --account="stablegpt" --partition=g40 --gpus=1 --cpus-per-gpu=12 --mem-per-cpu=11G --job-name="jp_eval" --pty bash -i
