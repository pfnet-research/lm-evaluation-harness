MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-ppo,use_fast=False"
TASK="jsquad-1.1-0.4,jcommonsenseqa-1.1-0.4,jnli-1.1-0.4,marc_ja-1.1-0.4"
python main.py     --model hf-causal     --model_args $MODEL_ARGS     --tasks $TASK     --num_fewshot "2,3,3,3"     --device "cuda"     --output_path "models/rinna-japanese-gpt-neox-3.6b-instruction-ppo/result.json"
