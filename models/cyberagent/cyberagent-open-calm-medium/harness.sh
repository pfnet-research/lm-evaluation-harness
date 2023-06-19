MODEL_ARGS="pretrained=cyberagent/open-calm-medium,use_fast=True"
TASK="jsquad-1.1-0.2,jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2"
python main.py     --model hf-causal     --model_args $MODEL_ARGS     --tasks $TASK     --num_fewshot "2,3,3,3"     --device "cuda"     --output_path "models/cyberagent-open-calm-medium/result.json"
