
# JP Tasks 

## [JGLUE](https://github.com/yahoojapan/JGLUE)
### JSQuAD
> JSQuAD is a Japanese version of [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar+, 2016), one of the datasets of reading comprehension.
Each instance in the dataset consists of a question regarding a given context (Wikipedia article) and its answer. JSQuAD is based on SQuAD 1.1 (there are no unanswerable questions). We used [the Japanese Wikipedia dump](https://dumps.wikimedia.org/jawiki/) as of 20211101.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jsquad-1.1-0.2" \
    --num_fewshot "2" \
    --output_path "result.json"
```

### JCommonsenseQA
> JCommonsenseQA is a Japanese version of [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) (Talmor+, 2019), which is a multiple-choice question answering dataset that requires commonsense reasoning ability. It is built using crowdsourcing with seeds extracted from the knowledge base [ConceptNet](https://conceptnet.io/).

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jcommonsenseqa-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

### JNLI
> JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. NLI is a task to recognize the inference relation that a premise sentence has to a hypothesis sentence. The inference relations are `entailment`, `contradiction`, and `neutral`.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jnli-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

### MARC-ja
> MARC-ja is a dataset of the text classification task. This dataset is based on the Japanese portion of [Multilingual Amazon Reviews Corpus (MARC)](https://docs.opendata.aws/amazon-reviews-ml/readme.html) (Keung+, 2020).

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "marc_ja-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

## [JaQuAD](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD)

> Japanese Question Answering Dataset (JaQuAD), released in 2022, is a human-annotated dataset created for Japanese Machine Reading Comprehension. JaQuAD is developed to provide a SQuAD-like QA dataset in Japanese. 

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jaquad-1.1-0.2" \
    --num_fewshot "2" \
    --output_path "result.json"
```

## [JBLiMP](https://github.com/osekilab/JBLiMP)

> JBLiMP is a novel dataset for targeted syntactic evaluations of language models in Japanese. JBLiMP consists of 331 minimal pairs, which are created based on acceptability judgments extracted from journal articles in theoretical linguistics. These minimal pairs are grouped into 11 categories, each covering a different linguistic phenomenon.

**NOTE:** JBLiMP is not used in official evaluations because it is too small compared to other datasets.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jblimp" \
    --num_fewshot "0" \
    --output_path "result.json"
```
