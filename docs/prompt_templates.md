# Prompt Templates
For JGLUE metrics, you can choose suitable prompt template for your model. 

Once you found the best one of the following supported templates, replace `TEMPLATE` to the template version. 

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

## `0.1`
- **Reference:** [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
- **Supported Tasks:** `jsquad`, `jaquad`, `jcommonsenseqa`
- **Format:**
  e.g. JCommonsenseQA
  ```
  [問題]に対する[答え]を[選択肢]の中から選んでください。

  [問題]:{question}
  [選択肢]:[{choice0}, {choice1}, ..., {choice4}]
  [答え]:{answer}
  ```
  For formats for other tasks, please see `lm_eval/tasks/TASK.py`.

## `0.2`
- **Reference:** [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
- **Supported Tasks:** `jsquad`, `jaquad`, `jcommonsenseqa`, `jnli`, `marc_ja`
- **Format:**
  e.g. JCommonsenseQA
  ```
  質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号(例:0)でするものとします。 

  質問:{question}
  選択肢:0.{choice0},1.{choice1}, ...,4.{choice4}
  回答:{index of answer}
  ```
  For formats for other tasks, please see `lm_eval/tasks/TASK.py`.


## `0.3`
This is intended to use for instruction-tuned models trained on [Japanese Alpaca](https://huggingface.co/datasets/fujiki/japanese_alpaca_data)

- **Reference:** 
  - [masa3141 /
japanese-alpaca-lora
](https://github.com/masa3141/japanese-alpaca-lora)
  - https://github.com/Stability-AI/gpt-neox/blob/bed0b5aa66142aa649299b76d4e3948efccd0bf4/finetune/templates.py
- **Supported Tasks:** `jsquad`, `jaquad`, `jcommonsenseqa`, `jnli`, `marc_ja`
- **Format:**
  e.g. JCommonsenseQA
  ```
  以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
  
  ### 指示: 
  与えられた選択肢の中から、最適な答えを選んでください。

  出力は以下から選択してください：
  - {choice0}
  - {choice1}
  ...
  - {choice4}

  ### 入力: 
  {question}

  ### 応答: 
  {answer}
  ```
  For formats for other tasks, please see `lm_eval/tasks/TASK.py`.


## `0.4`
This is intended to use for [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft).


- **Reference:** [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)
- **Supported Tasks:** `jsquad`, `jaquad`, `jcommonsenseqa`, `jnli`, `marc_ja`
- **Format:**
  e.g. JCommonsenseQA
  ```
  ユーザー: 与えられた選択肢の中から、最適な答えを選んでください。<NL>システム: 分かりました。
  
  <NL>ユーザー: 質問：{question}<NL>選択肢：<NL>- {choice0}<NL>- {choice1}<NL>...<NL>- {choice4}<NL>
  
  <NL>システム: {answer}
  ```
  For formats for other tasks, please see `lm_eval/tasks/TASK.py`.
