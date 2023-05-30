"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. 
JGLUE has been constructed from scratch without translation. 

Homepage: https://github.com/yahoojapan/JGLUE
"""
from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""



class JNLIWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.2
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JNLI"
    DESCRIPTION = "前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。\n\n" + \
        "制約:\n" + \
        "- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n" + \
        "- 前提と仮説が両立しえない場合はcontradictionと出力\n" + \
        "- そのいずれでもない場合はneutralと出力\n\n"
    CHOICES = ["entailment", "contradiction", "neutral"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "premise": doc["sentence1"],
            "hypothesis": doc["sentence2"],
            "choices": self.CHOICES,
            "gold": int(doc["label"]), 
        }

    def doc_to_text(self, doc):
        """
        前提:{premise}
        仮説:{hypothesis}
        関係:
        """
        return (
            f"前提:{doc['premise']}\n"
            f"仮説:{doc['hypothesis']}\n"
            "関係:"
        )

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls



class JNLIWithJAAlpacaPrompt(JNLIWithFintanPrompt):
    """
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n\n"
    INSTRUCTION = f"与えられた前提と仮説の関係を回答してください。\n\n出力は以下から選択してください：\n" + "\n".join(JNLIWithFintanPrompt.CHOICES)

    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示: 
        {instruction}

        ### 入力: 
        {input}

        ### 応答: 
        {response}
        """
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"



class JNLIWithRinnaInstructionSFT(JNLIWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: " + f"与えられた前提と仮説の関係を回答してください。出力は以下から選択してください：<NL>" + "<NL>".join(JNLIWithFintanPrompt.CHOICES) + "<NL>システム: 分かりました。"
    def doc_to_text(self, doc):
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"<NL>ユーザー: {input_text}<NL>システム: "
   


VERSIONS = [
    JNLIWithFintanPrompt,
    JNLIWithJAAlpacaPrompt,
    JNLIWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jnli-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
