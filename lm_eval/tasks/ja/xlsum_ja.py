"""
XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages
https://aclanthology.org/2021.findings-acl.413/

We present XLSum, a comprehensive and diverse dataset comprising 1.35 million professionally annotated article-summary pairs from BBC, extracted using a set of carefully designed heuristics. 
The dataset covers 45 languages ranging from low to high-resource, for many of which no public dataset is currently available. 
XL-Sum is highly abstractive, concise, and of high quality, as indicated by human and intrinsic evaluation. 

Homepage: https://github.com/csebuetnlp/xl-sum
"""
from rouge_score import rouge_scorer, scoring
from lm_eval.base import rf, Task


_CITATION = """
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    doi = "10.18653/v1/2021.findings-acl.413",
    pages = "4693--4703",
}
"""


class XLSumJa(Task):
    """ 
    - Use ROUGE-2 as [PaLM 2](https://ai.google/static/documents/palm2techreport.pdf)
    - Use Mecab tokenizer for Japanese eval 
    """
    VERSION = 0
    # this prompt was made by mkshing
    PROMPT_VERSION = 0.0
    DATASET_PATH = "mkshing/xlsum_ja"
    DATASET_NAME = None
    DESCRIPTION = "与えられたニュース記事を要約してください。\n\n"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from . import MecabTokenizer
        self.tokenizer = MecabTokenizer()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True
    
    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]
    
    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"ニュース記事:{doc['text']}\n要約:"

    def doc_to_target(self, doc):
        return doc["summary"]

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        continuation = results[0]
        ground_truth = doc["summary"]
        return {
            "rouge2": (
                continuation,
                ground_truth,
            )
        }
    
    def aggregation(self):
        return {
            "rouge2": self._rouge
        }

    def higher_is_better(self):
        return {
            "rouge2": True,
        }
    
    def _rouge(self, item):
        predictions, references = zip(*item)
        return self.rouge(refs=references, preds=predictions)["rouge2"]

    def rouge(self, refs, preds):
        rouge_types = ["rouge2"]
        # mecab-based rouge 
        scorer = rouge_scorer.RougeScorer(
            rouge_types,
            tokenizer=self.tokenizer,
        )

        # Accumulate confidence intervals.
        aggregator = scoring.BootstrapAggregator()
        for ref, pred in zip(refs, preds):
            aggregator.add_scores(scorer.score(ref, pred))
        result = aggregator.aggregate()
        return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


class XLSumJaWithJAAlpacaPrompt(XLSumJa):
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n\n"
    INSTRUCTION = "与えられたニュース記事を要約してください。\n\n"
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
        input_text = f"ニュース記事:{doc['text']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class XLSumJaWithRinnaInstructionSFT(XLSumJa):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられたニュース記事を要約してください。<NL>システム: 分かりました。"
    def doc_to_text(self, doc):
        input_text = f"ニュース記事:{doc['text']}"
        return f"<NL>ユーザー: {input_text}<NL>システム: "


VERSIONS = [
    XLSumJa,
    XLSumJaWithJAAlpacaPrompt,
    XLSumJaWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"xlsum_ja-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks