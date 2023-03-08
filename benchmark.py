# Model parameters
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import json
import requests
from rouge_score import rouge_scorer


class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained("t5-small"),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]

baseline_generator = KeyphraseGenerationPipeline(model="./models/baseline")
rouge_optimised_generator = KeyphraseGenerationPipeline(model="./models/rouge_optimised")

solr_url = "http://localhost:8983/solr"

baseline_rouges = []
rouge_optimised_rouges = []

with open("test.json", "r") as f:
  raw_test = f.read()
  raw_records = raw_test.split("\n")
  for record in raw_records[:50]:
    json_record = json.loads(record)
    # extractive_keyphrases = json_record["extractive_keyphrases"]
    # abstractive_keyphrases = json_record["abstractive_keyphrases"]
    # print("\nKeyphrases: ", "; ".join(extractive_keyphrases + abstractive_keyphrases))

    baseline_keyphrases = baseline_generator(" ".join(json_record["document"]))
    # print("\nBaseline: ", "; ".join(baseline_keyphrases[0]))

    rouge_optimised_keyphrases = rouge_optimised_generator(" ".join(json_record["document"]))
    # print("\nRouge-optimised: ", "; ".join(rouge_optimised_keyphrases[0]))
    # 
    
    # Baseline
    baseline_keyphrases = " ".join(baseline_keyphrases[0])
    url = f"{solr_url}/inspec_test/query?q=document:{baseline_keyphrases}&limit=1"
    resp = requests.get(url)
    match_docs = resp.json().get("response", {}).get("docs", [])
    if len(match_docs):
      scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
      scores = scorer.score(
        " ".join(" ".join(json_record["document"])),
        match_docs[0]["document"][0]
      )
      new_score = scores['rougeL'][2]
      baseline_rouges.append(new_score)
    
    # Rouge optimised
    rouge_optimised_keyphrases = " ".join(rouge_optimised_keyphrases[0])
    url = f"{solr_url}/inspec_test/query?q=document:{rouge_optimised_keyphrases}&limit=1"
    resp = requests.get(url)
    match_docs = resp.json().get("response", {}).get("docs", [])
    if len(match_docs):
      scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
      scores = scorer.score(
        " ".join(" ".join(json_record["document"])),
        match_docs[0]["document"][0]
      )
      new_score = scores['rougeL'][2]
      rouge_optimised_rouges.append(new_score)

  print("Baseline: ", sum(baseline_rouges) / len(baseline_rouges))
  print("Rouge optimised: ", sum(rouge_optimised_rouges) / len(rouge_optimised_rouges))

