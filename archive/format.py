from datasets import load_dataset
from ast import literal_eval
import json
import pandas as pd

with open("rouge_train.json", "r") as f:
  data = json.load(f)

for d in data:
  d["doc_bio_tags"] = []
  d["abstractive_keyphrases"] = []
  d["other_metadata"] = {"text":[],"bio_tags":[]}

with open("rouge_train_formatted.json", "w") as f:
  f.writelines([json.dumps(record) + "\n" for record in data])