import requests
import json
import re
from rouge_score import rouge_scorer

with open("train.json", "r") as f:
  data = json.load(f)

solr_url = "http://localhost:8983/solr"
threshold = 0.8

for record in data[800:]:
  # Preprocessing and removing non-character including punctuations (except for -)
  doc = " ".join(record["document"])
  doc = re.sub(r'[^a-zA-Z- ]', '', doc)
  doc = doc.split(" ")
  last_word = ""
  last_score = 0
  word_set = {}
  for word in doc:
    word = word.lower()
    url = f"{solr_url}/inspec_training/query?q=document:{word}&limit=1"
    resp = requests.get(url)
    match_docs = resp.json().get("response", {}).get("docs", [])
    if len(match_docs):
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
      scores = scorer.score(
        " ".join(doc),
        match_docs[0]["document"][0]
      )
      word_set[word] = scores['rougeL'][2]
    
    url = f"{solr_url}/inspec_training/query?q=document:{last_word}{word}&limit=1"
    resp = requests.get(url)
    match_docs = resp.json().get("response", {}).get("docs", [])
    if len(match_docs):
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
      scores = scorer.score(
        " ".join(doc),
        match_docs[0]["document"][0]
      )
      new_score = scores['rougeL'][2]
      if new_score > last_score:
        word_set[f"{last_word}{word}"] = new_score
        # Add space
        last_word = f"{last_word}{word} "
        last_score = new_score
      else:
        last_word = ""
        last_score = 0
  word_set = dict(sorted(word_set.items(), key=lambda x: x[1], reverse=True))
  record["extractive_keyphrases"] = [x[0] for x in word_set.items() if x[1] > threshold]

with open("rouge_train.json", "w") as f:
  json.dump(data, f, indent=4)