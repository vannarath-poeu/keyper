import argparse
import pathlib
from datasets import load_dataset
import json
from collections import Counter

from nltk.stem.porter import PorterStemmer

from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def fscore(precision: float, recall: float):
    return (2 * precision * recall) / (precision + recall)

def precision_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0
    precision = 0
    remaining_prediction = predictions[:]
    for e in labels:
        for p in remaining_prediction:
            if set(e).issubset(set(p)):
                remaining_prediction.remove(p)
                precision += 1
    precision /= len(predictions)
    return precision

def recall_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0
    recall = 0
    remaining_prediction = predictions[:]
    for e in labels:
        for p in remaining_prediction:
            if set(e).issubset(set(p)):
                remaining_prediction.remove(p)
                recall += 1
    recall /= len(labels)
    return recall

def stem_keywords(keyword_list: list):
    porter_stemmer = PorterStemmer()
    stemmed_keywords = []
    for kw in keyword_list:
        stemmed = [porter_stemmer.stem(q) for q in kw.split()]
        stemmed_keywords.append(stemmed)
    return stemmed_keywords

class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]

def keybart():
    dataset_name = "midas/ldkp100_local"
    split = "test"
    data_path = f"data/{dataset_name}/{split}.json"

    generator = KeyphraseGenerationPipeline(model="bloomberg/KeyBART")
    documents = []
    labels = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            keywords = data["keywords"]
            labels.append(keywords)
            documents.append(data["document"])

    keywords = generator(documents, truncation=True)
    precision = 0
    recall = 0
    k = 5
    for i in range(len(labels)):
        stemmed_labels = stem_keywords(labels[i])
        stemmed_preds = stem_keywords(keywords[i])
        precision += precision_at_k(stemmed_preds, stemmed_labels, k)
        recall += recall_at_k(stemmed_preds, stemmed_labels, k)
    precision = precision / len(labels)
    recall = recall / len(labels)
    print("precision", precision)
    print("recall", recall)
    print("f1", fscore(precision, recall))

    # k=5
    # precision 0.05855293573740187
    # recall 0.1527863086982366
    # f1 0.08466091509555998

    # k=10
    # precision 0.05832177531206667
    # recall 0.1539421108249125
    # f1 0.08459448625002532


if __name__ == "__main__":
    keybart()
