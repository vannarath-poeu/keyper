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
    if precision == 0 and recall == 0:
        return 0.0
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
    dataset_name = "midas/ldkp100"
    split = "test"
    data_path = f"data/{dataset_name}/{split}.json"

    generator = KeyphraseGenerationPipeline(model="bloomberg/KeyBART")

    with open(data_path, 'r') as f:
        extractive_precision = 0
        extractive_recall = 0
        abstractive_precision = 0
        abstractive_recall = 0
        k = 10
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            doc_id = data["id"]
            section_list = data["sections"]
            section_text_list = data["sec_text"]
            docs = [" ".join(section_text) for section_text in section_text_list]
            doc_phrases = generator(docs, truncation=True)
            new_doc = []
            for section_phrases in doc_phrases:
                new_section = " ".join(section_phrases)
                new_doc.append(new_section)
            keywords = generator("\n".join(new_doc))
            extractive_keyphrases = data["extractive_keyphrases"]
            abstractive_keyphrases = data["abstractive_keyphrases"]
            stemmed_extractive_keyphrases = stem_keywords(extractive_keyphrases)
            stemmed_abstractive_keyphrases = stem_keywords(abstractive_keyphrases)
            stemmed_preds = stem_keywords(keywords[0])

            extractive_precision += precision_at_k(stemmed_preds, stemmed_extractive_keyphrases, k)
            extractive_recall += recall_at_k(stemmed_preds, stemmed_extractive_keyphrases, k)

            abstractive_precision += precision_at_k(stemmed_preds, stemmed_abstractive_keyphrases, k)
            abstractive_recall += recall_at_k(stemmed_preds, stemmed_abstractive_keyphrases, k)
    
    extractive_precision = extractive_precision / i
    extractive_recall = extractive_recall / i
    print("extractive")
    print("precision", extractive_precision)
    print("recall", extractive_recall)
    print("f1", fscore(extractive_precision, extractive_recall))

    abstractive_precision = abstractive_precision / i
    abstractive_recall = abstractive_recall / i
    print("abstractive")
    print("precision", abstractive_precision)
    print("recall", abstractive_recall)
    print("f1", fscore(abstractive_precision, abstractive_recall))

    # k=5
    # extractive
    # precision 0.21801346801346796
    # recall 0.2036574899319997
    # f1 0.21059110109610726
    # abstractive
    # precision 0.023400673400673398
    # recall 0.030303030303030304
    # f1 0.026408283461574995

    # k = 10
    # extractive
    # precision 0.2183501683501683
    # recall 0.2061827424572522
    # f1 0.21209209171000035
    # abstractive
    # precision 0.023400673400673398
    # recall 0.030303030303030304
    # f1 0.026408283461574995


if __name__ == "__main__":
    keybart()
