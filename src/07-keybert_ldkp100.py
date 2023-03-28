import argparse
import pathlib
from datasets import load_dataset
from keybert import KeyBERT
import json
from collections import Counter



SEED = 100

def keybert():
    dataset_name = "midas/ldkp100"
    split = "train"
    data_path = f"data/{dataset_name}/{split}.json"
    kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')
    # whole_document = []
    with open(data_path, 'r') as f:
        result = []
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            doc_id = data["id"]
            section_list = data["sections"]
            section_text_list = data["sec_text"]
            extractive_keyphrases = data["extractive_keyphrases"]
            abstractive_keyphrases = data["abstractive_keyphrases"]
            docs = [" ".join(section_text) for section_text in section_text_list]
            section_keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.5, top_n=10)
            result.append({
                "id": doc_id,
                "section_list": section_list,
                "section_keywords": section_keywords,
                "extractive_keyphrases": extractive_keyphrases,
                "abstractive_keyphrases": abstractive_keyphrases,
            })
    with open("temp/midas-ldkp100-train-phrases.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    keybert()
