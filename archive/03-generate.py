import argparse
import pathlib
import json
import numpy as np
import nltk
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"

def generate(phrase_file: str, core_name: str, output_path: str):
    doc_keyphrases = {}
    with open(phrase_file, "r") as f:
        phrase_dic = json.load(f)
        phrase_set = set()
        for _, phrase_list in phrase_dic.items():
            phrase_set.update(phrase_list)
        new_phrase_set = set([p for p in phrase_set])
        for p1 in phrase_set:
            for p2 in phrase_set:
                if p1 != p2 and p1 in p2 and p1 in new_phrase_set:
                    new_phrase_set.remove(p1)
        j = 0
        print(len(new_phrase_set))

        for phrase in new_phrase_set:
            print(j)
            j += 1
            url = f"{LUCENE_URL}{core_name}/query?q=document:{phrase}"
            resp = requests.get(url)
            match_docs = resp.json().get("response", {}).get("docs", [])
            doc_id_list = [int(doc["id"]) for doc in match_docs]
            for i, doc_id in enumerate(doc_id_list):
                if doc_id not in doc_keyphrases:
                    doc_keyphrases[doc_id] = []
                score = (len(doc_id_list) - i) / len(doc_id_list)
                doc_keyphrases[doc_id].append([phrase, score])
    
    for doc, phrase_score in doc_keyphrases.items():
        doc_keyphrases[doc] = [item[0] for item in sorted(phrase_score, key=lambda x: x[1], reverse=True)]
    with open(f"{output_path}/silver.json", "w") as f:
        json.dump(doc_keyphrases, f)
        
if __name__ == "__main__":
    # Example: python3 03-generate.py --phrase_file temp/midas-kp20k-train-phrases.json --core kp20k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--phrase_file", type=str, required=True)
    parser.add_argument("--core", type=str, required=True)
    parser.add_argument("--output", type=str, default="temp")

    args = parser.parse_args()
    # Get all the variables
    phrase_file = args.phrase_file
    core_name = args.core
    output_path = args.output

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    generate(phrase_file, core_name, output_path)
