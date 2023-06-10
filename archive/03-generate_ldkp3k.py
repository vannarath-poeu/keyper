import argparse
import pathlib
import json
import numpy as np
import nltk
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"

def generate(phrase_file: str, core_name: str, output_path: str):
    with open(phrase_file, "r") as f:
        phrase_dic = json.load(f)

    doc_keyphrases = {}
    for id, phrase_list in phrase_dic.items():
        print(id)
        doc_keyphrases[id] = []
        for phrase in phrase_list:
            url = f"{LUCENE_URL}{core_name}/query?q=document:{phrase}"
            resp = requests.get(url)
            match_docs = resp.json().get("response", {}).get("docs", [])
            doc_id_list = [doc["id"] for doc in match_docs]
            try:
                idx = doc_id_list.index(id)
                score = (len(doc_id_list) - idx) / len(doc_id_list)
                doc_keyphrases[id].append([phrase, score])
            except ValueError:
                # Not found in top 10
                continue
    
    for doc, phrase_score in doc_keyphrases.items():
        doc_keyphrases[doc] = [item[0] for item in sorted(phrase_score, key=lambda x: x[1], reverse=True)]
    filename = phrase_file.split("/")[-1].replace("phrases", "silver")
    with open(f"{output_path}/{filename}", "w") as f:
        json.dump(doc_keyphrases, f)
        
if __name__ == "__main__":
    # Example: python3 03-generate_ldkp3k.py --phrase_file temp/midas-ldkp3k-train-phrases.json --core ldkp3k_train
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
