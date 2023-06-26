import argparse
import pathlib
import json
import numpy as np
import nltk
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"

def generate(phrase_file: str, core_name: str, output_path: str):
    with open(phrase_file, "r") as f:
        items = json.load(f)

    for item in items:
        doc_id = item["id"]
        for i, section in enumerate(item["section_list"]):
            print(i)
            keywords = item["section_keywords"][i]
            new_keywords = []
            for j, keyword in enumerate(keywords):
                phrase, keybert_score = keyword
                url = f"{LUCENE_URL}{core_name}/query?q= doc_id:{doc_id} AND document:{phrase}"
                resp = requests.get(url)
                match_docs = resp.json().get("response", {}).get("docs", [])
                doc_title_list = [doc["title"][0] for doc in match_docs if "title" in doc]
                try:
                    idx = doc_title_list.index(section)
                    score = (len(doc_title_list) - idx) / len(doc_title_list)
                    new_keywords.append((phrase, keybert_score * score))
                except ValueError:
                    # Not found in top 10
                    print("not found")
                    continue
            item["section_keywords"][i] = sorted(new_keywords, key=lambda x: x[1], reverse=True)

    filename = phrase_file.split("/")[-1].replace("phrases", "silver")
    with open(f"{output_path}/{filename}", "w") as f:
        json.dump(items, f)
        
if __name__ == "__main__":
    # Example: python3 03-generate_ldkp3k.py --phrase_file temp/midas-ldkp3k-train-phrases.json --core ldkp3k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--phrase_file", type=str, default="temp/midas-ldkp100-train-phrases.json")
    parser.add_argument("--core", type=str, default="ldkp100_train")
    parser.add_argument("--output", type=str, default="temp")

    args = parser.parse_args()
    # Get all the variables
    phrase_file = args.phrase_file
    core_name = args.core
    output_path = args.output

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    generate(phrase_file, core_name, output_path)
