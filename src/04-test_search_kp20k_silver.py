import argparse
import pathlib
import numpy as np
import json
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"
LIMIT = 2000

def search(core_name: str):
    scores = []
    data_path = f"temp/silver.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
        for id, phrase_list in data.items():
            score_i = 0
            for phrase in phrase_list[:10]:
                url = f"{LUCENE_URL}{core_name}/query?q=document:{phrase}"
                resp = requests.get(url)
                match_docs = resp.json().get("response", {}).get("docs", [])
                doc_id_list = set([doc["id"] for doc in match_docs])
                if id in doc_id_list:
                    score_i += 1
            scores.append(score_i / 10)
    total_score = sum(scores)
    print("Average: ", total_score / len(scores))

if __name__ == "__main__":
    # Example: python3 04-test_search_kp20k_silver.py --core_name kp20k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--core_name", type=str, required=True)

    args = parser.parse_args()
    # Get all the variables
    core_name = args.core_name
  
    search(core_name)