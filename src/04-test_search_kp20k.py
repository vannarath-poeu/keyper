import argparse
import pathlib
import numpy as np
import json
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"
LIMIT = 2000

def search(dataset_name: str, split: str, core_name: str):
    scores = []
    data_path = f"data/{dataset_name}/{split}.json"
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            print(i)
            if i == LIMIT:
                break
            data = json.loads(line)
            if "document" in data:
                #kp20k
                extractive_keyphrases = data["extractive_keyphrases"]
                abstractive_keyphrases = data["abstractive_keyphrases"]
                score_i = 0
                for phrase in extractive_keyphrases + abstractive_keyphrases:
                    url = f"{LUCENE_URL}{core_name}/query?q=document:{phrase}"
                    resp = requests.get(url)
                    match_docs = resp.json().get("response", {}).get("docs", [])
                    doc_id_list = set([int(doc["id"]) for doc in match_docs])
                    if i in doc_id_list:
                        score_i += 1
                scores.append(score_i / len(extractive_keyphrases + abstractive_keyphrases))
            else:
                pass
    total_score = sum(scores)
    print("Average: ", total_score / len(scores))

if __name__ == "__main__":
    # Example: python3 04-test_search_kp20k.py --dataset midas/kp20k --core_name kp20k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--core_name", type=str, required=True)

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    split = args.split
    core_name = args.core_name
  
    search(dataset_name, split, core_name)