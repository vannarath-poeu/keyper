import argparse
import pathlib
import numpy as np
import json
import requests

LUCENE_URL = "http://0.0.0.0:8983/solr/"
LIMIT = 200

def index_dataset(dataset_name: str, split: str, core_name: str):
    # Clear core
    requests.post(f"{LUCENE_URL}{core_name}/update?stream.body=<delete><query>*:*</query></delete>&commit=true")
    records = []
    data_path = f"data/{dataset_name}/{split}.json"
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            doc_id = data["id"]
            section_list = data["sections"]
            section_text_list = data["sec_text"]
            if len(section_list) != len(section_text_list):
                raise(f"section length != section_text length: ", i)
            for j, section in enumerate(section_list):
                records.append({
                    "doc_id": doc_id,
                    "title": section,
                    "document": " ".join(section_text_list[j]),
                })
            # Flush every 100 record to avoid memory overload
            if len(records) == 100:
                resp = requests.post(
                    f"{LUCENE_URL}{core_name}/update",
                    json=records
                )
                if resp.status_code != 200:
                    raise Exception(resp.text)
                records = []
        if len(records):
            resp = requests.post(
                f"{LUCENE_URL}{core_name}/update",
                json=records
            )
            if resp.status_code != 200:
                raise Exception(resp.text)

    # Reload core to allow search
    resp = requests.post(f"{LUCENE_URL}admin/cores?action=RELOAD&core={core_name}")
    if resp.status_code != 200:
        raise Exception(resp.text)

if __name__ == "__main__":
    # Example: python3 01-index_data.py --dataset midas/kp20k --core_name kp20k_train
    # Example2: python3 01-index_data.py --dataset midas/ldkp3k --core_name ldkp3k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, default="midas/ldkp100")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--core_name", type=str, default="ldkp100_train")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    split = args.split
    core_name = args.core_name
  
    index_dataset(dataset_name, split, core_name)