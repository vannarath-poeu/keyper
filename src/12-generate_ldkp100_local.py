import argparse
import pathlib
import numpy as np
import json
import requests
import nltk

LUCENE_URL = "http://0.0.0.0:8983/solr/"
LIMIT = 200

def extract_candidates(text: str, threshold=4):
    GRAMMAR_EN = """  NP:
{<NN.*|JJ>*<NN.*>}"""   # Adjective(s)(optional) + Noun(s)
    keyphrase_candidate = set()
    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    tag = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    trees = np_parser.parse_sents(tag)  # Generator with one tree per sentence

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, _ in subtree.leaves()))
    
    keyphrase_candidate = [kp for kp in keyphrase_candidate if len(kp.split()) <= threshold]
    return keyphrase_candidate

def generate_local(dataset_name: str, split: str):
    records = []
    data_path = f"data/{dataset_name}/{split}.json"
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            section_list = data["sections"]
            section_text_list = data["sec_text"]
            if len(section_list) != len(section_text_list):
                raise(f"section length != section_text length: ", i)
            for j, section in enumerate(section_list):
                if section.lower() in ["introduction", "conclusion", "title", "abstract", "acknowledgement"]:
                    continue
                records.append({
                    "keywords": extract_candidates(section),
                    "document": " ".join(section_text_list[j]),
                })
    
    with open(f"data/midas/ldkp100_local/{split}.json", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    # Example: python3 01-index_data.py --dataset midas/kp20k --core_name kp20k_train
    # Example2: python3 01-index_data.py --dataset midas/ldkp3k --core_name ldkp3k_train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, default="midas/ldkp100")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    split = args.split
  
    generate_local(dataset_name, split)