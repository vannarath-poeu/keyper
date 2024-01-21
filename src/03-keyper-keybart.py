import argparse
import pathlib
import os
import json
import pke
from keybert import KeyBERT
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import copy

from utils.evaluation import evaluate
from utils.nlp import stem_keywords
from utils.keybart import keybart_list, KeyphraseGenerationPipeline

def rank_keywords(node_scores, section_keywords, top_n=10):
    top_keywords = {}
    for si, sj in sorted(node_scores, key=node_scores.get, reverse=True):
        kw = section_keywords[si][sj]
        top_keywords[kw] = top_keywords.get(kw, 0) + node_scores[(si, sj)]

    stemmed_keywords = stem_keywords(top_keywords.keys())
    stemmed_keywords_map = {kw: s_kw for kw, s_kw in zip(top_keywords.keys(), stemmed_keywords)}
    
    sorted_keywords = sorted(top_keywords,
        key=lambda k: (len(k.split(" ")), -top_keywords.get(k, 0)))
    
    for i in range(len(sorted_keywords)):
        if top_keywords[sorted_keywords[i]] == 0:
            continue
        for j in range(i+1, len(sorted_keywords)):
            if stemmed_keywords_map[sorted_keywords[i]] == stemmed_keywords_map[sorted_keywords[j]]:
                top_keywords[sorted_keywords[i]] += top_keywords[sorted_keywords[j]]
                top_keywords[sorted_keywords[j]] = 0
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:10]
    return predicted_keyphrases

def keyper_score(
    dataset_name: str,
    data_path: str,
    output_path: str,
    split: str,
):
    # Load dataset
    dataset_path = f"{data_path}/{dataset_name}"
    test_jsonl = f"{dataset_path}/{split}.jsonl"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    dataset_output_path = f"{output_path}/{dataset_name}"
    pathlib.Path(dataset_output_path).mkdir(parents=True, exist_ok=True)

    with open(test_jsonl, "r") as f:
        test = f.readlines()
    results = {
        k : {
            "precision@5": 0,
            "recall@5": 0,
            "fscore@5": 0,
            "precision@10": 0,
            "recall@10": 0,
            "fscore@10": 0,
            "precision@M": 0,
            "recall@M": 0,
            "fscore@M": 0,
        }
        for k in ["abstractive", "extractive", "combined"]
    }

    num_docs = len(test)
    predictions = []

    # Init model
    kw_model = KeyBERT(model="microsoft/MiniLM-L12-H384-uncased")
    model_name = "bloomberg/KeyBART"
    generator = KeyphraseGenerationPipeline(model_name=model_name, truncation=True)

    all_keywords = {}

    all_keywords_jsonl = f"{dataset_output_path}/keyper-keybart-keywords-{split}.json"
    with open(all_keywords_jsonl, "r") as f:
        all_keywords = json.load(f)
    
    keyphrases_jsonl = f"{dataset_output_path}/keyper-keybart-preds-{split}.json"
    with open(keyphrases_jsonl, "r") as f:
        keyphrase_list = json.load(f)
    
    skipped_docs = 0

    for i in range(num_docs):
        test[i] = json.loads(test[i])

        # Dataset specific
        if dataset_name in ["midas/ldkp3k", "vannarathp/segmented-ldkp"]:
            sections = []
            for j, section in enumerate(test[i]["sections"]):
                if section.lower() != "abstract" and section.lower() != "title":
                    sections.append(" ".join(test[i]["sec_text"][j]))
            # doc = " ".join([s for s in sections])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        elif dataset_name in ["vannarathp/segmented-kptimes", "vannarathp/segmented-openkp"]:
            sections = []
            for j, section in enumerate(test[i]["document"]):
                sections.append(" ".join(section))
            # doc = " ".join([s for s in sections])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        else:
            raise NotImplementedError

        section_keywords = keybart_list(generator, sections, top_n=10)

        all_keywords[i] = section_keywords

        json.dump(all_keywords, open(f"{dataset_output_path}/keyper-keybart-keywords-{split}.json", "w"))

        if sum([len(kw) for kw in section_keywords]) == 0:
            print(f"Skipping {i} as no keywords found")
            predictions.append([])
            skipped_docs += 1
            continue

        keyword_pair_similarity = {}
        for si in range(len(sections)):
            if si + 1 < len(sections):
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 1]
                emb_1 = kw_model.model.embed(keyword_1)
                emb_2 = kw_model.model.embed(keyword_2)

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        keyword_pair_similarity[(e1, e2)] = cosine_similarity(
                            [emb_1[sj]],
                            [emb_2[sk]]
                        )[0][0]
            if si + 2 < len(sections):
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 2]
                emb_1 = kw_model.model.embed([kw for kw in keyword_1])
                emb_2 = kw_model.model.embed([kw for kw in keyword_2])

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        keyword_pair_similarity[(e1, e2)] = cosine_similarity(
                            [emb_1[sj]],
                            [emb_2[sk]]
                        )[0][0]

        num_sections = len(section_keywords)
        # Create graph and nodes
        G = nx.DiGraph()

        source_node = num_sections
        sink_node = num_sections + 1

        add_source = True

        # Add similarity score for each pair
        for si in range(num_sections):
            if len(section_keywords[si]) < 1:
                continue
            for sj, w1 in enumerate(section_keywords[si]):
                if add_source:
                    G.add_edge(source_node, (si, sj), capacity=10_000)
                if si + 1 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 1]):
                        similarity = keyword_pair_similarity[(w1, w2)]
                        G.add_edge((si, sj), (si + 1, sk), capacity=similarity)
                if si + 2 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 2]):
                        similarity = keyword_pair_similarity[(w1, w2)]
                        G.add_edge((si, sj), (si + 2, sk), capacity=similarity)
            add_source = False
        
        for si in reversed(range(num_sections)):
            if len(section_keywords[si]) < 1:
                continue
            for sj, w1 in enumerate(section_keywords[si]):
                G.add_edge((si, sj), sink_node, capacity=10_000)
            break

        _, flow_dict = nx.maximum_flow(G, source_node, sink_node)
        node_scores = {k: sum([score for _, score in flow_dict[k].items()]) for k in flow_dict if k not in [source_node, sink_node]}

        predicted_keyphrases = rank_keywords(node_scores, section_keywords, top_n=10)
        predictions.append(predicted_keyphrases)

        abstractive_keyphrases = stem_keywords(abstractive_keyphrases)
        extractive_keyphrases = stem_keywords(extractive_keyphrases)
        combined_keyphrases = abstractive_keyphrases + extractive_keyphrases

        predicted_keyphrases = stem_keywords(predicted_keyphrases)

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], abstractive_keyphrases)
            results["abstractive"][f"precision@{k}"] += p
            results["abstractive"][f"recall@{k}"] += r
            results["abstractive"][f"fscore@{k}"] += f

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], extractive_keyphrases)
            results["extractive"][f"precision@{k}"] += p
            results["extractive"][f"recall@{k}"] += r
            results["extractive"][f"fscore@{k}"] += f

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], combined_keyphrases)
            results["combined"][f"precision@{k}"] += p
            results["combined"][f"recall@{k}"] += r
            results["combined"][f"fscore@{k}"] += f
        
        for k in ["M"]:
            len_keyphrases = len(abstractive_keyphrases)
            p, r, f = evaluate(predicted_keyphrases[:len_keyphrases], abstractive_keyphrases)
            results["abstractive"][f"precision@{k}"] += p
            results["abstractive"][f"recall@{k}"] += r
            results["abstractive"][f"fscore@{k}"] += f

        for k in ["M"]:
            len_keyphrases = len(extractive_keyphrases)
            p, r, f = evaluate(predicted_keyphrases[:len_keyphrases], extractive_keyphrases)
            results["extractive"][f"precision@{k}"] += p
            results["extractive"][f"recall@{k}"] += r
            results["extractive"][f"fscore@{k}"] += f

        for k in ["M"]:
            len_keyphrases = len(combined_keyphrases)
            p, r, f = evaluate(predicted_keyphrases[:len_keyphrases], combined_keyphrases)
            results["combined"][f"precision@{k}"] += p
            results["combined"][f"recall@{k}"] += r
            results["combined"][f"fscore@{k}"] += f
        
        print(f"Processed {i+1} documents", end="\r")

        temp = copy.deepcopy(results)

        for k in temp.keys():
            for score in temp[k].keys():
                temp[k][score] /= (i+1 - skipped_docs)
        temp["num_docs"] = i+1 - skipped_docs
        json.dump(temp, open(f"{dataset_output_path}/keyper-keybart-{split}.json", "w"), indent=4)
        json.dump(predictions, open(f"{dataset_output_path}/keyper-keybart-preds-{split}.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/03-keyper-keybart.py --dataset midas/ldkp3k
    # Or: python3 src/03-keyper-keybart.py --dataset vannarathp/segmented-ldkp
    # Or: python3 src/03-keyper-keybart.py --dataset vannarathp/segmented-kptimes
    # Or: python3 src/03-keyper-keybart.py --dataset vannarathp/segmented-openkp
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    data_path = args.data
    output_path = args.output
    split = args.split

    keyper_score(dataset_name, data_path, output_path, split)
