import argparse
import pathlib
import os
import json
import pke
from keybert import KeyBERT
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import spacy
from collections import Counter
import math
import copy

from utils.evaluation import evaluate
from utils.nlp import stem_keywords

nlp = spacy.load('en_core_web_sm') 

def extract_noun_list(doc):
  nlp_doc = nlp(doc)
  noun_list = []
  for np in nlp_doc.noun_chunks:
    token = np.text.lower().strip().split(" ")
    filtered = []
    for word in token:
      lexeme = nlp.vocab[word]
      if lexeme.is_stop == False and len(word) >= 2:
          filtered.append(word)
    if len(filtered) >= 4:
      continue
    token = re.sub(r'[^a-zA-Z\s]', '', " ".join(filtered))
    token = re.sub(r'\s\s+', ' ', token)
    token = token.strip()
    if token:
      noun_list.append(token)
  return noun_list

def extract_keywords(kw_model, sections, top_n=10):
  # return model.get_key_phrases(doc)
  section_kewords = []
  section_noun_list = [extract_noun_list(sec) for sec in sections]
  all_noun_list = []
  section_counter_list = [Counter(noun_list) for noun_list in section_noun_list]
  section_emb = kw_model.model.embed(sections)
  for i, noun_list in enumerate(section_noun_list):
    section_kewords.append([])
    np_list = list(set(noun_list))
    np_emb = kw_model.model.embed(np_list)
    for j, np in enumerate(np_list):
      cosine_similarity_score = cosine_similarity(
        [section_emb[i]],
        [np_emb[j]]
      )[0][0]
      tf_idf_score = section_counter_list[i][np] / len(sections[i]) * math.log(len(sections) / sum([min(section_counter[np], 1) for section_counter in section_counter_list]))
      section_kewords[i].append((np, cosine_similarity_score * tf_idf_score))

  return [sorted(sec_kw, key=lambda x: x[1], reverse=True)[:10] for sec_kw in section_kewords]

def rank_keywords(node_scores, section_keywords, top_n=10):
    top_keywords = {}
    for si, sj in sorted(node_scores, key=node_scores.get, reverse=True):
        kw = section_keywords[si][sj][0]
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
):
    # Load dataset
    dataset_path = f"{data_path}/{dataset_name}"
    test_jsonl = f"{dataset_path}/test.jsonl"

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
        }
        for k in ["abstractive", "extractive", "combined"]
    }

    num_docs = len(test)
    predictions = []

    # Init model
    kw_model = KeyBERT(model="microsoft/MiniLM-L12-H384-uncased")

    for i in range(num_docs):
        test[i] = json.loads(test[i])

        # Dataset specific
        if dataset_name == "midas/ldkp3k":
            sections = []
            for j, section in enumerate(test[i]["sections"]):
                if section.lower() != "abstract":
                    sections.append(" ".join(test[i]["sec_text"][j]))
            # doc = " ".join([s for s in sections])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        else:
            raise NotImplementedError

        section_keywords = extract_keywords(kw_model, sections)

        keyword_pair_similarity = {}
        for si in range(len(sections)):
            if si + 1 < len(sections):
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 1]
                emb_1 = kw_model.model.embed([kw for kw, _ in keyword_1])
                emb_2 = kw_model.model.embed([kw for kw, _ in keyword_2])

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        keyword_pair_similarity[(e1[0], e2[0])] = cosine_similarity(
                            [emb_1[sj]],
                            [emb_2[sk]]
                        )[0][0]
            if si + 2 < len(sections):
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 2]
                emb_1 = kw_model.model.embed([kw for kw, _ in keyword_1])
                emb_2 = kw_model.model.embed([kw for kw, _ in keyword_2])

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        keyword_pair_similarity[(e1[0], e2[0])] = cosine_similarity(
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
                        similarity = keyword_pair_similarity[(w1[0], w2[0])]
                        G.add_edge((si, sj), (si + 1, sk), capacity=similarity)
                if si + 2 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 2]):
                        similarity = keyword_pair_similarity[(w1[0], w2[0])]
                        G.add_edge((si, sj), (si + 2, sk), capacity=similarity)
            add_source = False
        
        for si in reversed(range(num_sections)):
            if len(section_keywords[si]) < 1:
                continue
            for sj, w1 in enumerate(section_keywords[si]):
                G.add_edge((si, sj), sink_node, capacity=10_000)
            break

        max_flow_value, flow_dict = nx.maximum_flow(G, source_node, sink_node)
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
        
        print(f"Processed {i+1} documents", end="\r")

        temp = copy.deepcopy(results)

        for k in temp.keys():
            for score in temp[k].keys():
                temp[k][score] /= (i+1)
        temp["num_docs"] = i+1
        json.dump(temp, open(f"{dataset_output_path}/keyper-similarity.json", "w"), indent=4)
        json.dump(predictions, open(f"{dataset_output_path}/keyper-similarity-preds.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/03-keyper-similarity.py --dataset midas/ldkp3k
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    data_path = args.data
    output_path = args.output

    keyper_score(dataset_name, data_path, output_path)
