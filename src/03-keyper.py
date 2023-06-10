import argparse
import pathlib
import os
import json
import pke
import spacy
from nltk.stem.porter import PorterStemmer
from keybert import KeyBERT
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def list_jsonl(path):
    files = []
    for r, _, f in os.walk(path):
        for file in f:
            if '.jsonl' in file:
                files.append(os.path.join(r, file))
    return files

def extract_keywords(doc, top_n=10):
  extractor = pke.unsupervised.PositionRank()
  extractor.load_document(input=doc, language='en')
  extractor.candidate_selection()
  extractor.candidate_weighting()
  keyphrases = extractor.get_n_best(n=top_n)
  return keyphrases

# def keyper(doc, top_n=10):
#     extractor = pke.unsupervised.PositionRank()
#     extractor.load_document(input=doc, language='en')
#     extractor.candidate_selection()
#     extractor.candidate_weighting()
#     keyphrase_score = extractor.get_n_best(n=top_n)
#     return [k for k, _ in keyphrase_score]

def rank_keywords(node_scores, section_keywords, top_n=10):
    top_keywords = {}
    for si, sj in sorted(node_scores, key=node_scores.get, reverse=True):
        kw = section_keywords[si][sj][0]
        top_keywords[kw] = top_keywords.get(kw, 0) + node_scores[(si, sj)]
    
    # sorted_keywords = sorted(top_keywords,
    #     key=lambda k: (len(k.split(" ")), -top_keywords.get(k, 0)))
    
    # for i in range(len(sorted_keywords)):
    #     if top_keywords[sorted_keywords[i]] == 0:
    #         continue
    #     for j in range(i+1, len(sorted_keywords)):
    #         l = sorted_keywords[i].split(" ")
    #         p = sorted_keywords[j].split(" ")
    #         if set(l).issubset(set(p)):
    #             top_keywords[sorted_keywords[i]] += top_keywords[sorted_keywords[j]]
    #             top_keywords[sorted_keywords[j]] = 0
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:10]
    return predicted_keyphrases

def fscore(precision: float, recall: float):
    if precision == 0 or recall == 0:
        return 0
    
    assert precision >= 0 and precision <= 1, f"Precision is not between 0 and 1: {precision}"
    assert recall >= 0 and recall <= 1, f"Recall is not between 0 and 1: {recall}"
    
    f1_score = 2 * (precision * recall) / (precision + recall)

    assert 0 <= f1_score <= 1, f"F1 score is not between 0 and 1: {f1_score}, precision: {precision}, recall: {recall}"

    return f1_score

def precision_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0
    
    precision = 0
    remaining_labels = labels[:]
    for p in predictions:
        for l in remaining_labels:
            if set(l).issubset(set(p)) or set(p).issubset(set(l)):
                remaining_labels.remove(l)
                precision += 1
                # 1 label matches 1 prediction
                break
        
    precision /= len(predictions)

    assert 0 <= precision <= 1, f"Precision is not between 0 and 1: {precision}, predictions: {predictions}, labels: {labels}"
    return precision

def recall_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0

    recall = 0
    remaining_labels = labels[:]
    for p in predictions:
        for l in remaining_labels:
            if set(l).issubset(set(p)) or set(p).issubset(set(l)):
                remaining_labels.remove(l)
                recall += 1
                # 1 label matches 1 prediction
                break

    recall /= len(labels)
    assert 0 <= recall <= 1, f"Recall is not between 0 and 1: {recall}, predictions: {predictions}, labels: {labels}"
    return recall

def stem_keywords(keyword_list: list):
    porter_stemmer = PorterStemmer()
    stemmed_keywords = []
    for kw in keyword_list:
        stemmed = [porter_stemmer.stem(q) for q in kw.split()]
        stemmed_keywords.append(stemmed)
    return stemmed_keywords

def keyper_score(
    dataset_name: str,
    data_path: str,
    output_path: str,
):
    # Load dataset
    dataset_path = f"{data_path}/{dataset_name}"
    # json_l_list = list_jsonl(dataset_path)
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

        num_docs = 100 # len(test)

        for i in range(num_docs):
            test[i] = json.loads(test[i])

            # Dataset specific
            if dataset_name == "midas/ldkp3k":
                sections = []
                for j, section in enumerate(test[i]["sections"]):
                    if section.lower() != "abstract" and section.lower() != "title":
                        sections.append(" ".join(test[i]["sec_text"][j]))
                # doc = " ".join([s for s in sections])
                abstractive_keyphrases = test[i]["abstractive_keyphrases"]
                extractive_keyphrases = test[i]["extractive_keyphrases"]
            else:
                raise NotImplementedError

            combined_keyphrases = abstractive_keyphrases + extractive_keyphrases
            # predicted_keyphrases = keyper(doc, top_n=10)

            # Init model
            kw_model = KeyBERT(model="microsoft/MiniLM-L12-H384-uncased")

            section_keywords = [extract_keywords(doc) for doc in sections]
            # section_embeddings = kw_model.model.embed([" ".join([word for word, _ in sec_kw ]) for sec_kw in section_keywords])

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
            # max_keyphrases_per_section = 5
            # Create graph and nodes
            G = nx.DiGraph()

            source_node = num_sections
            sink_node = num_sections + 1

            # Add similarity score for each pair
            for si in range(num_sections):
                for sj, w1 in enumerate(section_keywords[si]):
                    if si == 0:
                        G.add_edge(source_node, (si, sj), capacity=10_000)
                    if si + 1 < num_sections:
                        for sk, w2 in enumerate(section_keywords[si + 1]):
                            similarity = keyword_pair_similarity[(w1[0], w2[0])]
                            G.add_edge((si, sj), (si + 1, sk), capacity=similarity)
                    if si + 2 < num_sections:
                        for sk, w2 in enumerate(section_keywords[si + 2]):
                            similarity = keyword_pair_similarity[(w1[0], w2[0])]
                            G.add_edge((si, sj), (si + 2, sk), capacity=similarity)
                    if si == num_sections - 1:
                        G.add_edge((si, sj), sink_node, capacity=10_000)
            max_flow_value, flow_dict = nx.maximum_flow(G, source_node, sink_node)
            node_scores = {k: sum([score for _, score in flow_dict[k].items()]) for k in flow_dict if k not in [source_node, sink_node]}
            # nlp = spacy.load('en_core_web_sm') # load spaCy model

            predicted_keyphrases = rank_keywords(node_scores, section_keywords, top_n=10)

            print("Extractive keyphrases", extractive_keyphrases)
            print("Abstractive keyphrases", abstractive_keyphrases)
            print("Predicted keyphrases", predicted_keyphrases)

            # for kw in sorted(top_keywords, key=top_keywords.get, reverse=True)[:]:
            #     print(kw)

            precision_at_5 = precision_at_k(predicted_keyphrases, abstractive_keyphrases, k=5)
            recall_at_5 = recall_at_k(predicted_keyphrases, abstractive_keyphrases, k=5)
            results["abstractive"]["precision@5"] += precision_at_5
            results["abstractive"]["recall@5"] += recall_at_5
            results["abstractive"]["fscore@5"] += fscore(precision_at_5, recall_at_5)
            precision_at_10 = precision_at_k(predicted_keyphrases, abstractive_keyphrases, k=10)
            recall_at_10 = recall_at_k(predicted_keyphrases, abstractive_keyphrases, k=10)
            results["abstractive"]["precision@10"] += precision_at_10
            results["abstractive"]["recall@10"] += recall_at_10
            results["abstractive"]["fscore@10"] += fscore(precision_at_10, recall_at_10)

            precision_at_5 = precision_at_k(predicted_keyphrases, extractive_keyphrases, k=5)
            recall_at_5 = recall_at_k(predicted_keyphrases, extractive_keyphrases, k=5)
            results["extractive"]["precision@5"] += precision_at_5
            results["extractive"]["recall@5"] += recall_at_5
            results["extractive"]["fscore@5"] += fscore(precision_at_5, recall_at_5)
            precision_at_10 = precision_at_k(predicted_keyphrases, extractive_keyphrases, k=10)
            recall_at_10 = recall_at_k(predicted_keyphrases, extractive_keyphrases, k=10)
            results["extractive"]["precision@10"] += precision_at_10
            results["extractive"]["recall@10"] += recall_at_10
            results["extractive"]["fscore@10"] += fscore(precision_at_10, recall_at_10)

            precision_at_5 = precision_at_k(predicted_keyphrases, combined_keyphrases, k=5)
            recall_at_5 = recall_at_k(predicted_keyphrases, combined_keyphrases, k=5)
            results["combined"]["precision@5"] += precision_at_5
            results["combined"]["recall@5"] += recall_at_5
            results["combined"]["fscore@5"] += fscore(precision_at_5, recall_at_5)
            precision_at_10 = precision_at_k(predicted_keyphrases, combined_keyphrases, k=10)
            recall_at_10 = recall_at_k(predicted_keyphrases, combined_keyphrases, k=10)
            results["combined"]["precision@10"] += precision_at_10
            results["combined"]["recall@10"] += recall_at_10
            results["combined"]["fscore@10"] += fscore(precision_at_10, recall_at_10)
            
            print(f"Processed {i+1} documents", end="\r")

        for k in results.keys():
            for score in results[k].keys():
                results[k][score] /= num_docs
        json.dump(results, open(f"{dataset_output_path}/keyper.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/03-keyper.py --dataset midas/ldkp3k
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
