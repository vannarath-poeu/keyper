import argparse
import pathlib
import os
import json
import pyomo.environ as pyo
from collections import defaultdict
import networkx as nx
import copy
from typing import List, Any
import multiprocessing
from multiprocessing import Pool
from functools import partial
from datetime import datetime

import chromadb
from chromadb.db.base import UniqueConstraintError
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity

from utils.evaluation import evaluate
from utils.nlp import stem_keywords

def rank_keywords_dynamic(top_keywords, top_n=10):
    stemmed_keywords = stem_keywords(top_keywords.keys())
    stemmed_keywords_map = {kw: s_kw for kw, s_kw in zip(top_keywords.keys(), stemmed_keywords)}
    
    sorted_keywords = sorted(top_keywords,
        key=lambda k: (len(k.split(" ")), -top_keywords.get(k, 0)))
    
    for i in range(len(sorted_keywords)):
        if top_keywords[sorted_keywords[i]] == 0:
            continue
        for j in range(i+1, len(sorted_keywords)):
            if stemmed_keywords_map[sorted_keywords[j]].startswith(stemmed_keywords_map[sorted_keywords[i]] + " "):
                top_keywords[sorted_keywords[i]] += top_keywords[sorted_keywords[j]]
                top_keywords[sorted_keywords[j]] = 0
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:top_n]
    return predicted_keyphrases

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
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:top_n]
    return predicted_keyphrases

def load_keywords(
    dataset_name: str,
    output_path: str,
    split: str,
) -> Any:
    keybard_keyword_json = f"{output_path}/{dataset_name}/keyper-keybart-keywords-{split}.json"
    assert os.path.exists(keybard_keyword_json), f"File {keybard_keyword_json} does not exist"
    with open(keybard_keyword_json, "r") as f:
        keywords = json.load(f)
    return keywords

def load_dataset(
    dataset_name: str,
    data_path: str,
    split: str,
) -> List[str]:
    data_jsonl = f"{data_path}/{dataset_name}/{split}.jsonl"
    assert os.path.exists(data_jsonl), f"File {data_jsonl} does not exist"
    with open(data_jsonl, "r") as f:
        data = f.readlines()
    return data

def init_score():
    return {
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

def update_score(
    results,
    abstractive_keyphrases: List[str],
    extractive_keyphrases: List[str],
    predicted_keyphrases: List[str]
):
    temp = copy.deepcopy(results)
    abstractive_keyphrases = stem_keywords(abstractive_keyphrases)
    extractive_keyphrases = stem_keywords(extractive_keyphrases)
    combined_keyphrases = abstractive_keyphrases + extractive_keyphrases

    predicted_keyphrases = stem_keywords(predicted_keyphrases)

    keyphrase_type_dict = {
        "abstractive": abstractive_keyphrases,
        "extractive": extractive_keyphrases,
        "combined": combined_keyphrases,
    }

    for k in [5, 10, "M"]:
        for keyphrase_type, gold_keyphrases in keyphrase_type_dict.items():
            prediction_k = len(gold_keyphrases) if k == "M" else k
            p, r, f = evaluate(predicted_keyphrases[:prediction_k], gold_keyphrases)
            temp[keyphrase_type][f"precision@{k}"] += p
            temp[keyphrase_type][f"recall@{k}"] += r
            temp[keyphrase_type][f"fscore@{k}"] += f
    return temp

# def pyomo_max_flow():
#     model = pyo.ConcreteModel("max_flow")

#     # nodes
#     nodes = set(["source", "sink", "early_exit"])
#     # edges
#     edges = defaultdict(float)

#     add_source = True
#     for si, _ in enumerate(section_keywords):
#         if len(section_keywords[si]) < 1:
#             continue
#         for sj, kw1 in enumerate(section_keywords[si]):
#             node_1 = f"{si}_{sj}_{kw1}"
#             nodes.add(node_1)
#             result_kw1 = collection.get(f"kw_{idx}_{si}_{sj}", include=["embeddings", "documents"])
#             kw1_embedding = result_kw1["embeddings"][0]
#             if add_source:
#                 edges[("source", node_1)] = 10_000
#             if si + 1 < num_sections:
#                 for sk, kw2 in enumerate(section_keywords[si + 1]):
#                     node_2 = f"{si + 1}_{sk}_{kw2}"
#                     result_kw2 = collection.get(f"kw_{idx}_{si + 1}_{sk}", include=["embeddings", "documents"])
#                     kw2_embedding = result_kw2["embeddings"][0]
#                     similarity = cosine_similarity(
#                         [kw1_embedding],
#                         [kw2_embedding],
#                     )[0][0]
#                     edges[(node_1, node_2)] = similarity
#             if si + 2 < num_sections:
#                 for sk, kw2 in enumerate(section_keywords[si + 2]):
#                     node_2 = f"{si + 2}_{sk}_{kw2}"
#                     result_kw2 = collection.get(f"kw_{idx}_{si + 2}_{sk}", include=["embeddings", "documents"])
#                     kw2_embedding = result_kw2["embeddings"][0]
#                     similarity = cosine_similarity(
#                         [kw1_embedding],
#                         [kw2_embedding]
#                     )[0][0]
#                     edges[(node_1, node_2)] = similarity
#         add_source = False

#     sink_node = "sink"
#     for si in reversed(range(num_sections)):
#         if len(section_keywords[si]) < 1:
#             continue
#         for sj, w1 in enumerate(section_keywords[si]):
#             kw1 = w1[0]
#             node_1 = f"{si}_{sj}_{kw1}"
#             edges[node_1, sink_node] = 10_000
#         else:
#             sink_node = "early_exit"
    
#     nodes = list(nodes)
#     node_products = [(s, t) for s in nodes for t in nodes]


#     model.f = pyo.Var(node_products, domain=pyo.NonNegativeReals)

#     # Maximize the flow into the sink nodes
#     def total_rule(model):
#         return sum(model.f[n] for n in node_products)

#     # Enforce an upper limit on the flow across each edge
#     def limit_rule(model, s, e):
#         return model.f[(s, e)] <= edges.get((s, e), 0)

#     # Enforce flow through each node
#     def flow_rule(model, node):
#         if node == "source" or node == "sink" or node == "early_exit":
#             return pyo.Constraint.Skip
#         inFlow  = sum(model.f[(source, node)] for source in nodes)
#         outFlow = sum(model.f[(node, dest)] for dest in nodes)
#         return inFlow == outFlow

#     model.total = pyo.Objective(rule=total_rule, sense=pyo.maximize)
#     model.limit = pyo.Constraint(nodes, nodes, rule=limit_rule)
#     model.flow = pyo.Constraint(nodes, rule=flow_rule)

#     # solver = pyo.SolverFactory('glpk')  # "glpk"
#     solver = pyo.SolverFactory('cbc')  # "cbc"

#     res = solver.solve(model)

#     pyo.assert_optimal_termination(res)

#     keyword_scores = defaultdict(float)
#     for val in model.f:
#         score = model.f[val].value
#         if score is None or score <= 0:
#             continue
#         node1, node2 = val
#         if node1 == "source" or node2 == "early_exit":
#             continue
#         _, _, kw1 = node1.split("_")
#         # _, _, kw2 = node2.split("_")
#         keyword_scores[kw1] += score
    
#     print(keyword_scores)
    
#     # predicted_keyphrases = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:10]
#     predicted_keyphrases = rank_keywords(keyword_scores, top_n=10)

def process_record(
    idx: int,
    dataset_name: str,
    dataset: List[str],
    all_keyword_dict: Any,
    collection: Any,
    experiment: str,
    results: Any,
):
    # Load record into JSON format
        record = json.loads(dataset[idx])
        # Load keywords for the records
        keywords = all_keyword_dict[str(idx)]

        if len(keywords) == 0:
            return {
                "skipped_docs": 1,
                "results": results,
                "predicted_keyphrases": [],
            }

        abstractive_keyphrases = record["abstractive_keyphrases"]
        extractive_keyphrases = record["extractive_keyphrases"]

        # Dataset specific
        if dataset_name in ["vannarathp/segmented-kptimes", "vannarathp/segmented-openkp"]:
            document_field = "document"
        elif dataset_name in ["vannarathp/segmented-ldkp"]:
            document_field = "sec_text"
        else:
            raise NotImplementedError

        ids = [f"sec_{idx}_{j}" for j in range(len(record[document_field]))]
        docs = [" ".join(section) for section in record[document_field]]
        collection.add(
            ids=ids,
            documents=docs,
        )

        ids = [f"kw_{idx}_{j}_{k}" for j in range(len(keywords)) for k in range(len(keywords[j]))]
        docs = [keywords[j][k] for j in range(len(keywords)) for k in range(len(keywords[j]))]
        collection.add(
            ids=ids,
            documents=docs,
        )

        section_keywords = [sk for sk in keywords]
        num_sections = len(section_keywords)

        keyword_pair_similarity = {}
        for si in range(num_sections):
            section_similarity = 1.0
            if si + 1 < num_sections:
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 1]
                # emb_1 = kw_model.model.embed(keyword_1)
                # emb_2 = kw_model.model.embed(keyword_2)

                if experiment.startswith("section"):
                    result_sec1 = collection.get(f"sec_{idx}_{si}", include=["embeddings", "documents"])
                    emb_s1 = result_sec1["embeddings"][0]
                    result_sec2 = collection.get(f"sec_{idx}_{si + 1}", include=["embeddings", "documents"])
                    emb_s2 = result_sec2["embeddings"][0]
                    section_similarity = cosine_similarity(
                        [emb_s1],
                        [emb_s2]
                    )[0][0]

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        result_kw1 = collection.get(f"kw_{idx}_{si}_{sj}", include=["embeddings", "documents"])
                        emb_1 = result_kw1["embeddings"][0]
                        result_kw2 = collection.get(f"kw_{idx}_{si + 1}_{sk}", include=["embeddings", "documents"])
                        emb_2 = result_kw2["embeddings"][0]
                        keyword_pair_similarity[(e1, e2)] = cosine_similarity(
                            [emb_1],
                            [emb_2]
                        )[0][0] * section_similarity
            if si + 2 < num_sections:
                keyword_1 = section_keywords[si]
                keyword_2 = section_keywords[si + 2]
                # emb_1 = kw_model.model.embed([kw for kw in keyword_1])
                # emb_2 = kw_model.model.embed([kw for kw in keyword_2])

                if experiment.startswith("section"):
                    result_sec1 = collection.get(f"sec_{idx}_{si}", include=["embeddings", "documents"])
                    emb_s1 = result_sec1["embeddings"][0]
                    result_sec2 = collection.get(f"sec_{idx}_{si + 2}", include=["embeddings", "documents"])
                    emb_s2 = result_sec2["embeddings"][0]
                    section_similarity = cosine_similarity(
                        [emb_s1],
                        [emb_s2]
                    )[0][0]

                for sj, e1 in enumerate(keyword_1):
                    for sk, e2 in enumerate(keyword_2):
                        result_kw1 = collection.get(f"kw_{idx}_{si}_{sj}", include=["embeddings", "documents"])
                        emb_1 = result_kw1["embeddings"][0]
                        result_kw2 = collection.get(f"kw_{idx}_{si + 2}_{sk}", include=["embeddings", "documents"])
                        emb_2 = result_kw2["embeddings"][0]
                        keyword_pair_similarity[(e1, e2)] = cosine_similarity(
                            [emb_1],
                            [emb_2]
                        )[0][0] * section_similarity
        
        # Create graph and nodes
        G = nx.DiGraph()

        source_node = num_sections
        sink_node = num_sections + 1

        if "dynamic" in experiment:
            add_source = True
            # Add similarity score for each pair
            for si in range(num_sections):
                if len(section_keywords[si]) < 1:
                    continue
                for sj, w1 in enumerate(section_keywords[si]):
                    if add_source:
                        G.add_edge(source_node, section_keywords[si][sj], capacity=10_000)
                    if si + 1 < num_sections:
                        for sk, w2 in enumerate(section_keywords[si + 1]):
                            similarity = keyword_pair_similarity[(w1, w2)]
                            G.add_edge(section_keywords[si][sj], section_keywords[si + 1][sk], capacity=similarity)
                    if si + 2 < num_sections:
                        for sk, w2 in enumerate(section_keywords[si + 2]):
                            similarity = keyword_pair_similarity[(w1, w2)]
                            G.add_edge(section_keywords[si][sj], section_keywords[si + 2][sk], capacity=similarity)
                add_source = False
            
            for si in reversed(range(num_sections)):
                if len(section_keywords[si]) < 1:
                    continue
                for sj, w1 in enumerate(section_keywords[si]):
                    G.add_edge(section_keywords[si][sj], sink_node, capacity=10_000)
                break
        else:
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
        if "dynamic" in experiment:
            node_scores = {k: sum([score for _, score in flow_dict[k].items()]) for k in flow_dict if k not in [source_node, sink_node]}
            predicted_keyphrases = rank_keywords_dynamic(node_scores, top_n=10)
            pass
        else:
            node_scores = {k: sum([score for _, score in flow_dict[k].items()]) for k in flow_dict if k not in [source_node, sink_node]}
            predicted_keyphrases = rank_keywords(node_scores, section_keywords, top_n=10)

        results = update_score(
            results,
            abstractive_keyphrases,
            extractive_keyphrases,
            predicted_keyphrases,
        )

        return {
            "skipped_docs": 0,
            "results": results,
            "predicted_keyphrases": predicted_keyphrases,
        }

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def process_chunk(
    chunk_list: List[int],
    dataset_name: str = "",
    dataset: List[str] = [],
    experiment: str = "",
    split: str = "",
    all_keyword_dict: Any = {},
    output_path: str = "",
):
    results = init_score()
    processed_docs = 0
    skipped_docs = 0

    # ChromaDB for each dataset
    client = chromadb.PersistentClient(path="keyflow/")
    em = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="microsoft/MiniLM-L12-H384-uncased")

    collection_name = dataset_name.split("/")[-1]
    try:
        collection = client.create_collection(name=collection_name, embedding_function=em)
    except:
        collection = client.get_collection(name=collection_name, embedding_function=em)

    predictions = []
    score_file_name = f"{output_path}/{dataset_name}/keyflow-{experiment}-{split}-scores-{chunk_list[0]}-{chunk_list[-1]}.json"
    pred_file_name = f"{output_path}/{dataset_name}/keyflow-{experiment}-{split}-preds-{chunk_list[0]}-{chunk_list[-1]}.json"

    for idx in chunk_list:
        response = process_record(
            idx,
            dataset_name,
            dataset,
            all_keyword_dict,
            collection,
            experiment,
            results,
        )
        skipped_docs += response["skipped_docs"]
        processed_docs += (1 - response["skipped_docs"])
        results = response["results"]
        predictions.append(response["predicted_keyphrases"])

        temp = copy.deepcopy(results)

        temp["num_docs"] = (processed_docs - skipped_docs)
        json.dump(temp, open(score_file_name, "w"), indent=4)
        json.dump(predictions, open(pred_file_name, "w"), indent=4)
    return score_file_name

def max_flow(
    dataset_name: str,
    data_path: str,
    output_path: str,
    split: str,
    experiment: str,
):
    dataset = load_dataset(
        dataset_name,
        data_path,
        split,
    )
    all_keyword_dict = load_keywords(
        dataset_name,
        output_path,
        split,
    )

    num_records = len(dataset)
    print(f"Number of records: {num_records}")

    multiprocessing.set_start_method('spawn')

    num_cpu = 4
    pool = Pool(num_cpu)
    chunk_list = list(chunks(range(0, num_records), num_records // num_cpu + 1))
    score_file_list = pool.map(
        partial(
            process_chunk,
            dataset_name=dataset_name,
            dataset=dataset,
            all_keyword_dict=all_keyword_dict,
            experiment=experiment,
            split=split,
            output_path=output_path,
        ),
        chunk_list,
    )

    pool.close()
    pool.join()

    combined_results = init_score()
    combined_results["num_docs"] = 0

    # started at 12:42:00 AM
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished at {dt_string}")

    for score_file in score_file_list:
        score = json.load(open(score_file, "r"))
        combined_results["num_docs"] += score["num_docs"]

        for k in [5, 10, "M"]:
            for keyphrase_type in ["abstractive", "extractive", "combined"]:
                combined_results[keyphrase_type][f"precision@{k}"] += score[keyphrase_type][f"precision@{k}"]
                combined_results[keyphrase_type][f"recall@{k}"] += score[keyphrase_type][f"recall@{k}"]
                combined_results[keyphrase_type][f"fscore@{k}"] += score[keyphrase_type][f"fscore@{k}"]
    
    for k in [5, 10, "M"]:
        for keyphrase_type in ["abstractive", "extractive", "combined"]:
            combined_results[keyphrase_type][f"precision@{k}"] /= combined_results["num_docs"]
            combined_results[keyphrase_type][f"recall@{k}"] /= combined_results["num_docs"]
            combined_results[keyphrase_type][f"fscore@{k}"] /= combined_results["num_docs"]
    
    score_file_name = f"{output_path}/{dataset_name}/keyflow-{experiment}-{split}-scores-n={num_records}.json"
    json.dump(combined_results, open(score_file_name, "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/15-keyflow.py --dataset vannarathp/segmented-ldkp
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-kptimes
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-openkp

    # python3 src/15-keyflow.py --dataset vannarathp/segmented-ldkp --experiment section
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-kptimes --experiment section
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-openkp --experiment section

    # python3 src/15-keyflow.py --dataset vannarathp/segmented-ldkp --experiment dynamic
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-kptimes --experiment dynamic
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-openkp --experiment dynamic

    # python3 src/15-keyflow.py --dataset vannarathp/segmented-ldkp --experiment section-dynamic
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-kptimes --experiment section-dynamic
    # Or: python3 src/15-keyflow.py --dataset vannarathp/segmented-openkp --experiment section-dynamic


    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--experiment", type=str, default="original")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    data_path = args.data
    output_path = args.output
    split = args.split
    experiment = args.experiment

    max_flow(
        dataset_name,
        data_path,
        output_path,
        split,
        experiment,
    )
