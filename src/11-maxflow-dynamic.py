import argparse
import pathlib
import os
import json
import pyomo.environ as pyo
from collections import defaultdict
import copy

from utils.evaluation import evaluate
from utils.nlp import stem_keywords

def rank_keywords(top_keywords, top_n=10):
    stemmed_keywords = stem_keywords(top_keywords.keys())
    stemmed_keywords_map = {kw: s_kw for kw, s_kw in zip(top_keywords.keys(), stemmed_keywords)}
    
    sorted_keywords = sorted(top_keywords,
        key=lambda k: (len(k.split(" ")), -top_keywords.get(k, 0)))
    
    for i in range(len(sorted_keywords)):
        if top_keywords[sorted_keywords[i]] == 0 or len(sorted_keywords[i]) < 3:
            continue
        for j in range(i+1, len(sorted_keywords)):
            if stemmed_keywords_map[sorted_keywords[j]].startswith(stemmed_keywords_map[sorted_keywords[i]] + " "):
                top_keywords[sorted_keywords[i]] += top_keywords[sorted_keywords[j]]
                top_keywords[sorted_keywords[j]] = 0
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:top_n]

    return predicted_keyphrases

def max_flow(
    data_path: str,
    output_path: str,
    max_keyphrases=10,
    num_records=None,
):
    if num_records is None:
        prediction_file_path = f"{data_path}/keyper-similarity-temp-preds.json"
        similarity_file_path = f"{data_path}/keyper-similarity-temp-sims.json"
    else:
        prediction_file_path = f"{data_path}/keyper-similarity-temp-preds-{num_records}.json"
        similarity_file_path = f"{data_path}/keyper-similarity-temp-sims-{num_records}.json"

    assert os.path.exists(prediction_file_path), f"File {prediction_file_path} does not exist"
    assert os.path.exists(similarity_file_path), f"File {similarity_file_path} does not exist"

    test_jsonl = f"data/midas/ldkp3k/test.jsonl"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    # Load prediction and similarity files
    with open(prediction_file_path) as f:
        predictions = json.load(f)

    with open(similarity_file_path) as f:
        similarities = json.load(f)

    new_predictions = []
    if num_records is None:
        num_records = len(predictions)

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

    print(f"Number of records: {num_records}")
    for i in range(num_records):
        model = pyo.ConcreteModel("max_flow")
        keyword_pair_similarity = defaultdict(float)
        for similarity in similarities[i]:
            kw0, kw1, sim = similarity
            keyword_pair_similarity[(kw0, kw1)] = sim

        section_keywords = [sk[:max_keyphrases] for sk in predictions[i]]
        num_sections = len(section_keywords)

        test[i] = json.loads(test[i])
        abstractive_keyphrases = test[i]["abstractive_keyphrases"]
        extractive_keyphrases = test[i]["extractive_keyphrases"]

        # nodes
        nodes = set(["source", "sink", "early_exit"])
        # edges
        edges = defaultdict(float)

        add_source = True
        for si, _ in enumerate(section_keywords):
            if len(section_keywords[si]) < 1:
                continue
            for sj, w1 in enumerate(section_keywords[si]):
                kw1 = w1[0]
                if len(kw1) < 3:
                    continue
                node_1 = f"{kw1}"
                # if node_1 in nodes:
                #     edges[(node_1, node_1)] = 1
                nodes.add(node_1)
                if add_source:
                    edges[("source", node_1)] = 10_000
                if si + 1 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 1]):
                        kw2 = w2[0]
                        if len(kw2) < 3:
                            continue
                        node_2 = f"{kw2}"
                        similarity = keyword_pair_similarity[(kw1, kw2)]
                        edges[(node_1, node_2)] = similarity
                if si + 2 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 2]):
                        kw2 = w2[0]
                        if len(kw2) < 3:
                            continue
                        node_2 = f"{kw2}"
                        similarity = keyword_pair_similarity[(kw1, kw2)]
                        edges[(node_1, node_2)] = similarity
            # add_source = False
        
        sink_node = "sink"
        for si in reversed(range(num_sections)):
            if len(section_keywords[si]) < 1:
                continue
            for _, w1 in enumerate(section_keywords[si]):
                kw1 = w1[0]
                if len(kw1) < 3:
                    continue
                node_1 = f"{kw1}"
                edges[node_1, sink_node] = 10_000
            else:
                sink_node = "early_exit"
        
        nodes = list(nodes)
        node_products = [(s, t) for s in nodes for t in nodes]
        model.f = pyo.Var(node_products, domain=pyo.NonNegativeReals)

        # Maximize the flow into the sink nodes
        def total_rule(model):
            return sum(model.f[n] for n in node_products)

        model.total = pyo.Objective(rule=total_rule, sense=pyo.maximize)

        # Enforce an upper limit on the flow across each edge
        def limit_rule(model, s, e):
            return model.f[(s, e)] <= edges.get((s, e), 0)

        model.limit = pyo.Constraint(nodes, nodes, rule=limit_rule)

        # Enforce flow through each node
        def flow_rule(model, node):
            if node == "source" or node == "sink" or node == "early_exit":
                return pyo.Constraint.Skip
            inFlow  = sum(model.f[(source, node)] for source in nodes)
            outFlow = sum(model.f[(node, dest)] for dest in nodes)
            return inFlow == outFlow

        model.flow = pyo.Constraint(nodes, rule=flow_rule)

        # solver = pyo.SolverFactory('glpk')  # "glpk"
        solver = pyo.SolverFactory('gurobi')  # "cbc"

        res = solver.solve(model)

        pyo.assert_optimal_termination(res)

        keyword_scores = defaultdict(float)
        for val in model.f:
            score = model.f[val].value
            if score <= 0:
                continue
            kw1, kw2 = val
            if kw1 == "source" or kw2 == "early_exit":
                continue
            keyword_scores[kw1] += score
        
        # predicted_keyphrases = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:10]
        predicted_keyphrases = rank_keywords(keyword_scores, top_n=10)
        new_predictions.append(predicted_keyphrases)

        # print(keyword_scores)
        # print(new_predictions)
        # raise

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
        json.dump(temp, open(f"{output_path}/scores-n={num_records}.json", "w"), indent=4)
        json.dump(new_predictions, open(f"{output_path}/predictions-n={num_records}.json", "w"), indent=4)



if __name__ == "__main__":
    # Example: python3 src/11-maxflow-dynamic.py
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--data", type=str, default="output/midas/ldkp3k")
    parser.add_argument("--output", type=str, default="output/max_flow/dynamic")

    args = parser.parse_args()
    # Get all the variables
    data_path = args.data
    output_path = args.output

    max_flow(data_path, output_path, max_keyphrases=10, num_records=None)
