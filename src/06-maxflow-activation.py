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
        if top_keywords[sorted_keywords[i]] == 0:
            continue
        for j in range(i+1, len(sorted_keywords)):
            if stemmed_keywords_map[sorted_keywords[i]] == stemmed_keywords_map[sorted_keywords[j]]:
                top_keywords[sorted_keywords[i]] += top_keywords[sorted_keywords[j]]
                top_keywords[sorted_keywords[j]] = 0
    
    predicted_keyphrases = sorted(top_keywords, key=top_keywords.get, reverse=True)[:top_n]
    return predicted_keyphrases

def max_flow(
    data_path: str,
    output_path: str,
    max_activation=10,
):
    prediction_file_path = f"{data_path}/keyper-similarity-temp-preds.json"
    similarity_file_path = f"{data_path}/keyper-similarity-temp-sims.json"
    assert os.path.exists(prediction_file_path), f"File {prediction_file_path} does not exist"
    assert os.path.exists(similarity_file_path), f"File {similarity_file_path} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    # Load prediction and similarity files
    with open(prediction_file_path) as f:
        predictions = json.load(f)

    with open(similarity_file_path) as f:
        similarities = json.load(f)

    new_predictions = []

    num_records = 1 #len(predictions)
    for i in range(num_records):
        model = pyo.ConcreteModel("max_flow")
        keyword_pair_similarity = defaultdict(float)
        for similarity in similarities[i]:
            kw0, kw1, sim = similarity
            keyword_pair_similarity[(kw0, kw1)] = sim

        section_keywords = [sk[:max_activation] for sk in predictions[i]]
        num_sections = len(section_keywords)

        # nodes
        nodes = set(["source", "sink"])
        # edges
        edges = defaultdict(float)
        sections = {}

        add_source = True
        for si, _ in enumerate(section_keywords):
            if len(section_keywords[si]) < 1:
                continue
            sections[si] = []
            for sj, w1 in enumerate(section_keywords[si]):
                kw1 = w1[0]
                node_1 = f"{si}_{sj}_{kw1}"
                nodes.add(node_1)
                sections[si].append(node_1)
                if add_source:
                    edges[("source", node_1)] = 10_000
                if si + 1 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 1]):
                        kw2 = w2[0]
                        node_2 = f"{si + 1}_{sk}_{kw2}"
                        # if (kw1, kw2) not in keyword_pair_similarity:
                        #     print(f"Missing similarity for {kw1} and {kw2}")
                        #     print(si, si + 1)
                        #     continue
                        similarity = keyword_pair_similarity[(kw1, kw2)]
                        edges[(node_1, node_2)] = similarity
                if si + 2 < num_sections:
                    for sk, w2 in enumerate(section_keywords[si + 2]):
                        kw2 = w2[0]
                        node_2 = f"{si + 2}_{sk}_{kw2}"
                        # if (kw1, kw2) not in keyword_pair_similarity:
                        #     print(f"Missing similarity for {kw1} and {kw2}")
                        #     print(si, si + 2)
                        #     continue
                        similarity = keyword_pair_similarity[(kw1, kw2)]
                        edges[(node_1, node_2)] = similarity
            add_source = False
        
        for si in reversed(range(num_sections)):
            if len(section_keywords[si]) < 1:
                continue
            for sj, w1 in enumerate(section_keywords[si]):
                kw1 = w1[0]
                node_1 = f"{si}_{sj}_{kw1}"
                edges[node_1, "sink"] = 10_000
            break
        
        nodes = list(nodes)
        node_products = [(s, t) for s in nodes for t in nodes]
        model.f = pyo.Var(node_products, domain=pyo.NonNegativeReals)
        model.a = pyo.Var(nodes, domain=pyo.Binary)

        # Maximize the flow into the sink nodes
        def total_rule(model):
            return sum(model.a[n[0]] * model.f[n] for n in node_products)

        model.total = pyo.Objective(rule=total_rule, sense=pyo.maximize)

        # Enforce an upper limit on the flow across each edge
        def limit_rule(model, s, e):
            return (model.a[s] * model.f[(s, e)]) <= edges.get((s, e), 0)

        model.limit = pyo.Constraint(nodes, nodes, rule=limit_rule)

        # Enforce a section rule
        def section_rule(model, section):
            return sum(model.a[n] for n in sections[section]) <= 5

        model.section = pyo.Constraint(sections.keys(), rule=section_rule)

        # Enforce flow through each node
        def flow_rule(model, node):
            inFlow  = sum(model.f[(source, node)] for source in nodes)
            outFlow = sum(model.f[(node, dest)] for dest in nodes)
            if node == "source" or node == "sink":
                return pyo.Constraint.Skip
            return inFlow == outFlow

        model.flow = pyo.Constraint(nodes, rule=flow_rule)

        # solver = pyo.SolverFactory('glpk')  # "glpk"
        solver = pyo.SolverFactory('ipopt')  # "cbc"
        solver.options['max_iter']= 100 #number of iterations you wish

        res = solver.solve(model)

        pyo.assert_optimal_termination(res)

        keyword_scores = defaultdict(float)
        for val in model.f:
            score = model.f[val].value
            if score <= 0:
                continue
            node1, node2 = val
            if node1 == "source" or node2 == "sink":
                continue
            _, _, kw1 = node1.split("_")
            _, _, kw2 = node2.split("_")
            keyword_scores[kw1] += score
        
        # predicted_keyphrases = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:10]
        predicted_keyphrases = rank_keywords(keyword_scores, top_n=10)
        new_predictions.append(predicted_keyphrases)
        print(f"Processed {i+1} documents", end="\r")

        json.dump(new_predictions, open(f"{output_path}/predictions.json", "w"), indent=4)



if __name__ == "__main__":
    # Example: python3 src/06-maxflow-activation.py
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--data", type=str, default="output/midas/ldkp3k")
    parser.add_argument("--output", type=str, default="output/max_flow/activation")

    args = parser.parse_args()
    # Get all the variables
    data_path = args.data
    output_path = args.output

    max_flow(data_path, output_path, max_activation=10)
