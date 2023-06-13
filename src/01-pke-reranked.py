import argparse
import pathlib
import os
import json
import pke

from utils.evaluation import evaluate
from utils.nlp import stem_keywords

def position_rank(doc, top_n=10):
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrase_score = extractor.get_n_best(n=top_n)
    return keyphrase_score

def pke_score(
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
    
    assert len(test) > 0, f"File {test_jsonl} is empty"

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

    for i in range(num_docs):
        test[i] = json.loads(test[i])

        # Dataset specific
        if dataset_name == "midas/ldkp3k":
            sections = []
            for j, section in enumerate(test[i]["sections"]):
                if section.lower() != "abstract":
                    sections.append(" ".join(test[i]["sec_text"][j]))
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        else:
            raise NotImplementedError
        
        # Create a list to store all keyphrases
        predicted_keyphrases = []
        for section in sections: 
            # Add top 10 keyphrases from each section
            predicted_keyphrases.extend(position_rank(section, top_n=10))
        # Rerank the keyphrases
        predicted_keyphrases.sort(key=lambda x: x[1], reverse=True)
        predicted_keyphrases = [k[0] for k in predicted_keyphrases][:10]

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

    # Average results
    for k in results.keys():
        for score in results[k].keys():
            results[k][score] /= num_docs
    # Write results to file
    json.dump(results, open(f"{dataset_output_path}/position_rank-reranked.json", "w"), indent=4)
    json.dump(predictions, open(f"{dataset_output_path}/position_rank-reranked-preds.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/01-pke-reranked.py --dataset midas/ldkp3k
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

    pke_score(dataset_name, data_path, output_path)
