import argparse
import pathlib
import os
import json
from nltk.stem.porter import PorterStemmer

from utils.evaluation import evaluate
from utils.nlp import stem_keywords
from utils.keybart import keybart, KeyphraseGenerationPipeline

def keybart_score(
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

    model_name = "bloomberg/KeyBART"
    generator = KeyphraseGenerationPipeline(model_name=model_name, truncation=True)

    for i in range(num_docs):
        test[i] = json.loads(test[i])

        # Dataset specific
        if dataset_name in [
            "midas/kp20k",
            "midas/nus",
            "midas/inspec",
            "midas/krapivin",
            "midas/semeval2010",
        ]:
            doc = " ".join(test[i]["document"])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        elif dataset_name == "midas/ldkp3k":
            sections = []
            for j, section in enumerate(test[i]["sections"]):
                if section.lower() != "abstract":
                    sections.append(" ".join(test[i]["sec_text"][j]))
            doc = " ".join([s for s in sections])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        elif dataset_name in ["vannarathp/segmented-kptimes", "vannarathp/segmented-openkp"]:
            sections = []
            for section in test[i]["document"]:
                sections.append(" ".join(section))
            doc = " ".join([s for s in sections])
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        else:
            raise NotImplementedError

        predicted_keyphrases = keybart(generator, doc, top_n=10)
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

    for k in results.keys():
        for score in results[k].keys():
            results[k][score] /= num_docs
    json.dump(results, open(f"{dataset_output_path}/keybart.json", "w"), indent=4)
    json.dump(predictions, open(f"{dataset_output_path}/keybart-preds.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/02-keybart.py --dataset midas/kp20k
    # Or python3 src/02-keybart.py --dataset midas/ldkp3k
    # Or python3 src/02-keybart.py --dataset midas/inspec
    # Or python3 src/02-keybart.py --dataset midas/semeval2010
    # Or python3 src/02-keybart.py --dataset midas/nus
    # Or python3 src/02-keybart.py --dataset midas/krapivin
    # Or python3 src/02-keybart.py --dataset vannarathp/segmented-kptimes
    # Or python3 src/02-keybart.py --dataset vannarathp/segmented-openkp
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

    keybart_score(dataset_name, data_path, output_path)
