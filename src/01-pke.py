import argparse
import pathlib
import os
import json
import pke
import spacy
from nltk.stem.porter import PorterStemmer

def list_jsonl(path):
    files = []
    for r, _, f in os.walk(path):
        for file in f:
            if '.jsonl' in file:
                files.append(os.path.join(r, file))
    return files

def position_rank(doc, top_n=10):
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrase_score = extractor.get_n_best(n=top_n)
    return [k for k, _ in keyphrase_score]

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

def pke_score(
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

        num_docs = 10 # len(test)

        for i in range(num_docs):
            test[i] = json.loads(test[i])

            # Dataset specific
            if dataset_name in [
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
            else:
                raise NotImplementedError

            combined_keyphrases = abstractive_keyphrases + extractive_keyphrases
            predicted_keyphrases = position_rank(doc, top_n=10)

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
        json.dump(results, open(f"{dataset_output_path}/topic_rank.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/00-load_data.py --dataset midas/kp20k --subset raw
    # Or python3 src/01-pke.py --dataset midas/ldkp3k
    # Or python3 src/01-pke.py --dataset midas/inspec
    # Or python3 src/01-pke.py --dataset midas/semeval2010
    # Or python3 src/01-pke.py --dataset midas/nus
    # Or python3 src/01-pke.py --dataset midas/krapivin
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
