from utils.evaluation import evaluate
from utils.nlp import stem_keywords
import json
import os
import pathlib
import copy

def validate():
    file_path = "output/midas/ldkp3k/keyper-similarity-temp-preds.json"
    test_jsonl = f"data/midas/ldkp3k/test.jsonl"
    output_path = "output/midas/ldkp3k"

    with open(file_path) as f:
        data = json.load(f)

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    with open(test_jsonl, "r") as f:
        test = f.readlines()
    
    num_records = len(data)
    print(f"Number of records: {num_records}")

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

    processed = 0
    for i in range(num_records):
        sections = [s for s in data[i]]
        phrases = []
        for s in sections:
            phrases.extend([kp[0] for kp in s])
        test[i] = json.loads(test[i])
        abstractive_keyphrases = test[i]["abstractive_keyphrases"]
        extractive_keyphrases = test[i]["extractive_keyphrases"]

        abstractive_keyphrases = stem_keywords(abstractive_keyphrases)
        extractive_keyphrases = stem_keywords(extractive_keyphrases)
        combined_keyphrases = abstractive_keyphrases + extractive_keyphrases

        predicted_keyphrases = stem_keywords(phrases)

        # print(abstractive_keyphrases)
        # print(extractive_keyphrases)
        # print(predicted_keyphrases)

        for k in [5, 10]:
            a_intersect = len(set(predicted_keyphrases) & set(abstractive_keyphrases))
            P = (min(k, a_intersect) / len(abstractive_keyphrases)) if len(abstractive_keyphrases) > 0 else 0
            R = min(k, a_intersect) / k
            F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
            results["abstractive"][f"precision@{k}"] += P
            results["abstractive"][f"recall@{k}"] += R
            results["abstractive"][f"fscore@{k}"] += F

        for k in [5, 10]:
            e_intersect = len(set(predicted_keyphrases) & set(extractive_keyphrases))
            P = (min(k, e_intersect) / len(extractive_keyphrases)) if len(extractive_keyphrases) > 0 else 0
            R = min(k, e_intersect) / k
            F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
            results["extractive"][f"precision@{k}"] += P
            results["extractive"][f"recall@{k}"] += R
            results["extractive"][f"fscore@{k}"] += F

        for k in [5, 10]:
            c_intersect = len(set(predicted_keyphrases) & set(combined_keyphrases))
            P = (min(k, c_intersect) / len(combined_keyphrases)) if len(combined_keyphrases) > 0 else 0
            R = min(k, c_intersect) / k
            F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
            results["combined"][f"precision@{k}"] += P
            results["combined"][f"recall@{k}"] += R
            results["combined"][f"fscore@{k}"] += F
        
        processed += 1
        print(f"Processed {processed} documents", end="\r")

        temp = copy.deepcopy(results)

        for k in temp.keys():
            for score in temp[k].keys():
                temp[k][score] /= (i+1)
        temp["num_docs"] = processed
        json.dump(temp, open(f"{output_path}/keyper-similarity-validation.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/04-keyper-similarity-validation.py
    validate()