import json
from nltk.stem.porter import PorterStemmer

def fscore(precision: float, recall: float):
    return (2 * precision * recall) / (precision + recall)

def precision_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0
    precision = 0
    for e in labels:
        if e in predictions:
            precision += 1
    precision /= len(predictions)
    return precision

def recall_at_k(predictions: list, labels: list, k=10):
    predictions = predictions[:min(k, len(predictions))]
    if len(predictions) == 0 or len(labels) == 0:
        return 0
    recall = 0
    for e in labels:
        if e in predictions:
            recall += 1
    recall /= len(labels)
    return recall

def evaluate():
    with open("temp/silver.json", "r") as f:
        for _, line in enumerate(f.readlines()):
            silver = json.loads(line)
    with open(f"data/midas/kp20k/train.json", "r") as f:
        extractive_precision = 0
        abstractive_precision = 0
        for i, line in enumerate(f.readlines()):
            if i >= 2000:
                break
            data = json.loads(line)
            phrases = silver[str(i)]
            extractive_keyphrases = data["extractive_keyphrases"]
            abstractive_keyphrases = data["abstractive_keyphrases"]
            extractive_precision += recall_at_k(phrases, extractive_keyphrases, 100)
            abstractive_precision += recall_at_k(phrases, abstractive_keyphrases, 100)
        print("extractive", extractive_precision / 2000)
        print("abstractive", abstractive_precision / 2000)

if __name__ == "__main__":
    evaluate()
