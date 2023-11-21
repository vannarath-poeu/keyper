import argparse
import pathlib
import os
import json
import copy
from transformers import pipeline

from utils.evaluation import evaluate
from utils.nlp import stem_keywords
from utils.keybart import keybart, KeyphraseGenerationPipeline

def summarize(summarizer, sections):
        word_count = sum([len(sec) for sec in sections])
        if word_count < 256:
            joined_sections = []
            for sec in sections:
                joined_sections.extend(sec)
            return joined_sections
        if word_count > 768:
            half = len(sections) // 2
            summarized_sections = summarize(summarizer, sections[:half]) + summarize(summarizer, sections[half:])
            joined_sections = [" ".join(sec) for sec in summarized_sections]
        else:
            joined_sections = [" ".join(sec) for sec in sections]
        article = " ".join(joined_sections).strip()
        result = summarizer(article, do_sample=False, truncation=True)
        summary_text = result[0]['summary_text']
        return [summary_text.split(" ")]

def bart_score(
    dataset_name: str,
    data_path: str,
    output_path: str,
):
    # Load dataset
    dataset_path = f"{data_path}/{dataset_name}"
    test_jsonl = f"{dataset_path}/test.jsonl"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    dataset_output_path = f"{output_path}/{dataset_name}"
    summary_output_path = f"{dataset_output_path}/summary"
    pathlib.Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(summary_output_path).mkdir(parents=True, exist_ok=True)

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
    model_name = "bloomberg/KeyBART"
    generator = KeyphraseGenerationPipeline(model_name=model_name, truncation=True)

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    for i in range(num_docs):
        test[i] = json.loads(test[i])

        # Dataset specific
        if dataset_name == "vannarathp/segmented-kptimes":
            id = test[i]["id"]
            sections = [sec for sec in test[i]["document"]]
            abstractive_keyphrases = test[i]["abstractive_keyphrases"]
            extractive_keyphrases = test[i]["extractive_keyphrases"]
        else:
            raise NotImplementedError
        
        if os.path.exists(f"{summary_output_path}/{id}.json"):
            with open(f"{summary_output_path}/{id}.json", "r") as f:
                summarized_doc = json.load(f)
        else:
            summarized_doc = summarize(summarizer, sections)
            with open(f"{summary_output_path}/{id}.json", "w") as f:
                json.dump(summarized_doc, f, indent=4)
        
        doc = " ".join(summarized_doc[0])

        predicted_keyphrases = keybart(generator, doc, top_n=10)
        print(predicted_keyphrases)
        
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
        json.dump(temp, open(f"{dataset_output_path}/bart-keybart.json", "w"), indent=4)
        json.dump(predictions, open(f"{dataset_output_path}/bart-keybart-preds.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/12-bart-keybart.py --dataset vannarathp/segmented-kptimes
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

    bart_score(dataset_name, data_path, output_path)
