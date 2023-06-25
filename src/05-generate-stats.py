import argparse
import pathlib
import os
import json
import pke

def generate_stats(
    dataset_name: str,
    data_path: str,
    output_path: str,
):
    # Load dataset
    dataset_path = f"{data_path}/{dataset_name}"
    train_jsonl = f"{dataset_path}/train.jsonl"
    test_jsonl = f"{dataset_path}/test.jsonl"
    validation_jsonl = f"{dataset_path}/validation.jsonl"

    dataset_output_path = f"{output_path}/{dataset_name}"
    pathlib.Path(dataset_output_path).mkdir(parents=True, exist_ok=True)

    records = []
    files = []

    for file in [train_jsonl, test_jsonl, validation_jsonl]:
        if os.path.exists(file):
            files.append(file)
            with open(file, "r") as f:
                records.extend(f.readlines())
    
    assert len(records) > 0, f"File {train_jsonl} is empty"

    num_docs = len(records)

    stats = {
        "num_docs": num_docs,
        "num_extractive_keyphrases": 0,
        "num_abstractive_keyphrases": 0,
        "num_sections": 0,
        "num_sentences": 0,
        "num_words": 0,
        "num_has_abstract": 0,
        "num_start_with_intro": 0,
        "files": files,
    }

    for i in range(num_docs):
        records[i] = json.loads(records[i])

        # Dataset specific
        if dataset_name in [
            "midas/kp20k",
            "midas/nus",
            "midas/inspec",
            "midas/krapivin",
            "midas/semeval2010",
        ]:
            doc = " ".join(records[i]["document"])
            abstractive_keyphrases = records[i]["abstractive_keyphrases"]
            extractive_keyphrases = records[i]["extractive_keyphrases"]
            stats["num_words"] += len(records[i]["document"])
        elif dataset_name == "midas/ldkp3k":
            sections = []
            for j, section in enumerate(records[i]["sections"]):
                if section.lower() != "abstract":
                    sections.append(" ".join(records[i]["sec_text"][j]))
            doc = " ".join([s for s in sections])
            abstractive_keyphrases = records[i]["abstractive_keyphrases"]
            extractive_keyphrases = records[i]["extractive_keyphrases"]
            stats["num_sections"] += len(sections)
            stats["num_words"] += sum([len(s) for s in records[i]["sec_text"]])
            if "abstract" in records[i]["sections"]:
                stats["num_has_abstract"] += 1
            if records[i]["sections"][0].lower() == "introduction":
                stats["num_start_with_intro"] += 1
        else:
            raise NotImplementedError
        
        num_sentences = len(doc.split("."))
        stats["num_sentences"] += num_sentences
        
        stats["num_extractive_keyphrases"] += len(extractive_keyphrases)
        stats["num_abstractive_keyphrases"] += len(abstractive_keyphrases)

    # Write results to file
    json.dump(stats, open(f"{dataset_output_path}/stats.json", "w"), indent=4)


if __name__ == "__main__":
    # Example: python3 src/05-generate-stats.py --dataset midas/kp20k
    # Or python3 src/05-generate-stats.py --dataset midas/ldkp3k
    # Or python3 src/05-generate-stats.py --dataset midas/inspec
    # Or python3 src/05-generate-stats.py --dataset midas/semeval2010
    # Or python3 src/05-generate-stats.py --dataset midas/nus
    # Or python3 src/05-generate-stats.py --dataset midas/krapivin
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

    generate_stats(dataset_name, data_path, output_path)
