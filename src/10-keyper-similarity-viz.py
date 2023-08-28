from utils.evaluation import evaluate
from utils.nlp import stem_keywords
import json
import os
import pathlib
import copy

def validate():
    file_path = "output/midas/ldkp3k/keyper-similarity-temp-preds.json"
    test_jsonl = f"data/midas/ldkp3k/test.jsonl"
    output_path = "output/midas/ldkp3k/keyper-similarity-viz"

    with open(file_path) as f:
        data = json.load(f)

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    with open(test_jsonl, "r") as f:
        test = f.readlines()
    
    num_records = len(data)
    print(f"Number of records: {num_records}")

    processed = 0
    for i in range(num_records):
        sections = [s for s in data[i]]
        test[i] = json.loads(test[i])
        abstractive_keyphrases = test[i]["abstractive_keyphrases"]
        extractive_keyphrases = test[i]["extractive_keyphrases"]

        abstractive_keyphrases = set(stem_keywords(abstractive_keyphrases))
        extractive_keyphrases = set(stem_keywords(extractive_keyphrases))

        with open(f"{output_path}/{i}.txt", "w") as f:
            for s in sections:
                phrases = [kp[0] for kp in s]
                stemmed_phrases = stem_keywords(phrases)

                for ii, p in enumerate(phrases):
                    sp = stemmed_phrases[ii]
                    if sp in abstractive_keyphrases:
                        f.write(f"(abs - {sp}) {p}")
                        # abstractive_keyphrases.remove(sp)
                    elif sp in extractive_keyphrases:
                        f.write(f"(ext - {sp}) {p}")
                        # extractive_keyphrases.remove(sp)
                    else:
                        f.write(p)
                    f.write("\t")

                f.write("\n\n")
        
        processed += 1
        print(f"Processed {processed} documents", end="\r")


if __name__ == "__main__":
    # Example: python3 src/10-keyper-similarity-viz.py
    validate()