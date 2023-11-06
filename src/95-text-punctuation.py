import statistics
import json
from deepmultilingualpunctuation import PunctuationModel


def main():
    dataset_name = "openkp"
    id = "sample"
    with open(f"output/midas/{dataset_name}/{id}.txt", "r") as f:
        doc = f.readlines()
    
    model = PunctuationModel()
    restored_text = model.restore_punctuation(doc[0])

    with open(f"output/midas/{dataset_name}/{id}-temp.txt", "w") as f:
        f.write(restored_text)

if __name__ == "__main__":
    main()