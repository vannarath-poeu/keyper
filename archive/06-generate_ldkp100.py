import argparse
import pathlib
from datasets import load_dataset

SEED = 100

def generate_dataset():
    dataset_name = "midas/ldkp3k"
    for split in ["train", "test", "validation"]:
        limit = 400 if split == "train" else 100
        data_path = f"data/{dataset_name}/{split}.json"
        ds = load_dataset("json", data_files=data_path)
        shuffled_dataset = ds["train"].shuffle(seed=SEED).select(range(limit))
        output_path = data_path.replace("ldkp3k", "ldkp100")
        shuffled_dataset.to_json(output_path)
        # with open(data_path, "r") as f:
        #     with open(output_path, "w") as f2:
        #         for i, line in enumerate(f.readlines()):
        #             if i >= limit:
        #                 break


if __name__ == "__main__":
    generate_dataset()
