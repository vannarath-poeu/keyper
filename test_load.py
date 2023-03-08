from datasets import load_dataset
from ast import literal_eval

dataset = load_dataset(
    "csv",
    data_files={
        "train": "rouge_train.csv",
        "test": "test.csv",
    }
)

for subset in ["train", "test"]:
  for record in dataset[subset]:
    for col in ["document", "extractive_keyphrases", "doc_bio_tags", "abstractive_keyphrases", "other_metadata"]:
      dataset[subset][col] = literal_eval(dataset[subset][col])