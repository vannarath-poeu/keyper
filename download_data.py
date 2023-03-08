from datasets import load_dataset

# Dataset parameters
dataset_full_name = "midas/inspec"
dataset_subset = "raw"
dataset_document_column = "document"

# Load dataset
dataset = load_dataset(dataset_full_name, dataset_subset)

dataset["train"].to_json("train.json")
dataset["test"].to_json("test.json")

