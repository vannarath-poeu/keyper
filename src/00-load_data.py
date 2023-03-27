from datasets import load_dataset
import argparse
import pathlib

def download_dataset(dataset_name: str, subset_name: str, output_path: str):
    # Load dataset
    dataset = load_dataset(dataset_name, subset_name)

    dataset_output_path = f"{output_path}/{dataset_name}"
    pathlib.Path(dataset_output_path).mkdir(parents=True, exist_ok=True)

    for split in ["train", "test", "validation"]:
        dataset[split].to_json(f"{dataset_output_path}/{split}.json")

if __name__ == "__main__":
    # Example: python3 00-load_data.py --dataset midas/kp20k --subset raw
    # Or python3 00-load_data.py --dataset midas/ldkp3k --subset small
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output", type=str, default="data")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    subset_name = args.subset
    output_path = args.output

    download_dataset(dataset_name, subset_name, output_path)
