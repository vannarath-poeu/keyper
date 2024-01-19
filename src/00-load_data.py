from datasets import load_dataset
import argparse
import pathlib

def download_dataset(
    dataset_name: str,
    subset_name: str,
    output_path: str,
    splits: list,
):
    # Load dataset
    dataset = load_dataset(dataset_name, subset_name)

    dataset_output_path = f"{output_path}/{dataset_name}"
    pathlib.Path(dataset_output_path).mkdir(parents=True, exist_ok=True)

    for split in splits:
        dataset[split].to_json(f"{dataset_output_path}/{split}.jsonl")

if __name__ == "__main__":
    # Example: python3 src/00-load_data.py --dataset midas/kp20k --subset raw
    # Or python3 src/00-load_data.py --dataset midas/ldkp3k --subset small
    # Or python3 src/00-load_data.py --dataset midas/inspec --subset generation
    # Or python3 src/00-load_data.py --dataset midas/semeval2010 --subset raw --splits train,test
    # Or python3 src/00-load_data.py --dataset midas/nus --subset raw --splits test
    # Or python3 src/00-load_data.py --dataset midas/krapivin --subset raw --splits test
    # Or python3 src/00-load_data.py --dataset midas/kptimes --subset raw
    # Or python3 src/00-load_data.py --dataset midas/openkp --subset raw
    # Or python3 src/00-load_data.py --dataset midas/ldkp10k --subset small
    # Or python3 src/00-load_data.py --dataset vannarathp/segmented-kptimes
    # Or python3 src/00-load_data.py --dataset vannarathp/segmented-ldkp
    # Or python3 src/00-load_data.py --dataset vannarathp/segmented-openkp
    
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--splits", type=str, default=None)

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    subset_name = args.subset
    output_path = args.output

    if args.splits is not None:
        splits = args.splits.split(",")
    else:
        splits = ["train", "test", "validation"]

    download_dataset(dataset_name, subset_name, output_path, splits)
