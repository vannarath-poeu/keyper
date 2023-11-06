import statistics
import json

def main():
    dataset_list = ["openkp"]
    for dataset_name in dataset_list:
        dataset_output_path = f"data/midas/{dataset_name}"
        for split in ["train", "test", "validation"]:
            sample_count = 0
            with open(f"{dataset_output_path}/{split}.jsonl", "r") as f:
                dataset = f.readlines()
            dataset = [json.loads(d) for d in dataset]
            for data in dataset:
                if sample_count >= 5:
                    break
                if len(data["document"]) < 5000:
                    continue
                sample_count += 1
                with open(f"output/midas/{dataset_name}/{id}.txt", "w") as out:
                    out.write(" ".join(data["document"]).strip())
                # if "other_metadata" in data and "abstract" in data["other_metadata"]:
                #     id = data["id"]
                #     with open(f"output/midas/{dataset_name}/{id}.txt", "w") as out:
                #         out.write(data["other_metadata"]["abstract"])
    

if __name__ == "__main__":
    main()