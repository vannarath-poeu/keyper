import statistics
import json

def main():
    dataset_list = ["openkp", "kptimes"]
    for dataset_name in dataset_list:
        dataset_output_path = f"data/midas/{dataset_name}"
        with open(f"output/midas/{dataset_name}/stats.txt", "w") as out:
            for split in ["train", "test", "validation"]:
                out.write(split + ":\n")
                with open(f"{dataset_output_path}/{split}.jsonl", "r") as f:
                    dataset = f.readlines()
                dataset = [json.loads(d) for d in dataset]
                length_list = [len(d["document"]) for d in dataset]

                out.write("num docs: " + str(len(length_list)) + "\n")
                out.write("max length: " + str(max(length_list)) + "\n")
                out.write("min length: " + str(min(length_list)) + "\n")
                out.write("average length: " + str(sum(length_list)/len(length_list)) + "\n")
                out.write("mean length: " + str(statistics.mean(length_list)) + "\n")
                out.write("standard deviation: " + str(statistics.stdev(length_list)) + "\n")
                for min_length in range(1000, 10000, 1000):
                    out.write("documents above " + str(min_length) + ": " + str(len([l for l in length_list if l > min_length])) + "\n")
                out.write("\n")
    

if __name__ == "__main__":
    main()