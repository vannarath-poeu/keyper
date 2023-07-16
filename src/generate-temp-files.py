import os
import json

def main():
  data_path = "output/midas/ldkp3k"

  small_set_size = 1000

  prediction_file_path = f"{data_path}/keyper-similarity-temp-preds.json"
  similarity_file_path = f"{data_path}/keyper-similarity-temp-sims.json"
  assert os.path.exists(prediction_file_path), f"File {prediction_file_path} does not exist"
  assert os.path.exists(similarity_file_path), f"File {similarity_file_path} does not exist"

  small_prediction_file_path = f"{data_path}/keyper-similarity-temp-preds-{small_set_size}.json"
  small_similarity_file_path = f"{data_path}/keyper-similarity-temp-sims-{small_set_size}.json"

  # Load prediction and similarity files
  with open(prediction_file_path) as f:
      predictions = json.load(f)
  
  json.dump(predictions[:small_set_size], open(small_prediction_file_path, "w"))

  with open(similarity_file_path) as f:
      similarities = json.load(f)
  
  json.dump(similarities[:small_set_size], open(small_similarity_file_path, "w"))

if __name__ == "__main__":
  main()