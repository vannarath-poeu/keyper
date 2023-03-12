import requests
import pandas as pd
import json
from ast import literal_eval

df = pd.read_csv("train.csv")
df = df[["id", "document"]]
df['document'] = df['document'].apply(literal_eval)
result = df.to_json(orient="records")
parsed = json.loads(result)
with open("train.json", "w") as f:
  json.dump(parsed, f, indent=4)

with open("train.json", "r") as f:
  data = json.load(f)
  # for record in data:
  #   id = record["id"]
  #   txt = " ".join(record["document"])
  #   print(txt)
  #   break
  solr_url = "http://localhost:8983/solr"

  resp = requests.post(
    f"{solr_url}/inspec_training/update",
    json=[{
      "id": record["id"],
      "document": " ".join(record["document"])
    } for record in data]
  )
  print(resp.status_code)

  requests.post(f"{solr_url}/admin/cores?action=RELOAD&core=inspec_training")