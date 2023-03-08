import requests
import json

with open("test.json", "r") as f:
  raw_test = f.read()
  raw_records = raw_test.split("\n")
  records = []
  for record in raw_records[1:]:
    try:
      records.append(json.loads(record))
    except:
      print("Not a json")

  solr_url = "http://localhost:8983/solr"

  resp = requests.post(
    f"{solr_url}/inspec_test/update",
    json=[{
      "id": record["id"],
      "document": " ".join(record["document"])
    } for record in records]
  )
  print(resp.status_code)

  requests.post(f"{solr_url}/admin/cores?action=RELOAD&core=inspec_test")