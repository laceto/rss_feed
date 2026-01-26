
from dotenv import load_dotenv, find_dotenv 
load_dotenv()

from openai import OpenAI
client = OpenAI()
import json


batch_job = client.batches.retrieve('batch_6977e730ee208190995568e837e3fd5a')

# # # Retrieving results
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
result_file_name = "batch_job_results_rss_feeds.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)

# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)

with open("data/rss_feeds.jsonl", "w") as f: 
    for row in results: f.write(json.dumps(row) + "\n")

import rich
# rich.print(results[0])

# res = results[0]
for res in results:
    rich.print(res['response']['body']['choices'][0]['message']['content'])
# for res in result[:2]:
#     rich.print(res['response'])
    # rich.print(res['response']['body']['choices'][0]['message']['content'])