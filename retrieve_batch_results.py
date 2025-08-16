from utils import *
load_dotenv()

from openai import OpenAI
client = OpenAI()

file_name = "data/batch_tasks_tickers.jsonl"


batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

# Checking batch status
# Note: this can take up to 24h, but it will usually be completed faster.
# You can continue checking until the status is 'completed'.

batch_job = client.batches.retrieve(batch_job.id)
print(batch_job.status)
print(batch_job.request_counts)

# # Retrieving results
# result_file_id = batch_job.output_file_id
# result = client.files.content(result_file_id).content
# result_file_name = "data/batch_job_results_movies.jsonl"

# with open(result_file_name, 'wb') as file:
#     file.write(result)

# # Loading data from saved file
# results = []
# with open(result_file_name, 'r') as file:
#     for line in file:
#         # Parsing the JSON string into a dict and appending to the list of results
#         json_object = json.loads(line.strip())
#         results.append(json_object)


# # Reading results
# # Reminder: the results are not in the same order as in the input file. Make sure to check the custom_id to match the results against the input requests

# # Reading only the first results
# for res in results[:5]:
#     task_id = res['custom_id']
#     # Getting index from task id
#     index = task_id.split('-')[-1]
#     result = res['response']['body']['choices'][0]['message']['content']
#     movie = df.iloc[int(index)]
#     description = movie['Overview']
#     title = movie['Series_Title']
#     print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
#     print("\n\n----------------------------\n\n")

