
import json
import rich

path = "data/rss_feeds.jsonl"

def read_jsonl_file(path):
    # Loading data from saved file
    results = []
    with open(path, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip()) 
            json_object = json_object['response']['body']['choices'][0]['message']['content']  # Extracting the 'response' field
            results.append(json_object)
    return results

result = read_jsonl_file(path)   
# rich.print_json(data=result)
                
import json
import pandas as pd

all_rows = []

for item in result:
    # Convert JSON string → Python dict
    obj = json.loads(item)

    # Extract sectors if present
    if "sectors" in obj:
        all_rows.extend(obj["sectors"])

# Convert to DataFrame
df = pd.DataFrame(all_rows)
rich.print(df)


# rich.print(result)
