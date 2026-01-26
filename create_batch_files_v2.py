from utils import *
from openai import OpenAI
import json
load_dotenv()
client = OpenAI()

from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from openai.lib._parsing._completions import type_to_response_format_param
from openai.lib._pydantic import to_strict_json_schema

file_list = get_file_paths('output', file_pattern='.txt')
dfs = [pd.read_csv(file, sep='\t') for file in file_list]
dfs = [df.drop_duplicates(subset=["description"]) for df in dfs]
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["description"])
df['pubDate'] = pd.to_datetime(df['pubDate'])
df['pubDate'] = df['pubDate'].dt.strftime('%Y-%m-%d')
dfs = [g.copy() for _, g in df.groupby('pubDate')]
dfs = [ df.assign(description=df['title'] + ': ' + df['description']) for df in dfs ]
dfs = [ df.assign(description=df['pubDate'] + ': ' + df['description']) for df in dfs ]
docs = [df_to_docs(df, content_column='description', metadata_columns=['link', 'guid', 'type', 'id', 'sponsored', 'pubDate']) for df in dfs]
joined_contents = [".".join(doc.page_content for doc in sublist) for sublist in docs]
# joined_contents = joined_contents[:5]

[print(info) for info in joined_contents]

class Sector(BaseModel):
        Date: str = Field(..., description='The publication date of the news')
        Name: Literal[
            "Commercial Services",
            "Communications",
            "Consumer Durables",
            "Consumer Non-Durables",
            "Consumer Services",
            "Distribution Services",
            "Electronic Technology",
            "Energy Minerals",
            "Finance",
            "Health Services",
            "Health Technology",
            "Industrial Services",
            "Non-Energy Minerals",
            "Process Industries",
            "Producer Manufacturing",
            "Retail Trade",
            "Technology Services",
            "Transportation",
            "Utilities"
            ] = Field(..., description="Name of the sector")
        Outlook: str = Field(..., description='A sector outlook describes the expected future performance and conditions based on data, trends, and risks.')
        Catalyst: str = Field(..., description="A primary catalyst of a sector is the single most influential force that is expected to drive major change—positive or negative—across an entire industry.")
        Trading_insights: str = Field(..., description="Practical, actionable interpretation of the sector's expected future conditions—something that can guide investment or trading decisions.")
        Direction_momentum: Literal["Strength", "Weakness", "Volatility", "Rotation in favor", "Rotation out of favor"] = Field(..., description="The direction of momentum of a sector is the prevailing trend in how that sector's prices, performance, and investor sentiment are moving over a given period.")
        Sector_vs_market_position: Literal['Leading', 'Lagging', 'In Line'] = Field(..., description="Relative positioning of a sector to the overall market is the sector's performance and strength compared to the broader market, showing whether it is leading, lagging, or moving in line with the market trend.")

class MultiSectorAnalysis(BaseModel):
        """Analysis of multiple sectors from news feed."""
        sectors: List[Sector] = Field(
            ...,
            description="List of ALL relevant sectors found in the news (1-8 max). Only include sectors with clear evidence.",
            default_factory=list
        )

Structured_Response = to_strict_json_schema(MultiSectorAnalysis)

system_prompt_template = f"""
You are Ava, a sharp trader assistant. Analyze the FULL news feed and extract **ALL relevant sectors** mentioned.

    CRITICAL RULES:
    - Output ONLY valid JSON matching this schema provided
    - If NO sector: use empty [] 
    - Each sector must use a Name from the enum
    - Fill ALL required fields for each sector
    - No duplicate sectors
    - No extra text outside JSON    
"""



tasks = []
for i in range(len(joined_contents)): 
    cont = joined_contents[i]
 
    task = {
        "custom_id": f"task-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4.1-nano",
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "structured_response",
                  "schema": Structured_Response,
                  "strict": True
                }
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_template
                },
                {
                    "role": "user",
                    "content": cont
                }
            ],
        }
    }
    
    tasks.append(task)

# print(tasks[:3])
# Creating the file

file_name = "data/batch_tasks_rss_feeds.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')


# Uploading the file
batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

print(batch_file)

# Creating the batch job
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

print(batch_job.id)

