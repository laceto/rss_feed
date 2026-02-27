from utils import *

file_list = get_file_paths('output', file_pattern='txt')
dfs = [pd.read_csv(file, sep='\t') for file in file_list]
dfs = pd.concat(dfs, ignore_index=True)
dfs['description'] = dfs['title'] + '. ' + dfs['description']
docs = df_to_docs(dfs, content_column='description', metadata_columns=['link', 'guid', 'type', 'id', 'sponsored', 'pubDate'])


# from sqlalchemy import create_engine
# from langchain.cache import SQLAlchemyCache
# from langchain_core.globals import set_llm_cache

# # Create a SQLite engine for a local file database
# engine = create_engine("sqlite:///./cache/langchain_cache.db")
# # Set the LLM cache to use SQLAlchemyCache with the engine
# set_llm_cache(SQLAlchemyCache(engine))

load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

# # Define the Pydantic model for structured output
class Ciaone(BaseModel):
    company: str = Field(..., description="Name of the listed company")
    # sector: Literal["Commercial Services", "Communications", "Consumer Durables", "Consumer Non-Durables", "Consumer Services", "Distribution Services", "Electronic Technology", "Energy Minerals", "Finance", "Health Services", "Health Technology", "Industrial Services", "Non-Energy Minerals", "Process Industries", "Producer Manufacturing", "Retail Trade", "Technology Services", "Transportation", "Utilities"] = Field(..., description="Name of the sector")
    # institution: str = Field(..., description="Name of the institution")
    # trading_decision: str = Field(..., description="Trading decision: long or short")
    # signal: str = Field(..., description="Signal: buy or sell")
    # motivation: str = Field(..., description="Reason for the trading decision")
    # news_topic: str = Field(..., description="Topic of the news, e.g., balance sheet, market share, new appointments, or other")

    
# # Create the base JSON output parser
base_parser = JsonOutputParser(pydantic_object=Ciaone)

# # Create the output-fixing parser wrapping the base parser and using an LLM to fix errors
llm_for_fixing = ChatOpenAI(temperature=0)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm_for_fixing)


# Define a rich persona in the system message with added expertise
system_message = SystemMessagePromptTemplate.from_template(
    """You are Ava, a sharp and insightful trader assistant with deep expertise in quantitative finance, advanced statistical models, and short selling techniques.
You provide clear, concise, and actionable investment insights based on news feeds.
Maintain a friendly, confident, and professional tone, making complex concepts accessible and useful."""
)


# # Define the human message template with instructions and JSON schema
human_message = HumanMessagePromptTemplate.from_template(
    """Extract the investment decision from the news feed below.

Output a JSON object with the following fields:
- "company_or_sector": The name of the listed company or the sector the news is about (string).
- "trading_decision": Either "long" or "short" (string).
- "signal": Either "buy" or "sell" (string).
- "motivation": A concise explanation of the reason for the trading decision, based on the news (string).
- "news_topic": What the news is about, such as "balance sheet", "market share", "new appointments", or "other" (string).

If no investment decision is present, respond with an empty JSON object: {{}}.

News feed:
{news_feed}

JSON output:"""
)

# Create the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# Compose the chain: prompt -> LLM -> fixing parser
llm = ChatOpenAI(temperature=0)
chain = chat_prompt | llm | fixing_parser

# Example news feed
news = docs[100].page_content

# Invoke the chain
result = chain.invoke({"news_feed": news})
print(result)

