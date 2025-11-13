from utils import *
load_dotenv()

# from openai.lib._parsing._completions import type_to_response_format_param
# from pydantic import BaseModel
from openai import OpenAI

# class Step(BaseModel):
#     explanation: str
#     output: str

# class MathResponse(BaseModel):
#     steps: list[Step]
#     final_answer: str

client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful math tutor."},
#         {"role": "user", "content": "solve 8x + 31 = 2"},
#     ],
#     response_format=type_to_response_format_param(MathResponse),
# )

# message = completion.choices[0].message
# print(message.content)

# Now build the prompt template string with escaped braces

from typing import List, Optional
from pydantic import BaseModel, Field
from openai.lib._parsing._completions import type_to_response_format_param

class Overview(BaseModel):
    restated_inputs: str = Field(..., description="Clear restatement of the provided input values")
    market_snapshot: str = Field(..., description="Short one-sentence market snapshot")

class MovingAverageStructureAnalysis(BaseModel):
    alignment: str = Field(..., description="MA alignment classification: Bullish, Bearish, or Mixed")
    spread_interpretation: str = Field(..., description="Interpretation of the percentage spreads between moving averages")

class RSIAndMomentumAssessment(BaseModel):
    rsi_classification: str = Field(..., description="RSI classification: Oversold, Neutral, Overbought")
    rsi_ma_interpretation: str = Field(..., description="Interpretation of RSI combined with MA classification")

class TrendStrengthRSIMatrix(BaseModel):
    trend_strength: str = Field(..., description="Trend strength classification: Weak, Moderate, Strong")
    rsi_zone: str = Field(..., description="RSI zone: Oversold, Neutral, Overbought")
    context: str = Field(..., description="Historical or statistical context for this combination")

class OverallMarketContext(BaseModel):
    trend_stage: str = Field(..., description="Trend stage: early, mid, late")
    volatility: str = Field(..., description="Volatility description, e.g., ATR rising or falling")
    price_vs_mas: str = Field(..., description="Price position relative to moving averages")
    sector_market_alignment: str = Field(..., description="Alignment with sector or overall market trend")

class RiskOpportunityAssessment(BaseModel):
    continuation_vs_pullback: str = Field(..., description="Evaluation of continuation vs pullback likelihood")
    warning_signs: Optional[List[str]] = Field(None, description="List of warning signs such as extreme spreads, RSI divergence, ATR spikes")

class TacticalTradeConsiderations(BaseModel):
    long_position: Optional[str] = Field(None, description="Stop placement, scaling out, risk control if already long")
    flat_position: Optional[str] = Field(None, description="Entry trigger conditions if flat")
    short_biased: Optional[str] = Field(None, description="Conditions for short bias and risk notes")

class SummarySignal(BaseModel):
    classification: str = Field(..., description="Summary signal classification")
    key_justifications: List[str] = Field(..., description="1-2 bullet points with key justifications")

class TraderAnalysis(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    overview: Overview
    moving_average_structure_analysis: MovingAverageStructureAnalysis
    rsi_and_momentum_assessment: RSIAndMomentumAssessment
    trend_strength_rsi_matrix: TrendStrengthRSIMatrix
    overall_market_context: OverallMarketContext
    risk_opportunity_assessment: RiskOpportunityAssessment
    tactical_trade_considerations: TacticalTradeConsiderations
    summary_signal: SummarySignal

system_prompt_template = f"""
You are an experienced algorithmic trader’s AI assistant specializing in trend analysis using moving averages, RSI, and related momentum indicators.
Your job is to take numerical inputs (MA values, RSI, and optionally price, ATR, volume) and produce a clear, comprehensive, structured 
report that a trader can immediately use for decision-making.


You will receive user input as a dictionary with the following keys:
- 'ticker': the stock ticker symbol
- 'rclose': the relative closing price of the stock compared to its benchmark
- 'ema_st': the short-term exponential moving average (50-day)
- 'ema_mt': the medium-term exponential moving average (100-day)
- 'ema_lt': the long-term exponential moving average (150-day)
- 'spread_stmt': the percentage spread between short-term and medium-term moving averages defined as (ema_st - ema_mt) / ema_st * 100
- 'spread_mtlt': the percentage spread between medium-term and long-term moving averages defined as (ema_mt - ema_lt) / ema_lt * 100
- 'spread_stlt': the percentage spread between short-term and long-term moving averages defined as (ema_st - ema_lt) / ema_lt * 100
- 'rsi': the latest Relative Strength Index value

Your task is to analyze this data and provide a comprehensive breakdown including:

1. Overview
    - Restate the provided inputs clearly.
    - Give a short one-sentence market snapshot.
2. Moving Average Structure Analysis
Alignment rules:
    - Bullish alignment = MA50 > MA100 > MA150
    - Bearish alignment = MA50 < MA100 < MA150
    - Mixed = no clear order 

Spread Interpretation

3. RSI & Momentum Assessment

Classify RSI:
    Oversold (<30) → Potential bounce zone.
    Neutral (30–70) → Healthy momentum range.
    Overbought (>70) → Strong momentum but higher risk of pullback.

Interpret RSI in combination with the MA classification.

4. Trend Strength vs Overbought Matrix  

Place the setup in the matrix:
    Trend: Weak / Moderate / Strong
    RSI Zone: Oversold / Neutral / Overbought
Provide historical or statistical context for what this combination usually means.

5. Overall Market Context
    Describe trend stage (early, mid, late), volatility (ATR rising/falling), price position vs MAs, and alignment with sector/market trend.

6. Risk & Opportunity Assessment

    - Evaluate continuation vs pullback likelihood based on classification tables and context.
    - Mention warning signs such as extreme spreads, RSI divergence, or ATR spikes.

    
7. Tactical Trade Considerations (Informational only — not advice)

    If already long: Stop placement, scaling out, risk control.
    If flat: Entry trigger conditions.
    If short-biased: Only if confirmation signals appear, noting trend strength risk.

8. Summary Signal
Classify into one of:
    Strong Bullish Momentum — Early Stage
    Strong Bullish Momentum — Late Stage / Overextended
    Neutral / Sideways
    Bearish Momentum
    Include 1–2 bullet points with key justifications.    

Tone & Style
Use professional trader language.
Avoid vague terms like “it might go up or down.”
Always give reasoning for each classification.
Be concise but thorough.
    
"""

# print(type_to_response_format_param(TraderAnalysis))

import json
from pathlib import Path

# 1. Load JSON file containing a list of input dicts
file_path = "./output_dict_list.txt"  # Your JSON file path
json_str = Path(file_path).read_text()

# 2. Parse the JSON string into a Python list of dicts
json_data = json.loads(json_str)



tasks = []
for data in json_data[:5]:
    
    description = data
    ticker = data['ticker']
    
    task = {
        "custom_id": f"task-{ticker}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            # "model": "gpt-4o-mini",
            "temperature": 0,
            "response_format": type_to_response_format_param(TraderAnalysis),
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_template
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)


# Creating the file

file_name = "data/batch_tasks_tickers.jsonl"

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



