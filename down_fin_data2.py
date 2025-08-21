# import yfinance as yf
# import json

# def get_financial_statements(symbol: str) -> str:
#     """Retrieve key financial statement data from Yahoo Finance via yfinance."""
#     try:
#         stock = yf.Ticker(symbol)
#         financials = stock.financials
#         balance_sheet = stock.balance_sheet
#         shares_info = stock.get_shares_full(start=None, end=None)  # to get shares numbers

#         latest_year = financials.columns[0]  # most recent year

#         def safe_get(df, row_name):
#             """Safely retrieve a value from the DataFrame."""
#             try:
#                 return float(df.loc[row_name, latest_year]) if row_name in df.index else None
#             except Exception:
#                 return None

#         # Calculate derived metrics
#         total_assets = safe_get(balance_sheet, "Total Assets")
#         total_liabilities = safe_get(balance_sheet, "Total Liab")
#         minority_interest = safe_get(balance_sheet, "Minority Interest")
#         total_equity = safe_get(balance_sheet, "Total Equity Gross Minority Interest") or safe_get(balance_sheet, "Total Equity")
#         current_assets = safe_get(balance_sheet, "Total Current Assets") or safe_get(balance_sheet, "Current Assets")
#         current_liabilities = safe_get(balance_sheet, "Total Current Liabilities") or safe_get(balance_sheet, "Current Liabilities")
#         intangible_assets = safe_get(balance_sheet, "Intangible Assets") or 0

#         data = {
#             "symbol": symbol,
#             "period": str(latest_year.year) if hasattr(latest_year, "year") else str(latest_year),

#             # Balance Sheet
#             "TotalAssets": total_assets,
#             "TotalLiabilitiesNetMinorityInterest": (total_liabilities - minority_interest) if total_liabilities is not None else None,
#             "TotalCapitalization": total_equity + safe_get(balance_sheet, "Total Debt") if total_equity is not None else None,
#             "CommonStockEquity": total_equity,
#             "NetTangibleAssets": total_assets - intangible_assets - total_liabilities if total_assets is not None else None,
#             "WorkingCapital": current_assets - current_liabilities if current_assets is not None and current_liabilities is not None else None,
#             "InvestedCapital": total_equity + safe_get(balance_sheet, "Total Debt") - (safe_get(balance_sheet, "Cash And Cash Equivalents") or 0) if total_equity is not None else None,
#             "TotalDebt": safe_get(balance_sheet, "Total Debt"),

#             # Shares info
#             "OrdinarySharesNumber": shares_info.get("sharesOutstanding", None) if shares_info is not None else None,
#             "TreasurySharesNumber": shares_info.get("treasuryStock", None) if shares_info is not None else None,
#         }

#         return json.dumps(data, indent=2)

#     except Exception as e:
#         return f"Error: {str(e)}"


# # Example usage
# if __name__ == "__main__":
#     print(get_financial_statements("AAPL"))




# import yfinance as yf
# import json

# def get_financial_statements(symbol: str, col) -> str:
#     """Retrieve key financial statement data."""
#     try:
#         stock = yf.Ticker(symbol)
#         financials = stock.financials
#         balance_sheet = stock.balance_sheet

#         latest_year = financials.columns[col]  # most recent year

#         def safe_get(df, row_name):
#             """Safely retrieve a value from the DataFrame."""
#             try:
#                 return float(df.loc[row_name, latest_year]) if row_name in df.index else None
#             except Exception:
#                 return None

#         data = {
#             "symbol": symbol,
#             "period": str(latest_year.year) if hasattr(latest_year, "year") else str(latest_year),

#             # Income Statement
#             "TotalRevenue": safe_get(financials, "Total Revenue"),
#             "CostOfRevenue": safe_get(financials, "Cost Of Revenue"),
#             "GrossProfit": safe_get(financials, "Gross Profit"),
#             "OperatingExpense": safe_get(financials, "Operating Expense"),
#             "PretaxIncome": safe_get(financials, "Pretax Income"),
#             "TaxProvision": safe_get(financials, "Tax Provision"),
#             "NetIncomeCommonStockholders": safe_get(financials, "Net Income Common Stockholders"),
#             "DilutedEPS": safe_get(financials, "Diluted EPS"),
#             "EBIT": safe_get(financials, "EBIT"),
#             "EBITDA": safe_get(financials, "EBITDA"),

#                 # Balance Sheet (useful for leverage, liquidity)
#             "TotalAssets": safe_get(balance_sheet, "Total Assets"),
#             "TotalDebt": safe_get(balance_sheet, "Total Debt"),
#             "TotalLiabilityNetMinorityInterest": safe_get(balance_sheet, "Total Liabilities Net Minority Interest"),
#             "TotalCapit": safe_get(balance_sheet, "Total Capitalization"),
#             "CommonStock EyEquity": safe_get(balance_sheet, "Common Stock Equity"),
#             "NetTangible AyAssets": safe_get(balance_sheet, "Net Tangible Assets"),
#             "WorkingCap": safe_get(balance_sheet, "Working Capital"),
#             "InvestedCa": safe_get(balance_sheet, "Invested Capital"),
#             "OrdinaryShareyNumber": safe_get(balance_sheet, "Ordinary Shares Number"),
#             "TreasuryShareyNumber": safe_get(balance_sheet, "Treasury Shares Number"),


#         }

#         return json.dumps(data, indent=2)
#     except Exception as e:
#         return f"Error: {str(e)}"


# ticker = 'AAPL'
# print(get_financial_statements(ticker, 1))


import json
file = 'yhallsym.txt'
# with open(file, 'r') as file:
#     my_dict = json.load(file)

with open(file, 'r', encoding='utf-8') as file:
    content = file.read()

import ast

my_dict = ast.literal_eval(content)

import pandas as pd

df = pd.DataFrame(list(my_dict.items()), columns=['Ticker', 'Company'])
# print(df)

import polars as pl
df = pl.DataFrame({
    "Ticker": list(my_dict.keys()),
    "Company": list(my_dict.values())
})

filtered_df = df.filter(
    df["Company"].str.contains("Fincantieri")
)

print(filtered_df)