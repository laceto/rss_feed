import yfinance as yf
import json
import polars as pl
import pandas as pd
import numpy as np


def save_xlsx(df: pl.DataFrame, path: str, sheet_name: str = "Sheet1") -> None:
    """
    Save a Polars DataFrame to an Excel (.xlsx) file.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to save.
    path : str
        The file path for the Excel file.
    sheet_name : str, optional
        Name of the Excel sheet (default = 'Sheet1').
    """

    # Save with pandas ExcelWriter (uses openpyxl by default)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

inc_s_item2filter = [
    "Total Revenue",
    "Cost of Revenue",
    "Gross Profit",
    "Operating Income",
    "Pretax Income",
    "Tax Provision",
    "Net Income Common Stockholders",
    "Diluted EPS",
    "EBIT",
    "EBITDA",
]

cf_item2filter = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "End Cash Position",
    "Capital Expenditure",
    "Issuance Of Debt",
    "Repayment Of Debt",
    "Repurchase Of Capital Stock",
    "Free Cash Flow",
]

bs_item2filter = [
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Total Capitalization",
    "Common Stock Equity",
    "Net Tangible Assets",
    "Working Capital",
    "Invested Capital",
    "Total Debt",
    "Ordinary Shares Number",
    "Treasury Shares Number",
]

def filter_col_by(df, col, by):
    return df[df[col].str.startswith(tuple(by))]

def get_balance_sheet(sym):
    bs = yf.Ticker(sym).balance_sheet
    if bs is None:
        return pd.DataFrame()  # return empty if no data
    bs = bs.reset_index()
    bs['ticker'] = sym
    bs['docs'] = 'balance_sheet'

    return bs

def get_income_stmt(sym):
    bs = yf.Ticker(sym).income_stmt
    if bs is None:
        return pd.DataFrame()  # return empty if no data
    bs = bs.reset_index()
    bs['ticker'] = sym
    bs['docs'] = 'income_stmt'

    return bs

def get_cashflow(sym):
    bs = yf.Ticker(sym).cash_flow
    if bs is None:
        return pd.DataFrame()  # return empty if no data
    bs = bs.reset_index()
    bs['ticker'] = sym
    bs['docs'] = 'cash_flow'

    return bs

def reshape_fin_data(df):

    pivot_vars = ["index", "ticker", "docs"]
    # Automatically select all columns except 'index' and 'ticker'
    value_vars = [c for c in df.columns if c not in pivot_vars]

    df = df.melt(
        id_vars=pivot_vars,
        value_vars=value_vars,
        var_name="time",
        value_name="value"
    )

    df = df.pivot_table(index=["ticker", "docs", "time"], columns="index", values="value").reset_index()

    return df

def merge_all_tables(bs, cf, inc_s):

    # First left join df1 and df2
    df_merged = bs.merge(cf, how="left", on=["ticker", "time"])
    # Now left join with df3
    df_final = df_merged.merge(inc_s, how="left", on=["ticker", "time"])

    return df_final

ticker = 'AVIO.MI'

# cf = filter_col_by(df=get_cashflow(ticker), col='index', by=cf_item2filter)
# inc_s = filter_col_by(df=get_income_stmt(ticker), col='index', by=inc_s_item2filter)
# bs = filter_col_by(df=get_balance_sheet(ticker), col='index', by=bs_item2filter)
# cf = reshape_fin_data(cf)
# inc_s = reshape_fin_data(inc_s)
# bs = reshape_fin_data(bs)
# df = merge_all_tables(bs, cf, inc_s)
# df.columns = df.columns.str.lower().str.replace(" ", "_")
# save_xlsx(df, "df.xlsx", ticker)
df = pd.read_excel('df.xlsx', sheet_name='AVIO.MI')
df.columns = df.columns.str.lower().str.replace(" ", "_")

# print(df.columns)



# Big list of keywords
keywords = ["debt", "working", "liab"]

# Create a regex pattern
pattern = "|".join(keywords)

# Select columns containing any of the keywords
df_selected = df.loc[:, df.columns.str.contains(pattern, case=False, regex=True)]

# print(df_selected)



# import numpy as np

class FundamentalTraderAssistant:
    def __init__(self, data: pd.DataFrame, weights: dict):
        """
        """
        self.d = data
        self.metrics = {}
        self.scores = {}
        self.red_flags = []
        self.w = weights

    def safe_div(self, num, den):
        return num / den if den not in (0, None) and num is not None else None
    
    def safe_div(self, num, den):
        return np.where((den != 0) & (den.notna()) & (num.notna()), num / den, None)

    def compute_metrics(self):
        d = self.d
        # m = {}

        # Profitability
        d["GrossMargin"] = self.safe_div(d["gross_profit"], d["total_revenue"])
        d["OperatingMargin"] = self.safe_div(d["operating_income"], d["total_revenue"])
        d["NetProfitMargin"] = self.safe_div(d["net_income_common_stockholders"], d["total_revenue"])
        d["EBITDAMargin"] = self.safe_div(d["ebitda"], d["total_revenue"])

        # Returns
        d["ROA"] = self.safe_div(d["net_income_common_stockholders"], d["total_assets"])
        d["ROE"] = self.safe_div(d["net_income_common_stockholders"], d["common_stock_equity"])

#         # Cash flow
        d["FCFToRevenue"] = self.safe_div(d["free_cash_flow"], d["total_revenue"])
        d["FCFYield"] = self.safe_div(d["free_cash_flow"], d["total_capitalization"])

#         # Leverage & Liquidity
        d["DebtToEquity"] = self.safe_div(d["total_debt"], d["common_stock_equity"])
        d["DebtToAssets"] = self.safe_div(d["total_debt"], d["total_assets"])
        d["CurrentRatio"] = self.safe_div(d["working_capital"], d["total_liabilities_net_minority_interest"])

        # Growth
        # d["RevenueGrowthYoY"] = self.safe_div(
        #     d["Revenue") - d["Revenue_t1"], d["Revenue_t1")
        # )
        # d["EPSGrowthYoY"] = self.safe_div(
        #     d["EPS") - d["EPS_t1"], d["EPS_t1")
        # )

        d = d.loc[:, ["ticker", "time"] + list(d.loc[:, "GrossMargin":"CurrentRatio"].columns)]

        # d = d.melt(
        #     id_vars=["ticker", "time"],
        #     value_vars=d.loc[:, "GrossMargin":"CurrentRatio"].columns,
        #     var_name="metrics",
        #     value_name="value"
        # )

        self.metrics = d
        return d

    def score_metric(self, df):
        """Apply trader-friendly scoring rules to a DataFrame with 'name' and 'value' columns."""
        thresholds = {
            "GrossMargin": [0.2, 0.3, 0.4, 0.5],
            "OperatingMargin": [0.05, 0.1, 0.15, 0.2],
            "NetProfitMargin": [0.03, 0.07, 0.12, 0.2],
            "EBITDAMargin": [0.1, 0.2, 0.3, 0.4],
            "ROA": [0.02, 0.05, 0.08, 0.12],
            "ROE": [0.05, 0.1, 0.15, 0.2],
            "FCFToRevenue": [0.02, 0.05, 0.1, 0.2],
            "FCFYield": [0.02, 0.04, 0.06, 0.1],
            "DebtToEquity": [0.5, 1.0, 1.5, 2.0],  # inverse scoring
            "CurrentRatio": [1.0, 1.2, 1.5, 2.0],
            # "RevenueGrowthYoY": [0, 0.05, 0.1, 0.2],
            # "EPSGrowthYoY": [0, 0.05, 0.1, 0.2],
        }

        def score_row(row):
            name, value = row['metrics'], row['value']
            if pd.isna(value):
                return 3
            if name in thresholds:
                score = np.digitize(value, thresholds[name]) + 1
                if name == "DebtToEquity":
                    return 6 - score  # inverse scoring
                return score
            return 3

        df['score'] = df.apply(score_row, axis=1)
        return df
    
        def metrics_red_flags(self, df):
            """Add red flag names to a long-format DataFrame with 'metrics' and 'value' columns."""

            # Step 1: Apply single-metric red flags
            def single_metric_flag(row):
                metric, value = row["metrics"], row["value"]
                if pd.isna(value):
                    return None

                if metric == "GrossMargin" and value < 0:
                    return "Negative Gross Margin"
                if metric == "OperatingMargin" and value < 0:
                    return "Negative Operating Margin"
                if metric == "NetMargin" and value < 0:
                    return "Negative Net Margin"
                if metric == "ROA" and value < 0:
                    return "Negative ROA"
                if metric == "ROE" and value < 0:
                    return "Negative ROE"
                if metric == "DebtToEquity" and value > 2:
                    return "High Debt-to-Equity (>2)"
                if metric == "DebtToAssets" and value > 0.8:
                    return "High Debt-to-Assets (>0.8)"
                if metric == "FreeCashFlow" and value < 0:
                    return "Negative Free Cash Flow"
                if metric == "FCFtoDebt" and value < 0.05:
                    return "Insufficient Free Cash Flow to cover debt"
                if metric == "OperatingCashFlow" and value < 0:
                    return "Negative Operating Cash Flow"
                return None

            df["red_flag"] = df.apply(single_metric_flag, axis=1)
            df = df[df["red_flag"].notna()].reset_index(drop=True)

            return df

    def evaluate(self):
        m = self.compute_metrics()

        m = m.melt(
            id_vars=["ticker", "time"],
            value_vars=m.loc[:, "GrossMargin":"CurrentRatio"].columns,
            var_name="metrics",
            value_name="value"
        )

        # m = self.red_flags(m)

        s = self.score_metric(m)
        

        weights = pd.DataFrame(list(self.w.items()), columns=["metrics", "Weight"])

        s = s.merge(weights, how="left", on=["metrics"])

        def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
            """
            Compute composite score per ticker per year (weighted average of metric scores).
            """
            # Weighted score = score * weight
            df["weighted_score"] = df["score"] * df["Weight"]

            # Group by ticker + year (time), aggregate safely
            composite = (
                df.groupby(["ticker", "time"], as_index=False)
                .agg(
                    total_weighted_score=("weighted_score", "sum"),
                    total_weight=("Weight", "sum")
                )
            )

            # Compute composite score
            composite["composite_score"] = composite["total_weighted_score"] / composite["total_weight"]

            # Drop helper columns if not needed
            composite = composite[["ticker", "time", "composite_score"]]

            return composite
        
        s= compute_composite_scores(s)
        self.scores = s

        self.red_flags = self.red_flags(m)

        # return s
        return {
            "metrics": m,
            "composite_scores": s,
            # "overall_score": round(overall, 1),
            # "regime": regime,
            "red_flags": self.red_flags,
            # "commentary": self.generate_comment(overall, regime),
        }

    # def generate_comment(self, score, regime):
    #     if regime == "Fundamentally Strong":
    #         return "Company shows solid profitability, efficient returns, and healthy balance sheet — supportive for bullish outlook."
    #     elif regime == "Fundamentally Moderate":
    #         return "Company has mixed fundamentals with areas of strength, but also risks that traders should monitor closely."
    #     else:
    #         return "Company fundamentals are weak — profitability, growth, or balance sheet risks suggest caution or bearish stance."


# # Example usage:
# data = {
#     "Revenue": 1000,
#     "GrossProfit": 420,
#     "OperatingIncome": 170,
#     "NetIncome": 120,
#     "EBITDA": 220,
#     "TotalAssets": 2400,
#     "TotalEquity": 860,
#     "FreeCashFlow": 110,
#     "MarketCap": 1800,
#     "TotalDebt": 700,
#     "CurrentAssets": 900,
#     "CurrentLiabilities": 650,
#     "Revenue_t1": 920,
#     "EPS": 2.5,
#     "EPS_t1": 2.1
# }

# weights = {
#     "GrossMargin": 1,
#     "OperatingMargin": 1.5,
#     "NetProfitMargin": 1.5,
#     "ROA": 1,
#     "ROE": 1,
#     "FCFtoDebt": 2,
#     "FCFMargin": 2,
#     "OCFMargin": 1.5,
#     "DebtToEquity": 1,
#     "DebtToAssets": 1,
# }

weights = {
    "GrossMargin": 8, 
    "OperatingMargin": 12, 
    "NetProfitMargin": 8, 
    "EBITDAMargin": 10,
    "ROA": 10, 
    "ROE": 12,
    "DebtToEquity": 12, 
    "DebtToAssets": 10,   
    "CurrentRatio": 8,
    "FCFToRevenue": 10, 
    "FCFYield": 10
}

assistant = FundamentalTraderAssistant(df, weights)
# metrics = assistant.compute_metrics()
# print(metrics)

# print(assistant.red_flags(metrics))

# a = assistant.evaluate()

# print(a.get("composite_scores"))

metrics = assistant.evaluate().get("metrics")
# print(metrics)

def metrics_red_flags(df, raw):
        """Add red flag names to a long-format DataFrame with 'metrics' and 'value' columns."""

        # Step 1: Apply single-metric red flags
        def single_metric_flag(row):
            metric, value = row["metrics"], row["value"]
            if pd.isna(value):
                return None

            if metric == "GrossMargin" and value < 0:
                return "Negative Gross Margin"
            if metric == "OperatingMargin" and value < 0:
                return "Negative Operating Margin"
            if metric == "NetMargin" and value < 0:
                return "Negative Net Margin"
            if metric == "ROA" and value < 0:
                return "Negative ROA"
            if metric == "ROE" and value < 0:
                return "Negative ROE"
            if metric == "DebtToEquity" and value > 2:
                return "High Debt-to-Equity (>2)"
            if metric == "DebtToAssets" and value > 0.8:
                return "High Debt-to-Assets (>0.8)"
            if metric == "FreeCashFlow" and value < 0:
                return "Negative Free Cash Flow"
            if metric == "FCFtoDebt" and value < 0.05:
                return "Insufficient Free Cash Flow to cover debt"
            if metric == "OperatingCashFlow" and value < 0:
                return "Negative Operating Cash Flow"
            return None

        df["red_flag"] = df.apply(single_metric_flag, axis=1)
        # df = df[df["red_flag"].notna()]
        df = df[df["red_flag"].notna()].reset_index(drop=True)

        return df




print(metrics_red_flags(metrics))
print(raw_red_flags(df))