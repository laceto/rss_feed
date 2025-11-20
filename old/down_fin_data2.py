import yfinance as yf
import json
import polars as pl
import pandas as pd
import numpy as np

import pandas as pd

class FinancialDataProcessor:
    def __init__(self, ticker, balance_sheet, income_stmt, cashflow):
        self.ticker = ticker
        self._balance_sheet = balance_sheet
        self._income_stmt = income_stmt
        self._cashflow = cashflow

    @classmethod
    def from_ticker(cls, ticker):
        try:
            bs = cls.__reshape_fin_data(cls.get_balance_sheet(ticker))
            inc = cls.__reshape_fin_data(cls.get_income_stmt(ticker))
            cf = cls.__reshape_fin_data(cls.get_cashflow(ticker))
            return cls(ticker, bs, inc, cf)
        except Exception as e:
            print(f"Failed to create FinancialDataProcessor for {ticker}: {e}")
            return cls(ticker, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def get_merged_data(self):
        """Public method to get merged financial data."""
        try:
            df_merged = self._balance_sheet.merge(self._cashflow, how="left", on=["ticker", "time"])
            return df_merged.merge(self._income_stmt, how="left", on=["ticker", "time"])
            # return pd.concat([, self._income_stmt, self._cashflow], ignore_index=True)
        except Exception as e:
            print(f"Error merging data: {e}")
            return pd.DataFrame()

    def export_to_csv(self, path):
        """Public method to export merged data to CSV."""
        try:
            df = self.get_merged_data()
            df.to_csv(path, index=False)
            print(f"Data exported to {path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            
    def export_to_xlsx(self, path, sheet_name):
        """Public method to export merged data to CSV."""
        try:
            df = self.get_merged_data()
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Data exported to {path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")

    @staticmethod
    def __reshape_fin_data(df):
        """Private method to reshape financial data."""
        try:
            pivot_vars = ["index", "ticker", "docs"]
            value_vars = [c for c in df.columns if c not in pivot_vars]

            df = df.melt(
                id_vars=pivot_vars,
                value_vars=value_vars,
                var_name="time",
                value_name="value"
            )

            df = df.pivot_table(
                index=["ticker", "docs", "time"],
                columns="index",
                values="value"
            ).reset_index()

            return df
        except Exception as e:
            print(f"Error reshaping financial data: {e}")
            return pd.DataFrame()

    @staticmethod
    def __filter_col_by(df, col, by):
        """Private method to filter a column by prefix."""
        try:
            return df[df[col].str.startswith(tuple(by))]
        except Exception as e:
            print(f"Error filtering column '{col}': {e}")
            return pd.DataFrame()

    # Placeholder methods for data retrieval
    @staticmethod
    def get_balance_sheet(sym):
        """Fetches and formats the balance sheet for a given ticker."""
        try:
            bs = yf.Ticker(sym).balance_sheet
            if bs is None or bs.empty:
                return pd.DataFrame()
            bs = bs.reset_index()
            bs['ticker'] = sym
            bs['docs'] = 'balance_sheet'
            return bs
        except Exception as e:
            print(f"Error retrieving balance sheet for {sym}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_income_stmt(sym):
        """Fetches and formats the income statement for a given ticker."""
        try:
            inc = yf.Ticker(sym).income_stmt
            if inc is None or inc.empty:
                return pd.DataFrame()
            inc = inc.reset_index()
            inc['ticker'] = sym
            inc['docs'] = 'income_stmt'
            return inc
        except Exception as e:
            print(f"Error retrieving income statement for {sym}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_cashflow(sym):
        """Fetches and formats the cash flow statement for a given ticker."""
        try:
            cf = yf.Ticker(sym).cash_flow
            if cf is None or cf.empty:
                return pd.DataFrame()
            cf = cf.reset_index()
            cf['ticker'] = sym
            cf['docs'] = 'cash_flow'
            return cf
        except Exception as e:
            print(f"Error retrieving cash flow for {sym}: {e}")
            return pd.DataFrame()

            
            
# Instantiate the processor for a specific ticker
processor = FinancialDataProcessor.from_ticker("AAPL")

# Get merged financial data
merged_data = processor.get_merged_data()
print(merged_data.head())

# Export to CSV
processor.export_to_csv("aapl_financials.csv")
processor.export_to_xlsx("aapl_financials.xlsx", "aapl")
