# src/revenue_analysis.py
from __future__ import annotations

import logging
import pandas as pd

# we import helper utilities that make the code safer and cleaner
from .utils import require_columns, to_datetime_safe, to_numeric_safe, get_logger

# create a module-specific logger name so logs show where they came from
logger = get_logger("revenue_analysis")


def top_k_revenue_products(data: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Return the top-k products by total revenue.

    Output columns: product_id, revenue
    
    Why we need this:
    - high revenue products can be targeted by fraud
    - helps identify which products create the most money and deserve monitoring
    """
    try:
        # 1) Security check
        # make sure required columns exist
        require_columns(data, ["product_id", "revenue"])
        
        # 2) validate the parameter k
        # prevents weird inputs
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        # 3) work on a copy so we don't modify the original dataframe
        df = data.copy()
        
        # 4) convert revenue to numeric safely
        # bad values become NaN instead of crashing
        df["revenue"] = to_numeric_safe(df["revenue"])


        # 5) Drop rows where product_id or revenue is missing
        df = df.dropna(subset=["product_id", "revenue"])

        # 6) group by product and sum revenue, then sort and return top-k
        # groupby + sum is efficient for large datasets
        result = (
            df.groupby("product_id", as_index=False)["revenue"]
              .sum()
              # sort highest revenue first
              .sort_values("revenue", ascending=False, kind="mergesort")
              # keep only top k rows
              .head(k)
              .reset_index(drop=True)
        )

        # 7) log sucess for traceability
        logger.info("Computed top-%d revenue products.", k)
        return result

    except Exception as e:
        # logs the full stack trace for debugging
        logger.exception("top_k_revenue_products failed: %s", e)
        raise


def rolling_avg_revenue(data: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Compute daily revenue + rolling average over 'window_days'.

    Output columns: date, revenue, rolling_avg_revenue

    Why we need this (fraud context):
    - Raw daily revenue is noisy.
    - Rolling averages smooth trends and help spot unusual spikes over time.
    """
    try:
        # 1) Ensure required columns exist
        require_columns(data, ["order_date", "revenue"])
        
        # 2) Validate window_days parameter
        if not isinstance(window_days, int) or window_days <= 0:
            raise ValueError("window_days must be a positive integer.")

        # 3) Work on a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # 4) Convert order_date safely to datetime
        # Invalid dates become NaT
        df["order_date"] = to_datetime_safe(df, "order_date")
        
        # 5) Convert revenue safely to numeric
        df["revenue"] = to_numeric_safe(df["revenue"])

        # 6) Remove rows with invalid dates or revenue values
        df = df.dropna(subset=["order_date", "revenue"])

        # 7) Aggregate revenue per day
        # We create a "date" column (just the day, no time)
        daily = (
            df.assign(date=df["order_date"].dt.date)
              .groupby("date", as_index=False)["revenue"]
              .sum()
        )

        # 8) Sort by date so rolling window is correct
        daily = daily.sort_values("date").reset_index(drop=True)
        
        # 9) Rolling mean (moving average)
        # min_periods=1 means first days still get a value instead of NaN
        daily["rolling_avg_revenue"] = daily["revenue"].rolling(window=window_days, min_periods=1).mean()

        # 10) Log success
        logger.info("Computed rolling average revenue with window_days=%d.", window_days)
        return daily

    except Exception as e:
        logger.exception("rolling_avg_revenue failed: %s", e)
        raise
