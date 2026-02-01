from __future__ import annotations
# This allows using type jints that refer to classes not yet fully defined
# It also helps with forward references and keeps typing cleaner in Python

import logging
# Build-in Python module used to create logs such as info warnings and errors
# Logging is important for secure systems, more traceability, and debugging

from typing import Iterable
# Iterable means "something you can loop over"
# we use it for the required columns parameter

import pandas as pd
# Used for DataFrame manipulations, series, and data cleaning

def get_logger(name: str = "retail_analytics") -> logging.Logger:
    """Get a module logger
    
    Why this exists:
    - We want consistent logging in every src file
    - Different modules can have different logger names
    """
    return logging.getLogger(name)

def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """
    Raise ValueError if required columns are missing.
    
    Why this exists:
    - Our algorithms depend on specific columns
    - If a column is missing, we stop early with a clear error message
    - this prevents wrong results and avoids confusing crashes
    """
    missing = [c for c in cols if c not in df.columns]
    # we build a list of all columns that are required but no found in the dataframe
    if missing:
        # if any required columns are missing, we raise an error
        raise ValueError(f"Missing required columns: {missing}")
    
def to_datetime_safe(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert a column to datetime safely (errors -> NaT).
    
    Why this exists:
    - dates in csv files may come as text strings
    - if some rows have invalid date, we don't want to crash
    - errors="coerce" turns invalid dates into Not a Time (NaT), so we can drop or handle them safely
    """
    return pd.to_datetime(df[col], errors="coerce")

def to_numeric_safe(s: pd.Series) -> pd.Series:
    """
    Convert a series to numeric safely (errors -> NaN).
    
    Why this exists:
    - revenue, unit_price, quantity should be numeric
    - csv data may contain text, symbol, or bad values.
    - errors="coerce" turns invalid numbers into NaN, so we can drop or handle them safely
    """
    return pd.to_numeric(s, errors="coerce")