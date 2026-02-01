# src/anomaly_detection.py
from __future__ import annotations

import logging
import pandas as pd
from scipy import stats
# stats.zscore is used to compute how far a value is from the mean (in standard deviations).


from .utils import require_columns, to_numeric_safe, get_logger

# Logger name shows where the logs come from
logger = get_logger("anomaly_detection")


def find_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return duplicate rows (keep all duplicates).

    Why we need this (fraud context):
    - Duplicate rows can mean the same order was processed twice.
    - This can be fraud (replay attack) or a system bug.
    - Duplicates can also inflate revenue and hide real fraud patterns.
    """
    try:
        # Validate input type
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        # duplicated(keep=False) marks ALL occurrences of duplicates as True
        # Example: if a row appears 3 times, all 3 rows are flagged.
        dup_mask = data.duplicated(keep=False)
        
        # Select only the duplicate rows and copy (so we don't modify original)
        dups = data.loc[dup_mask].copy()

        # Log how many duplicates were found
        logger.info("Duplicate rows found: %d", int(dup_mask.sum()))
        return dups

    except Exception as e:
        logger.exception("find_duplicates failed: %s", e)
        raise


def zscore_anomalies(data: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Flag rows where revenue |z-score| exceeds z_thresh.

    Adds columns:
    - revenue_numeric: revenue converted to a safe numeric column
    - z_score: absolute z-score value

    Returns:
    - Only the anomalies, sorted by z_score (highest first)

    Why we need this (fraud context):
    - Fraud often shows up as unusually large transactions (extreme revenue).
    - Z-score measures how unusual a value is compared to the dataset.
    """
    try:
        # 1) Ensure the required column exists
        require_columns(data, ["revenue"])
        
        # 2) Validate threshold (must be positive)
        if not isinstance(z_thresh, (int, float)) or z_thresh <= 0:
            raise ValueError("z_thresh must be a positive number.")

        # 3) Work on a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # 4) Convert revenue into a numeric column safely
        # Bad values become NaN
        df["revenue_numeric"] = to_numeric_safe(df["revenue"])

        # 5) Drop rows that have invalid revenue values
        df = df.dropna(subset=["revenue_numeric"])
        
        # If there are fewer than 2 rows, z-score cannot be computed properly
        if len(df) < 2:
            logger.warning("Not enough rows to compute z-scores (need >= 2).")
            return df.head(0).copy()

        # 6) Compute z-scores
        # z-score = (value - mean) / standard deviation
        # nan_policy="omit" ignores NaN values (we already dropped them anyway)
        z = stats.zscore(df["revenue_numeric"].astype(float), nan_policy="omit")
        
        # 7) Store absolute z-score (distance from normal)
        df["z_score"] = pd.Series(z, index=df.index).abs()

        # 8) Filter anomalies (values beyond the threshold)
        anomalies = df.loc[df["z_score"] > float(z_thresh)].copy()
        
        # 9) Sort anomalies so the most extreme ones are at the top
        anomalies = anomalies.sort_values("z_score", ascending=False).reset_index(drop=True)

        # 10) Log count of anomalies found
        logger.info("Anomalies found (z_thresh=%.2f): %d", float(z_thresh), len(anomalies))
        return anomalies

    except Exception as e:
        logger.exception("zscore_anomalies failed: %s", e)
        raise
