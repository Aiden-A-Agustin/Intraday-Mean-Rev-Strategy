"""
Utility functions for intraday backtesting.

Includes:
- Session filtering (restrict bars to market open/close).
- Winsorization (clip extremes at quantiles).
- Safe division (avoid divide-by-zero errors).
"""

from __future__ import annotations
import pandas as pd
import numpy as np


TZ = "America/New_York"


def sessionize(df: pd.DataFrame, open_t: str = "09:30", close_t: str = "16:00") -> pd.DataFrame:
    """
    Restrict intraday bars to regular session hours.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a 'datetime' column.
    open_t : str, default "09:30"
        Session open time (HH:MM).
    close_t : str, default "16:00"
        Session close time (HH:MM).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only bars within the session window.
    """

    idx = pd.to_datetime(df["datetime"]).dt.tz_convert(TZ)
    df = df.copy()
    df["dt"] = idx
    df["date"] = df["dt"].dt.date
    df = df[(df["dt"].dt.time >= pd.to_datetime(open_t).time()) & (df["dt"].dt.time <= pd.to_datetime(close_t).time())]
    return df


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """
    Clip extreme values at lower/upper quantiles.

    Parameters
    ----------
    s : pd.Series
        Input series.
    p : float, default 0.01
        Quantile threshold (clip at p and 1-p).

    Returns
    -------
    pd.Series
        Winsorized series.
    """

    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def safe_div(a, b):
    """
    Elementwise division with safe handling of zero denominators.

    Parameters
    ----------
    a : array-like
        Numerator.
    b : array-like
        Denominator.

    Returns
    -------
    np.ndarray
        Result of a / b, with zeros where b == 0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(b != 0, a / b, 0.0)
    return out