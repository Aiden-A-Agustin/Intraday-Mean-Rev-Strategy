"""
Data loading and basic preprocessing for minute-bar data.

Conventions
-----------
- Long-form DataFrame with columns:
  ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
- `datetime` is parsed to timezone-aware UTC.
- Caller is responsible for slicing by date/universe as needed.
"""

from __future__ import annotations
import pandas as pd


COLS = ["symbol","datetime","open","high","low","close","volume"]


def load_minute_bars(path: str) -> pd.DataFrame:
    """
    Load minute bars from Parquet (preferred) or CSV.

    Parameters
    ----------
    path : str
        File path. If it ends with '.parquet', uses pyarrow/Parquet;
        otherwise falls back to CSV.

    Returns
    -------
    pd.DataFrame
        Long-form bars with required columns; `datetime` is tz-aware (UTC).
    """

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df[COLS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def preprocess(df: pd.DataFrame, min_price: float = 5.0) -> pd.DataFrame:
    """
    Basic cleaning: price floor and deterministic sort.

    Parameters
    ----------
    df : pd.DataFrame
        Minute bars in long form.
    min_price : float
        Minimum allowed price; rows with close < min_price are dropped.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted bars (by ['symbol','datetime']).
    """

    df = df[df["close"] >= min_price].copy()
    df.sort_values(["symbol","datetime"], inplace=True)
    return df


def median_dollar_vol(df: pd.DataFrame, lookback_days: int = 30) -> pd.Series:
    """
    Approximate **daily median dollar volume** per symbol.

    Notes
    -----
    - Dollar volume proxy = `close * volume`.
    - This implementation computes the median across **all days present**
      in `df`. If you need a strict N-day lookback, slice the input
      DataFrame to the desired window before calling.
      (e.g., df[df['datetime'] >= cutoff]).

    Parameters
    ----------
    df : pd.DataFrame
        Minute bars in long form (must include 'close', 'volume').
    lookback_days : int
        Kept for API clarity; caller should pre-slice the last N days.

    Returns
    -------
    pd.Series
        Median daily dollar volume by symbol (index = symbol, name 'dollar_vol').
    """
    
    g = df.assign(dv=df["close"] * df["volume"]).groupby(["symbol", df["datetime"].dt.date])
    daily_dv = g["dv"].sum().reset_index(name="dollar_vol")
    med = daily_dv.groupby("symbol")["dollar_vol"].median()
    return med