"""
Simple plotting/series helpers for reporting backtest results.
"""
from __future__ import annotations
import pandas as pd


def equity_curve(df_bt: pd.DataFrame) -> pd.Series:
    """
    Build cumulative equity from per-bar net returns.

    Parameters
    ----------
    df_bt : pd.DataFrame
        Backtest output containing a 'net_ret' column (per-bar net return).

    Returns
    -------
    pd.Series
        Equity curve (cumulative product of 1 + net_ret), named "Equity".
    """
    return (1 + df_bt["net_ret"]).cumprod().rename("Equity")