"""
Feature construction helpers for minute-bar data.

All functions are written to avoid look-ahead: at time t, they only use data
up to and including t-1 unless explicitly stated otherwise.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from .utils import safe_div


def incremental_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP up to t-1 (exclude the current bar), per symbol.

    Parameters
    ----------
    df
        Long-form bars with at least ['datetime', 'symbol', 'close', 'volume'].
        Assumed sorted by ['symbol', 'datetime'].

    Returns
    -------
    pd.Series
        VWAP aligned to df's index (name='vwap'). For the first bar of the day,
        sets VWAP to the current close (early bars are typically skipped anyway).
    """

    px  = df["close"]  
    vol = df["volume"].astype(float)

    g_keys = [df["symbol"], df["datetime"].dt.date]
    num_cum = (px * vol).groupby(g_keys).cumsum().shift(1)
    den_cum = vol.groupby(g_keys).cumsum().shift(1)

    vwap = (num_cum / den_cum)
    vwap = vwap.fillna(px)
    return vwap.rename("vwap")



def rolling_vol(df: pd.DataFrame, lookback_bars: int = 6) -> pd.Series:
    """
    Per-symbol rolling volatility (std) of intraday log returns.

    Parameters
    ----------
    df
        Long-form bars with ['datetime', 'symbol', 'close'] sorted by
        ['symbol', 'datetime'].
    lookback_bars
        Window length in bars (default 6 ≈ 30 minutes on 5-min bars).

    Returns
    -------
    pd.Series
        Rolling standard deviation aligned to df's index (name='sigma').
    """

    ret = np.log(df["close"]).groupby(df["symbol"]).diff()

    vol = ret.groupby(df["symbol"]).apply(
        lambda s: s.rolling(lookback_bars, min_periods=max(2, lookback_bars // 2)).std()
    )
    vol.index = vol.index.droplevel(0)
    return vol.rename("sigma")



def zscore_vwap_dev(df: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    """
    Z-score of close minus incremental VWAP, scaled by recent intraday vol.

    This is a simple, stationary-ish deviation signal often used for
    mean-reversion: (close - vwap_{t-1}) / rolling_std(log_ret, L).

    Parameters
    ----------
    df
        Long-form bars with ['datetime','symbol','close','volume'] sorted by
        ['symbol','datetime'].
    lookback_bars
        Window length in bars for the rolling volatility estimator.

    Returns
    -------
    pd.DataFrame
        Columns:
          - 'vwap'  : incremental VWAP up to t-1
          - 'sigma' : rolling std of log-returns
          - 'z'     : (close - vwap) / sigma, finite + 0-filled
    """

    out = df.copy()
    out.sort_values(["symbol", "datetime"], inplace=True)

    vw = incremental_vwap(out)
    sg = rolling_vol(out, lookback_bars)

    out["vwap"]  = vw.to_numpy()
    out["sigma"] = sg.to_numpy()

    z_arr = safe_div(out["close"] - out["vwap"], out["sigma"])
    out["z"] = pd.Series(z_arr, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out

