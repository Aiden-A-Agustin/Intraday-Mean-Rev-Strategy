"""
Summary metrics for intraday backtests.

Reports raw/net Sharpe (annualized), max drawdown, turnover/day,
average daily cost (bps), and daily raw/net averages for quick diagnostics.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

TRADING_DAYS = 252

def _infer_bars_per_day(index: pd.DatetimeIndex) -> int:

    """Infer typical intraday bars/day from the timestamp index (NY time)."""

    idx = index.tz_convert("America/New_York")
    by_day = pd.Series(1, index=idx).groupby(idx.normalize()).sum()
    return int(by_day.median())

def annualize_sr(ret: pd.Series, bars_per_day: int | None = None) -> float:

    """
    Annualize Sharpe from bar-level returns.

    Uses TRADING_DAYS and inferred bars/day if not provided.
    """

    if bars_per_day is None:
        bars_per_day = _infer_bars_per_day(ret.index)
    mu = ret.mean()
    sd = ret.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    daily = mu * bars_per_day
    daily_sd = sd * np.sqrt(bars_per_day)
    return float(np.sqrt(TRADING_DAYS) * daily / daily_sd)

def max_dd(ret: pd.Series) -> float:
    """Compute maximum drawdown on the cumulative equity curve."""
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def summary(df_bt: pd.DataFrame) -> dict:
    """
    Summarize a backtest timeseries.

    Expected columns: 'raw_ret', 'net_ret', 'turnover', 'tc'.

    Returns
    -------
    dict
        Keys include Sharpe (raw/net), CAGR (net), Max DD (net),
        Turnover/day, Avg cost/day (bps), and daily raw/net averages.
    """
    bpd   = _infer_bars_per_day(df_bt.index)
    r_net = df_bt["net_ret"].astype(float)
    r_raw = df_bt["raw_ret"].astype(float) if "raw_ret" in df_bt else r_net

    years = len(r_net) / (TRADING_DAYS * bpd) if len(r_net) else 0.0
    cagr  = float((1 + r_net).prod() ** (1 / years) - 1) if years > 0 else 0.0

    turn        = df_bt["turnover"] if "turnover" in df_bt.columns else pd.Series(0.0, index=df_bt.index)
    tc          = df_bt["tc"]       if "tc"       in df_bt.columns else pd.Series(0.0, index=df_bt.index)
    avg_cost_day = float(tc.groupby(df_bt.index.normalize()).sum().mean())
    avg_cost_bps = avg_cost_day * 1e4
    daily_raw = df_bt["raw_ret"].groupby(df_bt.index.normalize()).sum()
    daily_net = df_bt["net_ret"].groupby(df_bt.index.normalize()).sum()

    return {
        "Ann. Sharpe (raw)": annualize_sr(r_raw, bpd),
        "Ann. Sharpe (net)": annualize_sr(r_net, bpd),
        "CAGR (approx, net)": cagr,
        "Max DD (net)": max_dd(r_net),
        "Turnover/day": float(turn.mean() * bpd),
        "Avg cost/day (bps)": avg_cost_bps,
        "Avg raw/day (bps)":  float(daily_raw.mean() * 1e4),
        "Avg net/day (bps)":  float(daily_net.mean() * 1e4),
        "Daily sd (bps)":     float(daily_net.std(ddof=1) * 1e4),
    }
