"""
Transaction-cost utilities.

The canonical intraday cost model here is **bps per unit turnover**:
tc_t = cost_bps * sum_i |w_{t,i} - w_{t-1,i}| / 1e4
"""
from __future__ import annotations
import pandas as pd


def costs_from_turnover(weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Compute per-bar transaction costs from portfolio turnover.

    Parameters
    ----------
    weights : pd.DataFrame
        Executed weights (time × symbols). Each row should sum to the
        portfolio weights at that time (after execution controls).
    cost_bps : float
        Round-trip cost in basis points per **unit turnover**.

    Returns
    -------
    pd.Series
        Per-bar transaction cost (same index as `weights`), named "tc".

    Notes
    -----
    - Turnover per bar is ∑|Δw_i| over all symbols.
    - Costs are in **return space** (fraction of NAV), not dollars.
    """
    dw = weights.diff().abs().sum(axis=1)
    return (dw * cost_bps / 1e4).rename("tc")