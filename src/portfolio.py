"""
Portfolio helpers: cross-sectional standardization, (optional) sector
neutralization, and position sizing with gross and per-name caps.

All functions operate on *wide* frames (index=time, columns=symbol).
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def cross_sectional_z(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score at each timestamp (row-wise standardization).

    Parameters
    ----------
    wide : DataFrame
        Matrix of values (time × symbols).

    Returns
    -------
    DataFrame
        Z-scored values, per row: (x - mean_row) / std_row.
        Rows with zero std are filled with 0.0.
    """

    mean = wide.mean(axis=1)
    std = wide.std(axis=1).replace(0, np.nan)
    z = (wide.sub(mean, axis=0)).div(std, axis=0).fillna(0.0)
    return z


def neutralize(wide: pd.DataFrame, sectors: pd.Series | None = None) -> pd.DataFrame:
    """
    De-mean cross-sectionally, optionally within sectors first.

    If `sectors` is provided (index must match columns of `wide`), we
    de-mean *within each sector*, concatenate, then de-mean the whole
    cross-section again to remove any residual market mean. If `sectors`
    is None, we simply de-mean cross-sectionally.

    Parameters
    ----------
    wide : DataFrame
        Matrix (time × symbols) to be de-meaned.
    sectors : Series or None
        Mapping from symbol -> sector; index must be symbols in `wide`.

    Returns
    -------
    DataFrame
        De-meaned matrix with the same shape/columns as `wide`.
    """
    
    if sectors is None:
        return wide.sub(wide.mean(axis=1), axis=0)
    out = []
    for sec, cols in sectors.groupby(sectors).groups.items():
        sub = wide[cols]
        out.append(sub.sub(sub.mean(axis=1), axis=0))
    out = pd.concat(out, axis=1).reindex(columns=wide.columns)
    # finally de‑mean market
    return out.sub(out.mean(axis=1), axis=0)


def size_positions(raw: pd.DataFrame, gross: float = 2.0, max_w: float = 0.004) -> pd.DataFrame:
    """
    Convert raw signals/memberships into sized positions with caps.

    Steps
    -----
    1) Scale rows so that the *sum of absolute weights* ≈ `gross`.
    2) Apply a per-name cap: weights clipped to ±`max_w`.
    3) Rescale once more to re-hit the target gross (approx).

    Parameters
    ----------
    raw : DataFrame
        Raw targets (time × symbols), e.g., {-1,0,+1} memberships or scores.
    gross : float
        Target gross leverage per timestamp (e.g., 2.0 for 200% gross).
    max_w : float
        Per-name absolute cap as a fraction of NAV (e.g., 0.004 = 40 bps).

    Returns
    -------
    DataFrame
        Sized and capped positions with the same shape as `raw`.
    """
    scaled = raw.div(raw.abs().sum(axis=1), axis=0) * gross
    capped = scaled.clip(-max_w, max_w).fillna(0.0)
    # re‑scale to hit gross approximately
    gross_now = capped.abs().sum(axis=1)
    adj = (gross / gross_now).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return capped.mul(adj, axis=0).fillna(0.0)