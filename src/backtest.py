"""
Vectorized intraday backtester for US equities.

Design
------
- Signal: cross-sectional K-bar returns on minute bars.
- Portfolio: sticky long/short membership with hysteresis (q_in, q_out).
- Execution: one of {rate-limit, no-trade band, grid}; lag to t+1 open.
- Returns: by default intraday open→close with optional EOD flatten.
- Costs: half-spread + impact in bps applied to turnover from |Δw|.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .portfolio import cross_sectional_z, neutralize, size_positions
from .costs import costs_from_turnover

def rate_limit_time(w_target: pd.DataFrame, step_bps: float, max_w: float) -> pd.DataFrame:
    """"
    Limit per-update changes in each name to at most `step_bps`.

    Parameters
    ----------
    w_target : DataFrame
        Target weights per time/name.
    step_bps : float
        Maximum absolute weight delta per update, in basis points.
    max_w : float
        Per-name absolute weight cap (fraction of NAV).

    Returns
    -------
    DataFrame
        Executed weights after rate limiting.
    """
    step = step_bps / 1e4
    cols = w_target.columns
    prev = pd.Series(0.0, index=cols)
    rows = []
    for _, row in w_target.iterrows():
        delta = (row - prev).clip(-step, step)
        prev = (prev + delta).clip(-max_w, max_w)
        rows.append(prev.values)
    return pd.DataFrame(rows, index=w_target.index, columns=cols)

def sticky_membership(score: pd.DataFrame,
                      q_in: float = 0.25, q_out: float = 0.35,
                      update_mask: pd.Series | None = None) -> pd.DataFrame:
    
    """
    Build sticky {-1,0,+1} membership with hysteresis.

    Parameters
    ----------
    score : DataFrame
        Cross-sectional signal (index=time, columns=symbol).
        Lower ranks are interpreted as "more long".
    q_in, q_out : float
        Enter and exit quantiles in [0,1]; require q_out >= q_in.
    update_mask : Series, optional
        Times when membership may update (True). If None, updates each bar.

    Returns
    -------
    DataFrame
        Membership in {-1, 0, +1}, forward-filled between allowed updates.
    """

    assert q_out >= q_in, "q_out must be >= q_in"
    ranks = score.rank(axis=1, method="first", pct=True)
    m = pd.DataFrame(0.0, index=score.index, columns=score.columns)

    if update_mask is None:
        update_mask = pd.Series(True, index=score.index)
    update_times = update_mask[update_mask].index

    prev = m.iloc[0].copy()
    for t in update_times:
        r = ranks.loc[t]

        was_long  = (prev ==  1.0)
        was_short = (prev == -1.0)

        stay_long  = was_long  & (r >= 1 - q_out)   # stay long while rank stays HIGH
        stay_short = was_short & (r <= q_out)       # stay short while rank stays LOW
        enter_long  = (~was_long)  & (r >= 1 - q_in)
        enter_short = (~was_short) & (r <= q_in)

        cur = pd.Series(0.0, index=score.columns)
        cur[stay_long | enter_long]   =  1.0
        cur[stay_short | enter_short] = -1.0

        m.loc[t] = cur.values
        prev = cur

    m = m.replace(0.0, np.nan).ffill().fillna(0.0)
    return m

def quantile_long_short(zwide: pd.DataFrame, q: float = 0.1) -> pd.DataFrame:
    """
    Construct a long-short portfolio based on quantile ranks of factor values.

    Parameters
    ----------
    zwide : pd.DataFrame
        Wide-format DataFrame (datetime × symbols) of factor values (e.g., z-scores).
    q : float, default 0.1
        Quantile threshold. Bottom q fraction is long; top q fraction is short.

    Returns
    -------
    pd.DataFrame
        Long/short signal weights:
        - +1 for long positions (bottom quantile).
        - -1 for short positions (top quantile).
        - 0 otherwise.
    """
    ranks = zwide.rank(axis=1, method="first", pct=True)
    long  = (ranks <= q).astype(float)
    short = (ranks >= 1 - q).astype(float) * -1.0
    return long + short

def to_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a long-format DataFrame into wide format by pivoting on symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ['datetime', 'symbol', col].
    col : str
        Column to pivot into wide format (e.g., 'close', 'volume').

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with:
        - Index = datetime
        - Columns = symbols
        - Values = col
    """
    return df.pivot(index="datetime", columns="symbol", values=col).sort_index()

def apply_no_trade_band(w_smooth: pd.DataFrame, band_bps: float = 10.0) -> pd.DataFrame:
    
    """
    Apply a symmetric no-trade band around the last executed weight.

    Trades only when |target - last| >= band.

    Parameters
    ----------
    w_target : DataFrame
        Target weights per time/name.
    band_bps : float
        Band half-width in basis points.

    Returns
    -------
    DataFrame
        Executed weights after band.
    """

    band = band_bps / 1e4
    cols = w_smooth.columns
    prev = pd.Series(0.0, index=cols)
    rows = []
    for _, row in w_smooth.iterrows():
        delta = row - prev
        trade = delta.abs() >= band
        prev = prev.where(~trade, row)   # only move when outside the band
        rows.append(prev.values)
    return pd.DataFrame(rows, index=w_smooth.index, columns=cols)

def vector_backtest(
    df_feat: pd.DataFrame,
    sectors: dict[str, str] | None,
    gross: float,
    max_w: float,
    cost_bps: float,
    skip_first_min: int,
    skip_last_min: int,
    smooth_halflife_bars: int = 0,
    exec_mode: str = "ratelimit",
    step_bps: float = 10.0,
    band_bps: float = 0.0,
    grid_bps: float = 0.0,
    rebalance_every_n_bars: int = 6,
    signal_mode: str = "momentum",   # "momentum" or "meanrev"
    K: int = 6,                      # reversal horizon in bars
    q_in: float = 0.30,
    q_out: float = 0.55,
    debug: bool = False,
) -> pd.DataFrame:

    """
    Run an intraday cross-sectional backtest on minute bars.

    Parameters
    ----------
    df_feat : DataFrame
        Long-form minute bars with columns at least: ['datetime','symbol','open','close'].
    sectors : dict or None
        Optional sector mapping (symbol -> sector). Neutralization is optional.
    gross : float
        Target gross leverage (e.g., 0.6 means 60% long and 60% short nominal).
    max_w : float
        Per-name absolute weight cap (fraction of NAV).
    cost_bps : float
        Round-trip cost per unit turnover, in basis points (half-spread + impact).
    skip_first_min, skip_last_min : int
        Minutes to skip after the open and before the close.
    smooth_halflife_bars : int
        Optional EWM halflife in bars. 0 disables smoothing.
    exec_mode : {'ratelimit','band','grid'}
        Execution control method.
    step_bps, band_bps, grid_bps : float
        Execution parameters (only the relevant one is used).
    rebalance_every_n_bars : int
        Update cadence in bars; weights are carried between updates.
    signal_mode : {'momentum','meanrev'}
        Orientation of the cross-sectional K-bar return.
    K : int
        Horizon in bars for the return used in the signal.
    q_in, q_out : float
        Hysteresis thresholds: enter at q_in, exit at q_out (q_out >= q_in).
    eod_flatten : bool
        If True, flatten exposure near the close (intraday-only book).
    hold_overnight : bool
        If True, compute open→open returns and rebalance at opens.
    debug : bool
        If True, prints small diagnostics; default False.

    Returns
    -------
    DataFrame
        Columns: ['raw_ret','tc','net_ret','turnover'] indexed by timestamp.
    """

    prices = to_wide(df_feat, "close")
    opens  = to_wide(df_feat, "open")

    idx_local = prices.index.tz_convert("America/New_York")
    minutes_from_open = ((idx_local - idx_local.normalize()) - pd.Timedelta(hours=9, minutes=30)).total_seconds()/60
    minutes_to_close  = (pd.Timedelta(hours=16) - (idx_local - idx_local.normalize())).total_seconds()/60
    mask = pd.Series((minutes_from_open >= skip_first_min) & (minutes_to_close >= skip_last_min),
                 index=prices.index)

    by_day  = pd.Series(idx_local.date, index=prices.index)
    bar_num = by_day.groupby(by_day).cumcount()
    update_mask = pd.Series((bar_num % rebalance_every_n_bars) == 0, index=prices.index)

    assert update_mask.index.equals(prices.index)
    assert isinstance(update_mask, pd.Series)


    logpx = np.log(prices)
    retk = np.log(prices).diff(K)

    if signal_mode.lower() == "momentum":
        score = -retk

    else:  
        score = retk

    memb = sticky_membership(score, q_in=q_in, q_out=q_out, update_mask=update_mask)
    memb = memb.loc[mask]

    w_target = memb.fillna(0.0).div(memb.abs().sum(axis=1), axis=0) * gross
    update_mask2 = update_mask.reindex(w_target.index, fill_value=False)
    w_target = w_target.where(update_mask2, np.nan).ffill().fillna(0.0)
    w_target = w_target.clip(-max_w, max_w)

    exec_mode = exec_mode.lower()

    if exec_mode == "ratelimit":
        w_pre  = apply_no_trade_band(w_target, band_bps) if band_bps and band_bps > 0 else w_target
        w_exec = rate_limit_time(w_pre, step_bps=step_bps, max_w=max_w)

    elif exec_mode == "band":
        w_exec = apply_no_trade_band(w_target, band_bps=band_bps)

    elif exec_mode == "grid":
        g = max(1e-8, grid_bps / 1e4)
        w_exec = (w_target / g).round() * g

    else:
        w_exec = w_target

    eod_flat = pd.Series(minutes_to_close < 10, index=prices.index).reindex(w_exec.index, fill_value=False)
    w_exec.loc[eod_flat.values] = 0.0

    rets_oc   = (prices / opens - 1.0).reindex(w_exec.index)
    w_lag     = w_exec.shift(1).fillna(0.0)
    pnl_gross = (w_lag * rets_oc).sum(axis=1).rename("raw_ret")

    tc       = costs_from_turnover(w_exec, cost_bps=cost_bps)
    turnover = w_exec.diff().abs().sum(axis=1).rename("turnover")

    if debug:
        diag_rets = (prices / opens - 1.0).reindex(w_exec.index)
        print(f"[DBG] using K={K}, q_in={q_in}, q_out={q_out}, mode={signal_mode}")
        print("sign check:", float((memb.shift(1).fillna(0.0) * diag_rets)
                               .sum(axis=1).mean()))
        print("exec mode:", exec_mode)
        print("executed sign check (mean raw):",
          float((w_exec.shift(1).fillna(0.0) * diag_rets)
                .sum(axis=1).mean()))
        

    pnl_net = (pnl_gross - tc).rename("net_ret")
    out = pd.concat({"raw_ret": pnl_gross, "tc": tc, "net_ret": pnl_net, "turnover": turnover}, axis=1)
    return out.dropna()
