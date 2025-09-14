"""
Rolling walk-forward evaluation for the intraday strategy.

What it does
------------
- Splits the timeline into rolling train/test windows with an embargo.
- Optionally performs a tiny grid search on TRAIN to pick simple knobs.
- Evaluates on TEST (true out-of-sample) and prints a compact table.

Notes
-----
- Assumes input bars (Parquet) are long-form with at least
  ['datetime','symbol','open','close','volume'].
- All times are handled in US/Eastern for session slicing.
"""

from __future__ import annotations
import argparse, yaml
import pandas as pd
from src.backtest import vector_backtest
from src.metrics import summary

NY = "America/New_York"



def _as_dtindex(x) -> pd.DatetimeIndex:
    """Accept Series/array/Index; return tz-aware DatetimeIndex (UTC)"""
    return pd.DatetimeIndex(pd.to_datetime(x, utc=True))

def day_index(x) -> pd.DatetimeIndex:
    """Return trading days (NY tz, midnight-normalized)."""
    idx = _as_dtindex(x).tz_convert(NY)
    return pd.DatetimeIndex(idx.normalize().unique()).sort_values()

def windows(x, train_days=20, test_days=5, step_days=5, embargo_days=1):
    """
    Rolling day-based windows with an embargo between train and test.

    Yields
    ------
    (tr0, tr1, te0, te1) : tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        Inclusive train start/end, and test start/end (day precision).
    """

    days = day_index(x)
    i = 0
    while i + train_days + embargo_days < len(days):
        tr0, tr1 = days[i], days[i + train_days - 1]
        te0      = days[i + train_days + embargo_days]
        te1      = days[min(i + train_days + embargo_days + test_days - 1, len(days)-1)]
        yield tr0, tr1, te0, te1
        i += step_days

def run_slice(df: pd.DataFrame, start_day: pd.Timestamp, end_day: pd.Timestamp, run_kwargs):
    """
    Run one backtest slice over [start_day, end_day] (inclusive, NY days).

    Returns
    -------
    dict | None
        Summary metrics (see src.metrics.summary) or None if slice empty.
    """

    idxL = _as_dtindex(df["datetime"]).tz_convert(NY)
    mask = (idxL >= start_day) & (idxL < end_day + pd.Timedelta(days=1))
    sl = df.loc[mask]
    if sl.empty:
        return None
    bt = vector_backtest(sl, **run_kwargs)
    return summary(bt)
def grid_search_on_train(df, tr0, tr1, base_kwargs):
    """
    Brute-force a tiny grid on TRAIN to pick a reasonable configuration.

    Objective
    ---------
    Maximize annualized net Sharpe on TRAIN.

    Returns
    -------
    dict | None
        Chosen kwargs to overlay on base_kwargs, or None if training slice empty.
    """
    
    grid = [
        dict(exec_mode="ratelimit", step_bps=s, rebalance_every_n_bars=r, q_in=qi, q_out=qo, K=K)
        for s in (7, 8, 10)
        for r in (6, 8)
        for (qi, qo) in ((0.30, 0.55), (0.35, 0.60))
        for K in (6, 9)
    ]
    best = None
    for g in grid:
        kw = base_kwargs | g
        s = run_slice(df, tr0, tr1, kw)
        if not s: 
            continue
        score = s["Ann. Sharpe (net)"]    # objective on TRAIN
        if (best is None) or (score > best[0]):
            best = (score, g)
    return (best[1] if best else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/minute_bars.parquet")
    parser.add_argument("--train-days", dest="train_days", type=int, default=20)
    parser.add_argument("--test-days",  dest="test_days",  type=int, default=5)
    parser.add_argument("--step-days",  dest="step_days",  type=int, default=5)
    parser.add_argument("--embargo-days", dest="embargo_days", type=int, default=1)

    # Signal knobs
    parser.add_argument("--signal-mode", dest="signal_mode",
                        choices=["momentum", "meanrev"], default="momentum")
    parser.add_argument("-K", dest="K", type=int, default=6)
    parser.add_argument("--q-in",  dest="q_in",  type=float, default=0.30)
    parser.add_argument("--q-out", dest="q_out", type=float, default=0.55)

    # Execution knobs
    parser.add_argument("--exec-mode", dest="exec_mode",
                        choices=["ratelimit", "band", "grid"], default="ratelimit")
    parser.add_argument("--step-bps", dest="step_bps", type=float, default=10.0)
    parser.add_argument("--band-bps", dest="band_bps", type=float, default=0.0)
    parser.add_argument("--grid-bps", dest="grid_bps", type=float, default=0.0)
    parser.add_argument("--rebars", dest="rebars", type=int, default=6)

    args = parser.parse_args()

    cfg  = yaml.safe_load(open("config.yml"))
    bars = pd.read_parquet(args.data).sort_values(["symbol", "datetime"])

    run_kwargs = dict(
        sectors=None,
        gross=cfg["portfolio"]["gross_leverage"],
        max_w=cfg["portfolio"]["max_weight_bps"] / 1e4,
        cost_bps=cfg["costs"]["half_spread_bps"] + cfg["costs"]["impact_bps"],
        skip_first_min=cfg["filters"]["skip_first_min"],
        skip_last_min=cfg["filters"]["skip_last_min"],
        exec_mode=args.exec_mode,
        step_bps=args.step_bps,
        band_bps=args.band_bps,
        grid_bps=args.grid_bps,
        rebalance_every_n_bars=args.rebars,
        signal_mode=args.signal_mode,
        K=args.K,
        q_in=args.q_in,
        q_out=args.q_out,
    )

    rows = []
    for (tr0, tr1, te0, te1) in windows(
        bars["datetime"], args.train_days, args.test_days, args.step_days, args.embargo_days
    ):
        choice = grid_search_on_train(bars, tr0, tr1, run_kwargs)
        if not choice:
            continue

        # 2) Evaluate once on TRAIN (to log) and once on TEST (true OOS)
        s_tr = run_slice(bars, tr0, tr1, run_kwargs | choice)
        s_te = run_slice(bars, te0, te1, run_kwargs | choice)
        if s_tr and s_te:
            rows.append({
                "train": f"{tr0.date()}→{tr1.date()}",
                "test":  f"{te0.date()}→{te1.date()}",
                "SR_raw_tr": s_tr["Ann. Sharpe (raw)"],
                "SR_net_tr": s_tr["Ann. Sharpe (net)"],
                "SR_raw_te": s_te["Ann. Sharpe (raw)"],
                "SR_net_te": s_te["Ann. Sharpe (net)"],
                "Turnover/day_te": s_te["Turnover/day"],
                "Cost_bps/day_te": s_te["Avg cost/day (bps)"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No folds produced — check date range or data.")
    else:
        print(df)
        print("Avg OOS net Sharpe:", float(df["SR_net_te"].mean()))
        df.to_csv("experiments/walkforward.csv", index=False)
