import pandas as pd, numpy as np, yaml
from src.features import zscore_vwap_dev
from src.backtest import to_wide

cfg = yaml.safe_load(open("config.yml"))
bars = pd.read_parquet("data/minute_bars.parquet").sort_values(["symbol","datetime"])

lb = cfg["signal"]["vol_lookback_min"] // cfg["bar_minutes"]
feat = zscore_vwap_dev(bars, lb)  # uses VWAP(t-1) per our last patch

Z   = to_wide(feat, "z")
DEV = to_wide(feat, "close") - to_wide(feat, "vwap")
O   = to_wide(feat, "open")
C   = to_wide(feat, "close")

# time-of-day mask (quiet middle of day)
idxL = Z.index.tz_convert("America/New_York")
mins_from_open = ((idxL - idxL.normalize()) - pd.Timedelta(hours=9, minutes=30)).total_seconds()/60
mins_to_close  = (pd.Timedelta(hours=16) - (idxL - idxL.normalize())).total_seconds()/60
mask = (mins_from_open >= cfg["filters"]["skip_first_min"]) & (mins_to_close >= cfg["filters"]["skip_last_min"])

def qls(sig, q=0.2):
    r = sig.rank(axis=1, method="first", pct=True)
    long  = (r <= q).astype(float)
    short = (r >= 1-q).astype(float) * -1.0
    return long + short

for H in [1, 3, 6, 9, 12]:
    # Trade at t+1 open, hold H bars, exit at t+H close (forward H-bar return)
    R = (C.shift(-H) / O.shift(-1) - 1.0)

    Zm, Dm, Rm = Z.loc[mask], DEV.loc[mask], R.loc[mask]

    ic_z   = Zm.mul(-1).corrwith(Rm, axis=1).mean()   # -Z for mean reversion
    ic_dev = Dm.mul(-1).corrwith(Rm, axis=1).mean()   # -(close - vwap)

    w = qls(Zm, q=0.2)
    ls = (w * Rm).sum(axis=1).mean()

    print(f"H={H:2d}:  IC_z={ic_z:.5g}  IC_dev={ic_dev:.5g}  mean L-S ret={ls:.5g}")
