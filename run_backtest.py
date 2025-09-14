"""
Run a simple backtest using VWAP z-score signals.

Steps
-----
1. Load intraday bars (minute frequency).
2. Apply universe filters (min price, top by $-volume).
3. Engineer features (VWAP z-score).
4. Run vectorized backtest with given config.
5. Save equity curve and print summary stats.
"""

from __future__ import annotations
import pandas as pd
import yaml
from src.data import load_minute_bars, preprocess, median_dollar_vol
from src.features import zscore_vwap_dev
from src.backtest import vector_backtest
from src.metrics import summary


CONFIG = yaml.safe_load(open("config.yml"))


# 1) Load data
bars = load_minute_bars("data/minute_bars.parquet")
bars = preprocess(bars, CONFIG["universe"]["min_price"]) # sorts, filters


# 2) Universe selection (monthly by median $ volume)
med = median_dollar_vol(bars, 30).sort_values(ascending=False)
universe = set(med.head(CONFIG["universe"]["top_by_median_dollar_vol"]).index)
bars = bars[bars.symbol.isin(universe)].copy()


# 3) Feature engineering
lookback_bars = CONFIG["signal"]["vol_lookback_min"] // CONFIG["bar_minutes"]
feat = zscore_vwap_dev(bars, lookback_bars)


# 4) Backtest
bt = vector_backtest(
    feat,
    sectors=None,
    gross=CONFIG["portfolio"]["gross_leverage"],
    max_w=CONFIG["portfolio"]["max_weight_bps"]/1e4,
    cost_bps=CONFIG["costs"]["half_spread_bps"] + CONFIG["costs"]["impact_bps"],
    #cost_bps=0,
    skip_first_min=CONFIG["filters"]["skip_first_min"],
    skip_last_min=CONFIG["filters"]["skip_last_min"],
)


# 5) Report
print(summary(bt))
(bt["net_ret"].add(1).cumprod()).to_frame("Equity").to_csv("experiments/equity_curve.csv")