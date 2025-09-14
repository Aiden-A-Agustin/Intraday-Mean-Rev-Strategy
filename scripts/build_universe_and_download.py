from __future__ import annotations
import os, math, time
import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

NY = "America/New_York"

def get_keys():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    return key, sec

def active_us_equities():
    key, sec = get_keys()
    tc = TradingClient(key, sec)
    assets = tc.get_all_assets(GetAssetsRequest(status="active", asset_class="us_equity"))
    # keep clean symbols (letters only) that are tradable
    return sorted([a.symbol for a in assets if a.tradable and a.symbol.isalpha()])

def median_dollar_volume(symbols, start, end, feed="iex"):
    key, sec = get_keys()
    dc = StockHistoricalDataClient(key, sec)
    tf = TimeFrame.Day
    out = []
    # chunk symbols so requests are reasonable
    CHUNK = 200
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i+CHUNK]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=tf,
            start=pd.Timestamp(start, tz=NY).tz_convert("UTC").to_pydatetime(),
            end=pd.Timestamp(end, tz=NY).tz_convert("UTC").to_pydatetime(),
            feed=feed,
            adjustment="split",
        )
        bars = dc.get_stock_bars(req).df
        if bars is None or bars.empty:
            continue
        df = bars.reset_index()
        df["dv"] = df["close"] * df["volume"]
        med = df.groupby("symbol")["dv"].median().rename("med_dv")
        out.append(med)
        time.sleep(0.2)  # be nice to the API
    if not out:
        return pd.Series(dtype=float)
    return pd.concat(out).groupby(level=0).max()  # max across chunks to be safe

def fetch_minutes(symbols, start, end, mins=5, feed="iex"):
    key, sec = get_keys()
    dc = StockHistoricalDataClient(key, sec)
    tf = TimeFrame(mins, TimeFrameUnit.Minute)
    parts = []
    CHUNK = 200
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i+CHUNK]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=tf,
            start=pd.Timestamp(start, tz=NY).tz_convert("UTC").to_pydatetime(),
            end=pd.Timestamp(end, tz=NY).tz_convert("UTC").to_pydatetime(),
            feed=feed,
            adjustment="split",
        )
        bars = dc.get_stock_bars(req).df
        if bars is None or bars.empty: 
            continue
        df = bars.reset_index()[["symbol","timestamp","open","high","low","close","volume"]]
        df = df.rename(columns={"timestamp":"datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # regular hours only
        dt_local = df["datetime"].dt.tz_convert(NY)
        mask = (dt_local.dt.time >= pd.Timestamp("09:30").time()) & (dt_local.dt.time <= pd.Timestamp("16:00").time())
        df = df[mask]
        parts.append(df.sort_values(["symbol","datetime"]))
        time.sleep(0.2)
    if not parts:
        return pd.DataFrame(columns=["symbol","datetime","open","high","low","close","volume"])
    return pd.concat(parts, ignore_index=True)

if __name__ == "__main__":
    import argparse, pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--mins", type=int, default=5)
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--feed", default="iex", choices=["iex","sip"])
    p.add_argument("--outfile", default="data/minute_bars.parquet")
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)

    # 1) candidate symbols
    cands = active_us_equities()

    # 2) rank by 30d median dollar volume (daily bars)
    start_dv = (pd.Timestamp(args.end) - pd.Timedelta(days=40)).date().isoformat()
    meddv = median_dollar_volume(cands, start=start_dv, end=args.end, feed=args.feed)
    top_syms = list(meddv.sort_values(ascending=False).head(args.top).index)
    print(f"Selected top {len(top_syms)} symbols by 30d median $ volume.")

    # 3) fetch minute bars for the top set
    df = fetch_minutes(top_syms, start=args.start, end=args.end, mins=args.mins, feed=args.feed)
    if df.empty:
        print("No minute data returned—check keys, dates, or feed permissions.")
    else:
        (df.sort_values(["symbol","datetime"])
           .to_parquet(args.outfile, index=False))
        print(f"Saved {len(df):,} rows for {df['symbol'].nunique()} symbols to {args.outfile}")
