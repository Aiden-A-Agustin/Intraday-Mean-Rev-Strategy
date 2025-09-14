from __future__ import annotations
import os
from datetime import datetime
from typing import Sequence
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


NY = "America/New_York"




def _get_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.")
    return StockHistoricalDataClient(key, sec)




def fetch_alpaca_minute_bars(
    symbols: Sequence[str],
    start: str | datetime,
    end: str | datetime,
    timeframe_minutes: int = 5,
    feed: str = "iex", # or "sip" if you are subscribed
    adjustment: str = "split", # split‑adjusted prices
    regular_hours_only: bool = True,
) -> pd.DataFrame:
    """Return bars with columns: symbol, datetime (UTC), open, high, low, close, volume."""
    client = _get_client()


    # Alpaca expects UTC timestamps; accept strings or datetimes in any tz.
    start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz=NY).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz=NY).tz_convert("UTC")


    tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=tf,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        feed=feed,
        adjustment=adjustment,
    )
    bars = client.get_stock_bars(req)
    df = bars.df # MultiIndex (symbol, timestamp)
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol","datetime","open","high","low","close","volume"]) # empty


    df = df.reset_index().rename(columns={"timestamp": "datetime"})


    # Keep only core columns and sort
    df = df[["symbol","datetime","open","high","low","close","volume"]].copy()


    # Ensure tz‑aware UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)


    if regular_hours_only:
        # Filter to 09:30–16:00 New York time. (Alpaca returns pre/post; we drop them.)
        dt_local = df["datetime"].dt.tz_convert(NY)
        mask = (dt_local.dt.time >= pd.Timestamp("09:30").time()) & (dt_local.dt.time <= pd.Timestamp("16:00").time())
        df = df[mask]


    df.sort_values(["symbol","datetime"], inplace=True)
    return df