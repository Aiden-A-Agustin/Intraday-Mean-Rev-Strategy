from src.vendors.alpaca_loader import fetch_alpaca_minute_bars
import pandas as pd, os

os.makedirs("data", exist_ok=True)
df = fetch_alpaca_minute_bars(
    ["AAPL","MSFT"], "2025-07-01", "2025-07-05",
    timeframe_minutes=5, feed="iex",
    adjustment="split", regular_hours_only=True
)
df.to_parquet("data/minute_bars.parquet", index=False)
print(df.head(), f"\nSaved {len(df):,} rows.")
