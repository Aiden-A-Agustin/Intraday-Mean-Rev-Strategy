import pandas as pd
from src.features import incremental_vwap, rolling_vol


def test_vwap_monotone():
    # VWAP should be well‑defined and finite for increasing cumulative volume
    df = pd.DataFrame({
        "symbol": ["AAA"]*4,
        "datetime": pd.date_range("2024-01-01 09:30", periods=4, freq="5min", tz="America/New_York"),
        "high": [10,10,10,10],
        "low": [10,10,10,10],
        "close":[10,10,10,10],
        "volume":[100,200,300,400],
    })
    v = incremental_vwap(df)
    assert v.is_monotonic_increasing is False # flat prices → flat VWAP
    assert v.notna().all()