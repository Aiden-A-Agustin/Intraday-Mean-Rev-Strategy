# Intraday US Equities Strategy (Mean Reversion / Momentum)

This project implements an **intraday cross-sectional equity strategy** on US stocks using minute bars.  
It supports both **momentum** and **mean reversion** signals, realistic **execution models**, and a **walk-forward framework** for out-of-sample testing.



---

## Features

- **Signal generation**
  - K-bar return on log prices (configurable horizon).
  - `momentum` (long recent winners) or `meanrev` (long recent losers).
- **Portfolio construction**
  - Sticky membership with hysteresis quantiles (`q_in`, `q_out`) to reduce churn.
  - Sector / market-neutral options.
- **Execution models**
  - **Rate-limit in time**: gradual move to target with `step_bps` cap.
  - **No-trade band**: ignore small target changes.
- **Transaction costs**
  - Half-spread + impact (bps).
  - Turnover and cost/day explicitly reported.
- **Walk-forward**
  - Rolling train/test split with embargo days to reduce look-ahead bias.
  - Reports raw vs net Sharpe OOS.

---

## Repository structure

├── README.md
├── requirements.txt
├── requirements-dev.txt # optional (tests + linting)
├── config.yml # session times, costs, portfolio bounds
├── data/ # ⚠️ ignored (minute bar parquet files)
├── scripts/
│ ├── run_backtest.py # single run
│ └── walkforward.py # rolling train/test
├── src/
│ ├── backtest.py # vectorized backtest & execution
│ ├── features.py # signal construction
│ ├── metrics.py # summary metrics
│ └── utils.py # helpers
└── tests/ # unit tests

---

## Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/<your-username>/intraday-mean-reversion.git
cd intraday-mean-reversion

# (optional) create a virtual env or conda env first
pip install -r requirements.txt
Data

The project uses minute bar data. By default, we support Alpaca.

Set your API keys as environment variables:

Linux/macOS

export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"


Windows PowerShell

$env:APCA_API_KEY_ID="your_key"
$env:APCA_API_SECRET_KEY="your_secret"


Download data:

python -m scripts.download_alpaca \
  --symbols AAPL MSFT AMZN \
  --start 2025-06-02 --end 2025-07-14 \
  --mins 5 --feed iex \
  --outfile data/minute_bars.parquet

Usage
Single backtest
# from project root
export PYTHONPATH="$PWD"     # PowerShell: $env:PYTHONPATH="$PWD"

python run_backtest.py


Produces Sharpe (raw/net), turnover/day, costs, and drawdowns.

Walk-forward
python -m scripts.walkforward \
  --signal-mode momentum -K 6 \
  --exec-mode ratelimit --step-bps 8 --rebars 8 \
  --q-in 0.35 --q-out 0.60


Key arguments:

--signal-mode {momentum, meanrev}

-K : lookback horizon in bars

--q-in / --q-out : entry/exit quantiles

--exec-mode {ratelimit, band}

--step-bps / --band-bps : execution controls

--rebars : cadence in bars

Output:

train_range | test_range | SR_raw_te | SR_net_te | Turnover/day_te | Cost_bps/day_te