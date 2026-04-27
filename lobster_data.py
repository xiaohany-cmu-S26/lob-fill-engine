"""
lobster_data.py
---------------
Shared data-loading and label-construction utilities for the
LOB fill estimator project.

Intended to be imported by:
  • lob_visualizer.py  (interactive chart)
  • training scripts   (feature / label pipelines)
  • testing / eval scripts

Pipeline order
--------------
  files   = discover_files(data_dir)
  msg, book = load_lobster(f["msg"], f["ob"], f["date"])   # per day
  features  = compute_features(msg, book)                  # per day
  labels    = construct_fill_labels(msg, book)             # per day
  df        = build_dataset(files, ticker)                 # multi-day
  train, val, test = make_splits(df)
  thresh    = fit_vol_regime(train)
  train, val, test = [apply_vol_regime(s, thresh)
                      for s in (train, val, test)]
"""

import os
import glob
import re
from datetime import timedelta

import numpy as np
import pandas as pd

# ── Schema constants ───────────────────────────────────────────────────────────

MAX_DEPTH        = 10     # price levels per side in the LOBSTER files
FILL_HORIZON_SEC = 5.0    # look-forward window for fill labels (seconds)
EMBARGO_SEC      = 30.0   # seconds to drop at the start of each val/test period
AUCTION_ROWS     = 60     # leading rows to drop (opening auction artefacts)

# Orderbook column names: ask_p{i}, ask_v{i}, bid_p{i}, bid_v{i}  (v = volume)
OB_COLS: list[str] = []
for _i in range(1, MAX_DEPTH + 1):
    OB_COLS += [f"ask_p{_i}", f"ask_v{_i}", f"bid_p{_i}", f"bid_v{_i}"]

# Price columns sit at even indices (0, 2, 4, …):  ask_p1, bid_p1, ask_p2, …
OB_PRICE_COLS: list[str] = OB_COLS[::2]

# Intraday time constants
_OPEN_SEC  = 34_200.0    # 09:30:00 in seconds from midnight
_CLOSE_SEC = 57_600.0    # 16:00:00
_DAY_MIN   = 390.0       # total trading minutes (390 = 6.5 h)


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_files(data_dir: str) -> list[dict]:
    """
    Scan a LOBSTER output directory and return file-pair dicts sorted by date.
    Only dates that have BOTH a message file and an orderbook file are included.

    Parameters
    ----------
    data_dir : directory containing  <ticker>_<date>_…_message_10.csv
               and                   <ticker>_<date>_…_orderbook_10.csv

    Returns
    -------
    list of dicts, each with keys:
        'date' : 'YYYY-MM-DD'
        'msg'  : absolute path to message file
        'ob'   : absolute path to orderbook file
    """
    ob_files = glob.glob(os.path.join(data_dir, "*_orderbook_10.csv"))
    pairs = []
    for ob_path in ob_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(ob_path))
        if not m:
            continue
        date     = m.group(1)
        msg_path = ob_path.replace("_orderbook_10.csv", "_message_10.csv")
        if os.path.exists(msg_path):
            pairs.append({"date": date, "msg": msg_path, "ob": ob_path})
    return sorted(pairs, key=lambda d: d["date"])


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_lobster(msg_path: str, book_path: str,
                 date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and normalise one day's LOBSTER message + orderbook CSV files,
    applying data-quality filters before returning.

    Parameters
    ----------
    msg_path  : path to  <ticker>_<date>_…_message_10.csv
    book_path : path to  <ticker>_<date>_…_orderbook_10.csv
    date      : trading date 'YYYY-MM-DD' — used as origin for timestamp parsing

    Returns (msg, book) where both DataFrames are row-aligned.

    msg columns
    -----------
    timestamp  : pd.Timestamp  (wall-clock UTC; seconds-since-midnight converted)
    event_type : int  1=new limit, 2=cancel partial, 3=cancel full,
                      4=exec visible, 5=exec hidden  (7=halt rows are removed)
    order_id   : int
    size       : int  (shares)
    price      : float  (USD; raw tick / 10 000)
    direction  : int   1=buy, -1=sell

    book columns
    ------------
    ask_p{i}, ask_v{i}, bid_p{i}, bid_v{i}  for i = 1 … MAX_DEPTH
    Prices in USD; volumes in shares.
    Row k = LOB state *after* the event on row k of msg.

    Quality filters applied
    -----------------------
    • Drop rows with event_type == 7  (trading halt)
    • Drop rows where ask_p1 <= bid_p1  (crossed book — data artefact)
    • Drop the first AUCTION_ROWS rows  (opening auction clearing)
    """
    msg = pd.read_csv(
        msg_path, header=None,
        names=["timestamp", "event_type", "order_id", "size", "price", "direction"],
    )
    msg["price"]     = msg["price"] / 10_000
    msg["timestamp"] = pd.to_datetime(msg["timestamp"], unit="s", origin=date)

    book = pd.read_csv(book_path, header=None, names=OB_COLS)
    book[OB_PRICE_COLS] = book[OB_PRICE_COLS] / 10_000

    # Align lengths (guard against truncated file writes)
    n = min(len(msg), len(book))
    assert n > 0, f"Empty files: {msg_path}"
    msg  = msg.iloc[:n]
    book = book.iloc[:n]

    # ── Quality filter 1: remove trading halt rows ─────────────────────────────
    valid = msg["event_type"] != 7
    msg, book = msg[valid], book[valid]

    # ── Quality filter 2: drop crossed-book rows ───────────────────────────────
    valid = book["ask_p1"] > book["bid_p1"]
    msg, book = msg[valid], book[valid]

    # ── Quality filter 3: drop opening auction rows ────────────────────────────
    msg  = msg.iloc[AUCTION_ROWS:]
    book = book.iloc[AUCTION_ROWS:]

    return msg.reset_index(drop=True), book.reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────────

def compute_features(msg: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the feature matrix for one day's aligned (msg, book) pair.

    Returns a DataFrame with one row per event (same length as msg/book).
    To get features at limit-order submission points only, filter afterwards:
        feats = compute_features(msg, book)
        lo_feats = feats.iloc[labels["book_idx"]]

    All features are computed using only information available at time t
    (no future data leakage).  Rolling windows are bounded to the current day.

    Features
    --------
    imbalance       : (bid_v1 - ask_v1) / (bid_v1 + ask_v1)   bounded [-1, 1]
    depth_imbalance : same ratio summed over levels 1-3
    queue_ahead     : volume at best level on the order's own side
                      (bid_v1 for buys, ask_v1 for sells)
    spread_norm     : (ask_p1 - bid_p1) / mid_price            stationary
    local_vol       : rolling std of mid_price over past 30 s  (min_periods=2)
    aggressive_flow : rolling count of exec events (type 4 or 5) over past 10 s
    direction       : 1=buy, -1=sell  (from msg)
    time_sin        : sin(2π × minute_from_open / 390)         cyclical encoding
    time_cos        : cos(2π × minute_from_open / 390)
    time_bucket     : 0=open(<30 min), 1=mid-morning, 2=midday, 3=close(>270 min)

    Note: vol_regime is NOT computed here — it requires the training-set median
    of local_vol and must be fitted on training data then applied with
    apply_vol_regime().
    """
    # ── Book-state features (vectorized across all rows) ───────────────────────
    bid_v1 = book["bid_v1"].values.astype(float)
    ask_v1 = book["ask_v1"].values.astype(float)
    bid_p1 = book["bid_p1"].values
    ask_p1 = book["ask_p1"].values
    mid    = (ask_p1 + bid_p1) / 2.0

    total_v1 = bid_v1 + ask_v1
    imbalance = np.where(total_v1 > 0, (bid_v1 - ask_v1) / total_v1, 0.0)

    bid_v3 = sum(book[f"bid_v{i}"].values.astype(float) for i in range(1, 4))
    ask_v3 = sum(book[f"ask_v{i}"].values.astype(float) for i in range(1, 4))
    total_v3 = bid_v3 + ask_v3
    depth_imbalance = np.where(total_v3 > 0,
                               (bid_v3 - ask_v3) / total_v3, 0.0)

    direction   = msg["direction"].values
    queue_ahead = np.where(direction == 1, bid_v1, ask_v1)

    spread_norm = np.where(mid > 0, (ask_p1 - bid_p1) / mid, 0.0)

    # ── Time-based rolling features (require DatetimeIndex) ───────────────────
    ts_idx = pd.DatetimeIndex(msg["timestamp"])

    mid_series  = pd.Series(mid, index=ts_idx)
    exec_series = pd.Series(
        msg["event_type"].isin([4, 5]).astype(float).values, index=ts_idx
    )

    local_vol       = (mid_series.rolling("30s", min_periods=2)
                                 .std()
                                 .fillna(0.0)
                                 .values)
    aggressive_flow = (exec_series.rolling("10s", min_periods=0)
                                  .sum()
                                  .values)

    # ── Intraday seasonality ───────────────────────────────────────────────────
    # seconds from midnight → minutes from open
    t0_date         = pd.Timestamp(ts_idx[0].date())
    sec_from_mid    = (ts_idx - t0_date).total_seconds().values
    min_from_open   = (sec_from_mid - _OPEN_SEC) / 60.0

    angle     = 2 * np.pi * min_from_open / _DAY_MIN
    time_sin  = np.sin(angle)
    time_cos  = np.cos(angle)

    time_bucket = np.select(
        [min_from_open < 30,
         min_from_open < 150,
         min_from_open < 270],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)

    return pd.DataFrame({
        "imbalance":       imbalance,
        "depth_imbalance": depth_imbalance,
        "queue_ahead":     queue_ahead,
        "spread_norm":     spread_norm,
        "local_vol":       local_vol,
        "aggressive_flow": aggressive_flow,
        "direction":       direction,
        "time_sin":        time_sin,
        "time_cos":        time_cos,
        "time_bucket":     time_bucket,
    })


# ── Label construction ─────────────────────────────────────────────────────────

def construct_fill_labels(msg: pd.DataFrame,
                          book: pd.DataFrame,
                          fill_horizon_sec: float = FILL_HORIZON_SEC,
                          ) -> pd.DataFrame:
    """
    Build binary fill labels for every limit-order submission (event_type == 1).

    For each submission a synthetic passive order is placed at the current
    best bid (buy) or best ask (sell).  Label = 1 if any execution event
    (type 4 or 5) at-or-through that price occurs within fill_horizon_sec.

    Implementation: O(M log N) via numpy.searchsorted on the timestamp array,
    where M = number of limit orders and N = total events.  This replaces the
    original O(M·N) DataFrame-filter loop.

    Parameters
    ----------
    msg              : message DataFrame from load_lobster
    book             : orderbook DataFrame from load_lobster
    fill_horizon_sec : look-forward window in seconds

    Returns
    -------
    pd.DataFrame with columns:
        timestamp   : pd.Timestamp  time of the submission
        direction   : int   1=buy, -1=sell
        entry_price : float  best bid or ask at submission (USD)
        filled      : int   1 = filled within horizon, 0 = not
        book_idx    : int   positional index into msg / book for this row
    """
    # Pre-extract numpy arrays — avoids repeated pandas indexing in the loop
    ts_ns       = msg["timestamp"].values.astype(np.int64)   # nanoseconds
    horizon_ns  = int(fill_horizon_sec * 1e9)

    prices      = msg["price"].values
    exec_mask   = msg["event_type"].isin([4, 5]).values
    directions  = msg["direction"].values
    bid_p1      = book["bid_p1"].values
    ask_p1      = book["ask_p1"].values
    timestamps  = msg["timestamp"].values

    limit_idxs  = np.where(msg["event_type"].values == 1)[0]
    m           = len(limit_idxs)

    filled_arr      = np.zeros(m, dtype=np.int8)
    entry_price_arr = np.empty(m, dtype=np.float64)

    for k, i in enumerate(limit_idxs):
        d = directions[i]
        entry_price = bid_p1[i] if d == 1 else ask_p1[i]
        entry_price_arr[k] = entry_price

        lo = i + 1
        hi = int(np.searchsorted(ts_ns, ts_ns[i] + horizon_ns, side="right"))

        if lo < hi:
            w_exec   = exec_mask[lo:hi]
            w_prices = prices[lo:hi]
            if d == 1:
                filled_arr[k] = np.any(w_exec & (w_prices >= entry_price))
            else:
                filled_arr[k] = np.any(w_exec & (w_prices <= entry_price))

    return pd.DataFrame({
        "timestamp":   timestamps[limit_idxs],
        "direction":   directions[limit_idxs],
        "entry_price": entry_price_arr,
        "filled":      filled_arr.astype(np.int8),
        "book_idx":    limit_idxs,
    })


# ── Multi-day dataset builder ──────────────────────────────────────────────────

def build_dataset(file_pairs: list[dict],
                  ticker: str,
                  fill_horizon_sec: float = FILL_HORIZON_SEC,
                  ) -> pd.DataFrame:
    """
    Load, featurise, and label all days in file_pairs into one DataFrame.

    Parameters
    ----------
    file_pairs : list of dicts from discover_files()
                 each must have keys: 'date', 'msg', 'ob'
    ticker     : stock symbol string, e.g. 'AAPL'

    Returns
    -------
    pd.DataFrame — one row per limit-order submission across all days.

    Columns
    -------
    All features from compute_features(), plus:
        filled      : int   fill label
        timestamp   : pd.Timestamp
        entry_price : float
        direction   : int
        ticker      : str
        date        : str   'YYYY-MM-DD'
        day_of_week : int   0=Monday … 4=Friday
    """
    frames = []
    for fp in file_pairs:
        date = fp["date"]
        try:
            msg, book = load_lobster(fp["msg"], fp["ob"], date)
        except Exception as e:
            print(f"[build_dataset] skipping {ticker} {date}: {e}")
            continue

        labels   = construct_fill_labels(msg, book, fill_horizon_sec)
        features = compute_features(msg, book)

        if labels.empty:
            continue

        # Align features to limit-order rows using positional index
        lo_feats = features.iloc[labels["book_idx"]].reset_index(drop=True)

        day_df = lo_feats.copy()
        day_df["filled"]      = labels["filled"].values
        day_df["timestamp"]   = labels["timestamp"].values
        day_df["entry_price"] = labels["entry_price"].values
        day_df["direction"]   = labels["direction"].values
        day_df["ticker"]      = ticker
        day_df["date"]        = date
        day_df["day_of_week"] = pd.Timestamp(date).day_of_week

        frames.append(day_df)

    if not frames:
        raise ValueError(f"No valid days loaded for {ticker}.")
    return pd.concat(frames, ignore_index=True)


# ── Train / val / test splits ──────────────────────────────────────────────────

def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    fill_horizon_sec: float = FILL_HORIZON_SEC,
    embargo_sec:      float = EMBARGO_SEC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / val / test split by date, with purging and embargoing.

    Split logic (Lopez de Prado, AFML Ch.7)
    ----------------------------------------
    1. Sort unique dates; assign first 70 % to train, next 15 % to val,
       last 15 % to test.
    2. Purge: drop training rows whose 5-second label window extends into the
       val period (i.e. timestamp > val_start - fill_horizon_sec).
    3. Embargo: drop the first `embargo_sec` seconds of val and test periods
       to prevent information leakage across the boundary.

    Parameters
    ----------
    df               : output of build_dataset()  (must have 'date', 'timestamp')
    train_frac       : fraction of dates for training   (default 0.70)
    val_frac         : fraction of dates for validation  (default 0.15)
    fill_horizon_sec : label horizon for purge calculation
    embargo_sec      : seconds to drop after each split boundary

    Returns
    -------
    (train_df, val_df, test_df)
    """
    dates = sorted(df["date"].unique())
    n     = len(dates)
    if n < 3:
        raise ValueError(f"Need at least 3 dates for a 3-way split; got {n}.")

    n_train = max(1, int(np.floor(n * train_frac)))
    n_val   = max(1, int(np.floor(n * val_frac)))
    # test gets the remainder (avoids rounding gaps)

    train_dates = dates[:n_train]
    val_dates   = dates[n_train: n_train + n_val]
    test_dates  = dates[n_train + n_val:]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df   = df[df["date"].isin(val_dates)].copy()
    test_df  = df[df["date"].isin(test_dates)].copy()

    # First event timestamps at each boundary
    val_start_ts  = val_df["timestamp"].min()
    test_start_ts = test_df["timestamp"].min()

    # ── Purge: drop train rows whose label window bleeds into val ──────────────
    purge_cutoff = val_start_ts - timedelta(seconds=fill_horizon_sec)
    train_df     = train_df[train_df["timestamp"] <= purge_cutoff]

    # ── Embargo: drop leading seconds at val and test boundaries ───────────────
    val_df  = val_df[val_df["timestamp"]   >= val_start_ts  + timedelta(seconds=embargo_sec)]
    test_df = test_df[test_df["timestamp"] >= test_start_ts + timedelta(seconds=embargo_sec)]

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ── Vol-regime feature (train-set fit required) ────────────────────────────────

def fit_vol_regime(train_df: pd.DataFrame) -> float:
    """
    Compute the vol_regime threshold from training data only.

    Returns the median of local_vol across all training rows.
    Apply to val/test with apply_vol_regime() using this threshold.
    """
    return float(train_df["local_vol"].median())


def apply_vol_regime(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Add the vol_regime binary column to df.

    Parameters
    ----------
    df        : any split DataFrame that has a 'local_vol' column
    threshold : value returned by fit_vol_regime(train_df)

    Returns a copy of df with an added 'vol_regime' column (int 0/1).
    """
    return df.assign(vol_regime=(df["local_vol"] > threshold).astype(np.int8))
