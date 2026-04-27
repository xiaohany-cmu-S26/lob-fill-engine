"""
fill_estimator.py
-----------------
Public interface for the LOB fill probability model.

Decoupled from all training infrastructure so the Market Making simulator
can import and call predict_proba() without depending on sklearn pipelines,
feature selection code, or LOBSTER data utilities.

Two entry points
----------------
1.  predict_proba(state)   — low-level: caller pre-computes every model feature
                             (time_sin, time_cos, time_bucket, day_of_week, …)
2.  estimate(...)          — convenience: caller passes intuitive inputs
                             (date, time-of-day, direction, queue_ahead, spread,
                             order_size, …) and the wrapper computes the
                             cyclical / categorical encodings.

Usage
-----
    from fill_estimator import FillEstimator

    est = FillEstimator.load("models/fill_estimator.pkl")

    # convenience API — what the MM simulator calls
    prob = est.estimate(
        date="2023-07-20",
        time_of_day="11:30:00",
        direction=1,                # 1=buy, -1=sell
        order_size=100,             # shares
        queue_ahead=850,            # shares already in queue ahead of you
        spread=0.0002,              # (ask - bid) / mid
        imbalance=0.12,
        depth_imbalance=0.08,
        local_vol=0.015,
        aggressive_flow=3.0,
    )

    ev = prob * expected_spread_capture - (1 - prob) * inventory_cost
"""

import numpy as np
import pandas as pd
import joblib


# Trading-session constants (NASDAQ regular session)
_OPEN_SEC = 34_200.0     # 09:30:00 in seconds from midnight
_DAY_MIN  = 390.0        # total trading minutes (390 = 6.5 h)


def _encode_time_of_day(seconds_from_midnight: float) -> tuple[float, float, int]:
    """
    Encode wall-clock time-of-day into the three model features:
      time_sin, time_cos  — cyclical sin/cos of minute-from-open / 390
      time_bucket         — 0=open(<30 m), 1=mid-morning, 2=midday, 3=close(>270 m)
    """
    min_from_open = (seconds_from_midnight - _OPEN_SEC) / 60.0
    angle    = 2.0 * np.pi * min_from_open / _DAY_MIN
    time_sin = float(np.sin(angle))
    time_cos = float(np.cos(angle))
    if   min_from_open < 30:   bucket = 0
    elif min_from_open < 150:  bucket = 1
    elif min_from_open < 270:  bucket = 2
    else:                       bucket = 3
    return time_sin, time_cos, bucket


def _parse_time_of_day(time_of_day) -> float:
    """Return seconds-from-midnight for an HH:MM[:SS] string or numeric input."""
    if isinstance(time_of_day, (int, float)):
        return float(time_of_day)
    parts = str(time_of_day).split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"time_of_day must be HH:MM or HH:MM:SS (got {time_of_day!r})")
    h = int(parts[0]); m = int(parts[1]); s = int(parts[2]) if len(parts) == 3 else 0
    return h * 3600.0 + m * 60.0 + s


class FillEstimator:
    """
    Wrapper around a fitted sklearn classifier that provides a clean,
    stateless predict_proba() interface for the MM simulator.

    vol_regime is computed automatically from local_vol if not supplied,
    using the training-set median stored at fit time.

    Attributes
    ----------
    model            : fitted sklearn Pipeline or classifier
    feature_names    : ordered list of feature columns the model expects
    vol_threshold    : training-set median of local_vol (for vol_regime inference)
    fill_horizon_sec : labelled fill window (seconds) — embedded for traceability
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        vol_threshold: float,
        fill_horizon_sec: float = 1.0,
    ):
        self.model            = model
        self.feature_names    = feature_names
        self.vol_threshold    = vol_threshold
        self.fill_horizon_sec = fill_horizon_sec

    # ── Low-level: caller passes all model features pre-computed ──────────────

    def predict_proba(self, state: dict) -> float:
        """
        Fill probability in [0, 1] for a single limit order whose feature dict
        already contains every encoded column the model expects.

        vol_regime is inferred from local_vol if omitted.
        """
        s = dict(state)
        if "vol_regime" not in s and "local_vol" in s:
            s["vol_regime"] = int(s["local_vol"] > self.vol_threshold)
        row = pd.DataFrame([{f: s[f] for f in self.feature_names}])
        return float(self.model.predict_proba(row)[0, 1])

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch fill probabilities for backtesting.

        df must have all feature columns; vol_regime is inferred if absent.
        """
        df = df.copy()
        if "vol_regime" not in df.columns and "local_vol" in df.columns:
            df["vol_regime"] = (df["local_vol"] > self.vol_threshold).astype(int)
        return self.model.predict_proba(df[self.feature_names])[:, 1]

    # ── High-level convenience API ────────────────────────────────────────────

    def estimate(
        self,
        date,
        time_of_day,
        direction: int,
        order_size: int,
        queue_ahead: float,
        spread: float,
        imbalance: float = 0.0,
        depth_imbalance: float = 0.0,
        local_vol: float = 0.0,
        aggressive_flow: float = 0.0,
    ) -> float:
        """
        Convenience wrapper that takes intuitive market inputs and computes
        the cyclical / categorical encodings (time_sin, time_cos, time_bucket,
        day_of_week, vol_regime) internally.

        Parameters
        ----------
        date            : 'YYYY-MM-DD' string or pd.Timestamp
        time_of_day     : 'HH:MM' / 'HH:MM:SS' string or seconds-from-midnight
        direction       : 1=buy, -1=sell
        order_size      : your order size in shares (the limit order being placed)
        queue_ahead     : shares already at the best level on your side
        spread          : (ask_p1 - bid_p1) / mid_price  (already normalised)
        imbalance       : (bid_v1 - ask_v1) / (bid_v1 + ask_v1), default 0
        depth_imbalance : same ratio over levels 1-3, default 0
        local_vol       : rolling std of mid_price over past 30 s, default 0
        aggressive_flow : count of executions in past 10 s, default 0

        Returns
        -------
        float probability that the order fills within the model's labelled
        horizon (self.fill_horizon_sec seconds).
        """
        sec  = _parse_time_of_day(time_of_day)
        sin_, cos_, bucket = _encode_time_of_day(sec)
        dow  = int(pd.Timestamp(date).day_of_week)

        return self.predict_proba({
            "imbalance":       imbalance,
            "depth_imbalance": depth_imbalance,
            "queue_ahead":     queue_ahead,
            "spread_norm":     spread,
            "local_vol":       local_vol,
            "aggressive_flow": aggressive_flow,
            "direction":       direction,
            "order_size":      order_size,
            "time_sin":        sin_,
            "time_cos":        cos_,
            "time_bucket":     bucket,
            "day_of_week":     dow,
        })

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FillEstimator":
        return joblib.load(path)
