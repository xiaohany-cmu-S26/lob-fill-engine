"""
fill_estimator_fractional.py
----------------------------
Regression-based fill fraction estimator for the MM simulator.

Returns E[fill_fraction] ∈ [0, 1]:
    0.0  the order is almost certainly never reached by executions
    0.5  roughly half the order is expected to be filled
    1.0  the full order is expected to be filled within the horizon

Unlike FillEstimator (binary classifier → P(fully filled)), this model
predicts the continuous fraction of the order absorbed by executions, so
partial fills contribute meaningful signal rather than being collapsed into
the "unfilled" bucket.

EV usage in the MM simulator:
    frac = est.estimate(...)
    ev   = frac * expected_spread_capture - (1 - frac) * inventory_cost

Two entry points
----------------
1. predict_fill_fraction(state)         low-level: all features pre-computed
2. estimate(date, time_of_day, ...)     convenience: computes time encodings
"""

import numpy as np
import pandas as pd
import joblib

_OPEN_SEC = 34_200.0   # 09:30:00 in seconds from midnight
_DAY_MIN  = 390.0      # total trading minutes


def _encode_time_of_day(seconds_from_midnight: float) -> tuple[float, float, int]:
    min_from_open = (seconds_from_midnight - _OPEN_SEC) / 60.0
    angle    = 2.0 * np.pi * min_from_open / _DAY_MIN
    time_sin = float(np.sin(angle))
    time_cos = float(np.cos(angle))
    if   min_from_open < 30:  bucket = 0
    elif min_from_open < 150: bucket = 1
    elif min_from_open < 270: bucket = 2
    else:                      bucket = 3
    return time_sin, time_cos, bucket


def _parse_time_of_day(time_of_day) -> float:
    if isinstance(time_of_day, (int, float)):
        return float(time_of_day)
    parts = str(time_of_day).split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"time_of_day must be HH:MM or HH:MM:SS (got {time_of_day!r})")
    h = int(parts[0]); m = int(parts[1]); s = int(parts[2]) if len(parts) == 3 else 0
    return h * 3600.0 + m * 60.0 + s


class FillEstimatorFractional:
    """
    Wrapper around a fitted sklearn regressor that predicts
    E[fill_fraction] ∈ [0, 1] for a passive limit order.

    vol_regime is inferred from local_vol if not supplied, using the
    training-set median stored at fit time.

    Attributes
    ----------
    model            : fitted sklearn Pipeline or regressor
    feature_names    : ordered list of feature columns the model expects
    vol_threshold    : training-set median of local_vol
    fill_horizon_sec : labelled fill window (seconds)
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

    # ── Low-level ─────────────────────────────────────────────────────────────

    def predict_fill_fraction(self, state: dict) -> float:
        """
        Expected fill fraction in [0, 1] for a single limit order whose
        feature dict already contains every encoded column the model expects.

        vol_regime is inferred from local_vol if absent.
        """
        s = dict(state)
        if "vol_regime" not in s and "local_vol" in s:
            s["vol_regime"] = int(s["local_vol"] > self.vol_threshold)
        row = pd.DataFrame([{f: s[f] for f in self.feature_names}])
        raw = float(self.model.predict(row)[0])
        return float(np.clip(raw, 0.0, 1.0))

    def predict_fill_fraction_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch fill fractions for backtesting.
        df must have all feature columns; vol_regime is inferred if absent.
        """
        df = df.copy()
        if "vol_regime" not in df.columns and "local_vol" in df.columns:
            df["vol_regime"] = (df["local_vol"] > self.vol_threshold).astype(int)
        raw = self.model.predict(df[self.feature_names])
        return np.clip(raw, 0.0, 1.0)

    # ── High-level convenience ────────────────────────────────────────────────

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
        Convenience wrapper — takes intuitive market inputs, returns
        E[fill_fraction] ∈ [0, 1].

        Parameters mirror FillEstimator.estimate() so the two estimators
        are drop-in interchangeable in the MM simulator.
        """
        sec             = _parse_time_of_day(time_of_day)
        sin_, cos_, bkt = _encode_time_of_day(sec)
        dow             = int(pd.Timestamp(date).day_of_week)

        return self.predict_fill_fraction({
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
            "time_bucket":     bkt,
            "day_of_week":     dow,
        })

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FillEstimatorFractional":
        return joblib.load(path)
