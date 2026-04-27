"""
test_oos.py — Out-of-Sample Evaluation on CSCO (Unseen Ticker)
==============================================================
All three models were trained on AAPL (July 2023) only.
This script loads the full CSCO July 2023 dataset — data the models
have never seen — and reports:

  • Tick-level ROC-AUC and Brier score
  • Day-level AUC mean ± std  (honest uncertainty, avoids autocorrelation)
  • ROC curve for each model  → plots/oos_roc_curves.png
  • Calibration curve          → plots/oos_calibration.png
  • Per-day AUC bar chart      → plots/oos_day_auc.png

Run:  python test_oos.py
"""

import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from lobster_data import discover_files, build_dataset, apply_vol_regime
from fill_estimator import FillEstimator

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

CSCO_DIR = os.path.join(
    BASE, "Data", "CSCO_2023-07-01_2023-07-31_10",
    "output-2023-07", "0", "0", "75",
)
MODELS_DIR = os.path.join(BASE, "models")
PLOTS_DIR  = os.path.join(BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plot style (matches lob_visualizer.py dark palette) ──────────────────────

BG      = "#0f172a"
PANEL   = "#1e293b"
GRID    = "#334155"
TEXT    = "#e2e8f0"
COLORS  = ["#38bdf8", "#f472b6", "#4ade80"]   # LR, RF, GBM

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
})

# ── Load models ────────────────────────────────────────────────────────────────

print("Loading trained models …")
estimator: FillEstimator = FillEstimator.load(os.path.join(MODELS_DIR, "fill_estimator.pkl"))
lr_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
rf_model  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
gbm_model = joblib.load(os.path.join(MODELS_DIR, "gbm.pkl"))
calibrators = joblib.load(os.path.join(MODELS_DIR, "platt_calibrators.pkl"))

FEATURE_NAMES = estimator.feature_names
VOL_THRESHOLD = estimator.vol_threshold

MODELS = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "GBM":                 gbm_model,
}

# ── Load CSCO data ─────────────────────────────────────────────────────────────

print("Discovering CSCO files …")
csco_files = discover_files(CSCO_DIR)
if not csco_files:
    raise FileNotFoundError(f"No CSCO file pairs found in {CSCO_DIR}")

print(f"Found {len(csco_files)} CSCO trading days. Building dataset …")
csco_df = build_dataset(csco_files, ticker="CSCO")
print(f"  CSCO rows: {len(csco_df):,}  |  fill rate: {csco_df['filled'].mean():.3f}")

# Apply the AAPL-fitted vol_regime threshold — no re-fitting on CSCO.
csco_df = apply_vol_regime(csco_df, VOL_THRESHOLD)

# Ensure all required features are present.
missing = [f for f in FEATURE_NAMES if f not in csco_df.columns]
if missing:
    raise KeyError(f"Missing features in CSCO dataset: {missing}")

X_csco = csco_df[FEATURE_NAMES]
y_csco = csco_df["filled"]

# ── Evaluation helpers ─────────────────────────────────────────────────────────

def day_level_auc(df: pd.DataFrame, prob_col: str = "prob") -> tuple[float, float]:
    vals = []
    for _, grp in df.groupby("date"):
        if grp["filled"].nunique() < 2:
            continue
        vals.append(roc_auc_score(grp["filled"], grp[prob_col]))
    return float(np.mean(vals)), float(np.std(vals))


def day_level_brier(df: pd.DataFrame, prob_col: str = "prob") -> tuple[float, float]:
    vals = [brier_score_loss(grp["filled"], grp[prob_col])
            for _, grp in df.groupby("date")]
    return float(np.mean(vals)), float(np.std(vals))


# ── Run evaluation ────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"{'Model':<22}  {'Tick AUC':>9}  {'Day AUC':>16}  {'Day Brier':>14}")
print("=" * 65)

results = {}
prob_df = csco_df[["date", "filled"]].copy()

for name, model in MODELS.items():
    probs = model.predict_proba(X_csco)[:, 1]
    prob_df[name] = probs

    tick_auc   = roc_auc_score(y_csco, probs)
    tick_brier = brier_score_loss(y_csco, probs)

    tmp = prob_df[["date", "filled", name]].rename(columns={name: "prob"})
    day_auc_m,   day_auc_s   = day_level_auc(tmp)
    day_brier_m, day_brier_s = day_level_brier(tmp)

    results[name] = {
        "tick_auc":       tick_auc,
        "tick_brier":     tick_brier,
        "day_auc_mean":   day_auc_m,
        "day_auc_std":    day_auc_s,
        "day_brier_mean": day_brier_m,
        "day_brier_std":  day_brier_s,
        "probs":          probs,
    }

    print(
        f"{name:<22}  {tick_auc:>9.4f}  "
        f"{day_auc_m:>6.4f} ± {day_auc_s:.4f}  "
        f"{day_brier_m:>6.4f} ± {day_brier_s:.4f}"
    )

print("=" * 65)

# ── Calibrated metrics (Platt scaling) ───────────────────────────────────────
print(f"\n{'Model':<22}  {'Tick AUC':>9}  {'Day AUC':>16}  {'Day Brier':>14}  {'Note':>12}")
print("-" * 80)
for name, res in results.items():
    raw_probs = res["probs"]
    cal_probs = calibrators[name].predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    tmp = prob_df[["date", "filled"]].copy()
    tmp["prob"] = cal_probs
    tick_auc   = roc_auc_score(y_csco, cal_probs)
    tick_brier = brier_score_loss(y_csco, cal_probs)
    day_auc_m, day_auc_s     = day_level_auc(tmp)
    day_brier_m, day_brier_s = day_level_brier(tmp)
    results[name]["cal_probs"]      = cal_probs
    results[name]["cal_tick_auc"]   = tick_auc
    results[name]["cal_brier"]      = tick_brier
    results[name]["cal_day_auc_m"]  = day_auc_m
    results[name]["cal_day_auc_s"]  = day_auc_s
    print(f"{name:<22}  {tick_auc:>9.4f}  "
          f"{day_auc_m:>6.4f} ± {day_auc_s:.4f}  "
          f"{day_brier_m:>6.4f} ± {day_brier_s:.4f}  Platt-cal")
print("-" * 80)

# ── Aggregate calibration check ───────────────────────────────────────────────
# Sum of predicted probabilities should equal the total number of actual fills
# if the model is well-calibrated in aggregate. A ratio far from 1.0 means the
# model is systematically over- or under-predicting on CSCO vs AAPL.

actual_fills = int(y_csco.sum())
print(f"\nAggregate calibration (CSCO fill rate: {y_csco.mean():.4f})")
print(f"  Actual fills: {actual_fills:,}")
print(f"  {'Model':<22}  {'Raw ratio':>10}  {'Cal ratio':>10}  {'Raw bias':>10}  {'Cal bias':>10}")
print(f"  {'-'*68}")
for name, res in results.items():
    raw_pred = res["probs"].sum()
    cal_pred = res["cal_probs"].sum()
    raw_ratio = raw_pred / actual_fills
    cal_ratio = cal_pred / actual_fills
    print(f"  {name:<22}  {raw_ratio:>10.4f}  {cal_ratio:>10.4f}  "
          f"{raw_pred - actual_fills:>+10.0f}  {cal_pred - actual_fills:>+10.0f}")

# ── Plot 1: ROC curves ────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)

for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_csco, res["probs"])
    ax.plot(fpr, tpr, color=color, lw=1.8,
            label=f"{name}  AUC={res['tick_auc']:.4f}")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("OOS ROC Curves — CSCO (AAPL-trained models)")
ax.legend(fontsize=9)
ax.grid(True, lw=0.5)
fig.tight_layout()
path = os.path.join(PLOTS_DIR, "oos_roc_curves.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved {path}")

# ── Plot 2: Calibration curves ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1, label="Perfect calibration")

for (name, res), color in zip(results.items(), COLORS):
    frac_pos, mean_pred = calibration_curve(y_csco, res["probs"], n_bins=15)
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=1.5, ms=4, label=name)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("OOS Calibration — CSCO (AAPL-trained models)")
ax.legend(fontsize=9)
ax.grid(True, lw=0.5)
fig.tight_layout()
path = os.path.join(PLOTS_DIR, "oos_calibration.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")

# ── Plot 3: Per-day AUC (GBM) ────────────────────────────────────────────────

tmp = prob_df[["date", "filled", "GBM"]].rename(columns={"GBM": "prob"})
day_aucs = {}
for date, grp in tmp.groupby("date"):
    if grp["filled"].nunique() < 2:
        continue
    day_aucs[date] = roc_auc_score(grp["filled"], grp["prob"])

dates = list(day_aucs.keys())
aucs  = [day_aucs[d] for d in dates]

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(dates))
ax.bar(x, aucs, color=COLORS[2], alpha=0.85, width=0.6)
ax.axhline(np.mean(aucs), color=TEXT, lw=1.2, ls="--",
           label=f"Mean = {np.mean(aucs):.4f}")
ax.set_xticks(x)
ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ROC-AUC")
ax.set_title("GBM Day-Level AUC — CSCO OOS (AAPL-trained)")
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=9)
ax.grid(True, axis="y", lw=0.5)
fig.tight_layout()
path = os.path.join(PLOTS_DIR, "oos_day_auc.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")

# ── Save per-row predictions ──────────────────────────────────────────────────

out_path = os.path.join(BASE, "csco_oos_predictions.csv")
save_cols = ["date", "filled"] + list(MODELS.keys())
prob_df[save_cols].rename(
    columns={k: f"prob_{k.lower().replace(' ', '_')}" for k in MODELS}
).to_csv(out_path, index=False)
print(f"Saved per-row predictions → {out_path}")

print("\nDone.")
