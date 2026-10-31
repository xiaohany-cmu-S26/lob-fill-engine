"""
order_size_summary.py
---------------------
Aggregate fill statistics for AAPL and CSCO across all available trading days.

For each stock and combined, counts limit orders bucketed into log10 size tiers:
    [1,10)  [10,100)  [100,1000)  [1000,10000)  [10000,∞)

Reports: total orders, filled, not filled, fill ratio.
"""

import os
import sys

import numpy as np
import pandas as pd

# ── Reuse existing pipeline ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lobster_data import discover_files, build_dataset

BASE     = os.path.dirname(os.path.abspath(__file__))
AAPL_DIR = os.path.join(BASE, "Data", "AAPL_2023-07-01_2023-07-31_10",
                         "output-2023-07", "0", "0", "13")
CSCO_DIR = os.path.join(BASE, "Data", "CSCO_2023-07-01_2023-07-31_10",
                         "output-2023-07", "0", "0", "75")

# Log10 size tier edges  (right-open: [lo, hi))
TIER_EDGES  = [1, 10, 100, 1_000, 10_000, np.inf]
TIER_LABELS = ["[1, 10)", "[10, 100)", "[100, 1 000)", "[1 000, 10 000)", "[10 000+)"]


def assign_tier(sizes: pd.Series) -> pd.Categorical:
    return pd.cut(
        sizes,
        bins=TIER_EDGES,
        labels=TIER_LABELS,
        right=False,          # left-inclusive, right-exclusive
        include_lowest=True,
    )


def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = assign_tier(df["order_size"])

    grp = df.groupby("tier", observed=False)["filled"].agg(
        total="count",
        filled_count="sum",
    ).reset_index()

    grp["not_filled"]  = grp["total"] - grp["filled_count"]
    grp["fill_ratio"]  = (grp["filled_count"] / grp["total"]).round(4)
    grp["ticker"]      = label

    return grp[["ticker", "tier", "total", "filled_count", "not_filled", "fill_ratio"]]


def print_table(title: str, tbl: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    header = f"  {'Tier':<18} {'Total':>10} {'Filled':>10} {'Not Filled':>12} {'Fill Ratio':>12}"
    print(header)
    print("  " + "-" * 64)
    for _, row in tbl.iterrows():
        if row["total"] == 0:
            continue
        print(f"  {str(row['tier']):<18} {row['total']:>10,} "
              f"{row['filled_count']:>10,} {row['not_filled']:>12,} "
              f"{row['fill_ratio']:>11.2%}")
    totals = tbl[tbl["total"] > 0]
    grand_total  = totals["total"].sum()
    grand_filled = totals["filled_count"].sum()
    grand_ratio  = grand_filled / grand_total if grand_total else 0
    print("  " + "-" * 64)
    print(f"  {'ALL SIZES':<18} {grand_total:>10,} "
          f"{grand_filled:>10,} {grand_total - grand_filled:>12,} "
          f"{grand_ratio:>11.2%}")


def main() -> None:
    print("Loading AAPL …", end=" ", flush=True)
    aapl_pairs = discover_files(AAPL_DIR)
    df_aapl    = build_dataset(aapl_pairs, "AAPL")
    print(f"{len(df_aapl):,} limit-order rows across {len(aapl_pairs)} days")

    print("Loading CSCO …", end=" ", flush=True)
    csco_pairs = discover_files(CSCO_DIR)
    df_csco    = build_dataset(csco_pairs, "CSCO")
    print(f"{len(df_csco):,} limit-order rows across {len(csco_pairs)} days")

    df_all = pd.concat([df_aapl, df_csco], ignore_index=True)

    aapl_tbl = summarise(df_aapl, "AAPL")
    csco_tbl = summarise(df_csco, "CSCO")
    all_tbl  = summarise(df_all,  "AAPL+CSCO")

    print_table("AAPL - Order Size Fill Summary (all days, July 2023)", aapl_tbl)
    print_table("CSCO - Order Size Fill Summary (all days, July 2023)", csco_tbl)
    print_table("AAPL + CSCO Combined - Order Size Fill Summary",       all_tbl)

    # ── Per-stock daily breakdown ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Per-Day Fill Rates by Size Tier")
    print(f"{'=' * 70}")

    for ticker, df in [("AAPL", df_aapl), ("CSCO", df_csco)]:
        df = df.copy()
        df["tier"] = assign_tier(df["order_size"])
        daily = (
            df.groupby(["date", "tier"], observed=False)["filled"]
            .agg(total="count", filled_count="sum")
            .reset_index()
        )
        daily["fill_ratio"] = (daily["filled_count"] / daily["total"]).round(4)
        pivot = daily.pivot(index="date", columns="tier", values="fill_ratio")
        pivot = pivot[[t for t in TIER_LABELS if t in pivot.columns]]
        print(f"\n  {ticker}  (fill_ratio by tier × day)")
        print("  " + pivot.to_string().replace("\n", "\n  "))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    combined = pd.concat([aapl_tbl, csco_tbl, all_tbl], ignore_index=True)
    out_path = os.path.join(BASE, "order_size_fill_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
