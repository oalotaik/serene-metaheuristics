"""Slice a completed CFLP study's results.csv into per-size-tier summary tables.

The full study (`run_cflp_study.py --full`) covers all 37 OR-Library instances
across three size tiers. This post-processor re-derives the same summary tables
(gap-by-budget-fraction, within-host efficiency Wilcoxon, Friedman ranks) for a
single tier, without re-running the expensive study - it just filters the
committed results.csv.

    python scripts/slice_cflp_tier.py            # large tier (50x50, cap111-134)
    python scripts/slice_cflp_tier.py --medium   # 25x50
    python scripts/slice_cflp_tier.py --small    # 16x50
"""

import os
import sys

import pandas as pd

from serene_mh.experiments.analysis import rank_table, within_host_vs

SIZE_TIERS = {
    "small":  [f"cap{n}" for n in [41, 42, 43, 44, 51, 61, 62, 63, 64, 71, 72, 73, 74]],
    "medium": [f"cap{n}" for n in [81, 82, 83, 84, 91, 92, 93, 94, 101, 102, 103, 104]],
    "large":  [f"cap{n}" for n in [111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134]],
}
CUTOFFS = ["q25", "q50", "q75", "q100"]
SRC = "results/tables/cflp_main/results.csv"


def main(tier="large"):
    df = pd.read_csv(SRC)
    d = df[df["instance"].isin(SIZE_TIERS[tier])].copy()
    out = f"results/tables/cflp_main_{tier}"
    os.makedirs(out, exist_ok=True)
    print(f"{tier} tier: {d.instance.nunique()} instances x {d.seed.nunique()} seeds "
          f"-> {out}/")

    d.to_csv(os.path.join(out, "results.csv"), index=False)
    pd.concat([rank_table(d, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "ranks.csv"), index=False)
    pd.concat([within_host_vs(d, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "within_host_efficiency.csv"), index=False)
    gap = d.groupby(["host", "selector"])[["g25", "g50", "g75", "g100", "auc"]].mean()
    gap[["g25", "g50", "g75", "g100"]] *= 100  # to percent, matching the study driver
    gap.reset_index().to_csv(os.path.join(out, "gap_by_budget_fraction.csv"), index=False)
    print(f"wrote results + efficiency tables to {out}/")


if __name__ == "__main__":
    args = sys.argv[1:]
    selected = "small" if "--small" in args else "medium" if "--medium" in args else "large"
    main(tier=selected)
