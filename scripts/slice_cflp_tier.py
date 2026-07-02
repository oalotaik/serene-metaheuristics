"""Slice a completed CFLP study/ablation results.csv into per-size-tier summaries.

A full run (`run_cflp_study.py --full` or `run_cflp_ablations.py --full`) covers all
37 OR-Library instances across three size tiers. This post-processor re-derives the
same summary tables (gap-by-budget-fraction, within-host Wilcoxon, Friedman ranks)
for one tier, without re-running the expensive job - it just filters results.csv.

The within-host comparison uses `serene-full` as the reference if present (ablation
runs) else `serene` (main-study runs), matching how each run was written.

    python scripts/slice_cflp_tier.py                          # large tier of cflp_main
    python scripts/slice_cflp_tier.py --medium                 # 25x50 of cflp_main
    python scripts/slice_cflp_tier.py --src cflp_full_ablations # large tier of the ablation
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


def main(tier="large", src="cflp_main"):
    df = pd.read_csv(f"results/tables/{src}/results.csv")
    d = df[df["instance"].isin(SIZE_TIERS[tier])].copy()
    out = f"results/tables/{src}_{tier}"
    os.makedirs(out, exist_ok=True)
    ref = "serene-full" if "serene-full" in set(d["selector"]) else "serene"
    vs_name = "vs_full.csv" if ref == "serene-full" else "within_host_efficiency.csv"
    print(f"{src} {tier} tier: {d.instance.nunique()} instances x {d.seed.nunique()} seeds "
          f"(ref={ref}) -> {out}/")

    d.to_csv(os.path.join(out, "results.csv"), index=False)
    pd.concat([rank_table(d, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "ranks.csv"), index=False)
    pd.concat([within_host_vs(d, v, reference=ref) for v in ["auc"] + CUTOFFS],
              ignore_index=True).to_csv(os.path.join(out, vs_name), index=False)
    gap = d.groupby(["host", "selector"])[["g25", "g50", "g75", "g100", "auc"]].mean()
    gap[["g25", "g50", "g75", "g100"]] *= 100  # to percent, matching the run drivers
    gap.reset_index().to_csv(os.path.join(out, "gap_by_budget_fraction.csv"), index=False)
    print(f"wrote results + summary tables to {out}/")


if __name__ == "__main__":
    args = sys.argv[1:]
    tier = "small" if "--small" in args else "medium" if "--medium" in args else "large"
    src = "cflp_main"
    if "--src" in args:
        src = args[args.index("--src") + 1]
    main(tier=tier, src=src)
