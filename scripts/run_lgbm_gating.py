"""Does a LightGBM gating surrogate beat the linear (posterior-mean) gating?

Ablation finding: gating is the load-bearing component, and greedy (no Thompson
exploration) was best on TSP. Here we compare, under the ALNS host on the 20
TSPLIB instances:

  serene-linear         linear gating (the current full controller)
  serene-lgbm           LightGBM gating
  serene-lgbm-greedy    LightGBM gating + greedy proposal (nu = 0)
  serene-greedy-linear  linear gating + greedy proposal (the ablation winner)
  roulette              reference

Writes tables to results/tables/tsp_lgbm/ and a figure. Comparisons are vs
serene-linear, blocked by instance.

    python scripts/run_lgbm_gating.py
"""

import os

from serene_mh.core import RecordToRecord, Roulette
from serene_mh.controller import SereneMH
from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table, within_host_vs
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
OUT = "results/tables/tsp_lgbm"
FIGURES = "results/figures"


def _alns():
    return RecordToRecord(deviation=0.02)


CONFIGS = [
    Config("serene-linear|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, gating="linear"), _alns),
    Config("serene-lgbm|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, gating="lgbm"), _alns),
    Config("serene-lgbm-greedy|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, gating="lgbm", exploration=0.0), _alns),
    Config("serene-greedy-linear|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, gating="linear", exploration=0.0), _alns),
    Config("roulette|ALNS", lambda a: Roulette(a), _alns),
]


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"lgbm gating: {len(instances)} TSPLIB x {len(CONFIGS)} configs x {len(SEEDS)} seeds, {MAX_EVALS} evals")

    study = run_study(instances, CONFIGS, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)
    df = enrich(study, optima)

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    for value in ("auc", "gap"):
        rank_table(df, value).to_csv(os.path.join(OUT, f"ranks_{value}.csv"), index=False)
        within_host_vs(df, value, reference="serene-linear").to_csv(
            os.path.join(OUT, f"vs_linear_{value}.csv"), index=False)
    print("wrote tables to", OUT)

    print("\nmean AUC and gap by config (lower=better):")
    print(df.groupby("selector")[["auc", "gap"]].mean().sort_values("auc").round(4).to_string())

    os.makedirs(FIGURES, exist_ok=True)
    order = ["serene-lgbm-greedy|ALNS", "serene-lgbm|ALNS", "serene-greedy-linear|ALNS",
             "serene-linear|ALNS", "roulette|ALNS"]
    plot_regret_curves(study, order, os.path.join(FIGURES, "tsp_lgbm_gating"),
                       references=optima, title="LightGBM vs linear gating (TSPLIB, ALNS host)")


if __name__ == "__main__":
    main()
