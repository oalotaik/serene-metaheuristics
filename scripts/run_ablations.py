"""Ablation study: which parts of SERENE-MH drive the win?

Under a fixed host (ALNS) and the same operator portfolio, compare the full
controller against versions with one component removed:

  serene-full     contextual TS + surrogate-gated slate (the real thing)
  serene-noctx    non-contextual TS + gating          (drop the context features)
  serene-nogate   contextual TS, no gating            (slate_size = n_exec = 1)
  serene-plainTS  non-contextual TS, no gating         (drop both)
  serene-noexpl   contextual, gating, no exploration   (Thompson noise nu = 0)
  roulette        ALNS's native adaptive selection     (reference)
  uniform         random operator choice               (control)

Efficiency (AUC) and final gap are compared against serene-full, blocked by
instance. Writes CSVs to results/tables/tsp_ablations/ and a figure.

    python scripts/run_ablations.py
"""

import os

import pandas as pd

from serene_mh.core import RecordToRecord, Roulette, UniformRandom
from serene_mh.controller import SereneMH
from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table, within_host_vs
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
OUT = "results/tables/tsp_ablations"
FIGURES = "results/figures"


def _alns():
    return RecordToRecord(deviation=0.02)


CONFIGS = [
    Config("serene-full|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, use_context=True), _alns),
    Config("serene-noctx|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, use_context=False), _alns),
    Config("serene-nogate|ALNS", lambda a: SereneMH(a, slate_size=1, n_exec=1, use_context=True), _alns),
    Config("serene-plainTS|ALNS", lambda a: SereneMH(a, slate_size=1, n_exec=1, use_context=False), _alns),
    Config("serene-noexpl|ALNS", lambda a: SereneMH(a, slate_size=3, n_exec=1, use_context=True, exploration=0.0), _alns),
    Config("roulette|ALNS", lambda a: Roulette(a), _alns),
    Config("uniform|ALNS", lambda a: UniformRandom(a), _alns),
]


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"ablations: {len(instances)} TSPLIB x {len(CONFIGS)} configs x {len(SEEDS)} seeds, {MAX_EVALS} evals")

    study = run_study(instances, CONFIGS, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)
    df = enrich(study, optima)

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    for value in ("auc", "gap"):
        rank_table(df, value).to_csv(os.path.join(OUT, f"ranks_{value}.csv"), index=False)
        within_host_vs(df, value, reference="serene-full").to_csv(
            os.path.join(OUT, f"vs_full_{value}.csv"), index=False)
    df.groupby("selector")[["gap", "auc", "q25"]].mean().reset_index().to_csv(
        os.path.join(OUT, "means.csv"), index=False)
    print(f"wrote ablation tables to {OUT}/")

    # console summary
    print("\nmean AUC (lower=better) and gap by config:")
    print(df.groupby("selector")[["auc", "gap"]].mean().sort_values("auc").round(4).to_string())

    os.makedirs(FIGURES, exist_ok=True)
    order = ["serene-full|ALNS", "serene-nogate|ALNS", "serene-noctx|ALNS",
             "serene-plainTS|ALNS", "roulette|ALNS"]
    plot_regret_curves(study, order, os.path.join(FIGURES, "tsp_ablations_alns"),
                       references=optima, title="SERENE-MH ablations (TSPLIB, ALNS host)")


if __name__ == "__main__":
    main()
