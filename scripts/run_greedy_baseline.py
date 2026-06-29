"""Is SERENE-MH's Bayesian value estimate a real contribution, or just greedy?

Discriminating experiment: compare SERENE-MH's greedy config (which exploits a
Bayesian posterior-mean = ridge-shrunk operator value) against a non-Bayesian
GreedyMean baseline that exploits the *raw* running mean. They differ only in how
the per-operator value is estimated, so if SERENE wins - especially on the early-
budget metrics (q25, evals-to-target) where shrinkage helps most - the Bayesian
estimation is the contribution; if they tie, the win is incidental greedy.

ALNS host, 20 TSPLIB instances, 5 seeds, 1500 evals. Tables -> results/tables/tsp_greedy/.

    python scripts/run_greedy_baseline.py
"""

import os

import pandas as pd
from scipy.stats import wilcoxon

from serene_mh.core import RecordToRecord, Roulette, GreedyMean
from serene_mh.controller import SereneMH
from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
OUT = "results/tables/tsp_greedy"
FIGURES = "results/figures"


def _alns():
    return RecordToRecord(deviation=0.02)


CONFIGS = [
    Config("serene-greedy|ALNS", lambda a: SereneMH(a, 3, 1, exploration=0.0), _alns),   # Bayesian shrunk mean
    Config("serene-full|ALNS", lambda a: SereneMH(a, 3, 1, exploration=0.3), _alns),      # Bayesian + exploration
    Config("greedy-mean|ALNS", lambda a: GreedyMean(a, epsilon=0.0), _alns),              # raw mean, greedy
    Config("greedy-mean-eps|ALNS", lambda a: GreedyMean(a, epsilon=0.1), _alns),          # raw mean, eps-greedy
    Config("roulette|ALNS", lambda a: Roulette(a), _alns),
]

# the head-to-head we care about: Bayesian greedy vs raw greedy
CONTRASTS = [("serene-greedy", "greedy-mean"), ("serene-greedy", "greedy-mean-eps")]


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"greedy baseline: {len(instances)} TSPLIB x {len(CONFIGS)} configs x {len(SEEDS)} seeds")

    study = run_study(instances, CONFIGS, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)
    df = enrich(study, optima)

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    for value in ("auc", "q25", "gap"):
        rank_table(df, value).to_csv(os.path.join(OUT, f"ranks_{value}.csv"), index=False)

    rows = []
    for value in ("auc", "q25", "ttt", "gap"):
        piv = df.pivot_table(index="instance", columns="selector", values=value)
        for a, b in CONTRASTS:
            _, p = wilcoxon(piv[a], piv[b])
            winner = a if piv[a].mean() < piv[b].mean() else b
            rows.append({"metric": value, "a": a, "b": b, "winner": winner, "p": p,
                         "mean_a": piv[a].mean(), "mean_b": piv[b].mean()})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(os.path.join(OUT, "bayesian_vs_rawmean.csv"), index=False)
    print("wrote tables to", OUT)

    print("\nmean by config (lower=better):")
    print(df.groupby("selector")[["auc", "q25", "ttt", "gap"]].mean().sort_values("auc").round(4).to_string())
    print("\nBayesian (serene-greedy) vs raw-mean greedy:")
    print(contrasts.round(4).to_string(index=False))

    os.makedirs(FIGURES, exist_ok=True)
    order = ["serene-greedy|ALNS", "greedy-mean|ALNS", "greedy-mean-eps|ALNS",
             "serene-full|ALNS", "roulette|ALNS"]
    plot_regret_curves(study, order, os.path.join(FIGURES, "tsp_greedy_baseline"),
                       references=optima, title="Bayesian vs raw-mean greedy AOS (TSPLIB, ALNS)")


if __name__ == "__main__":
    main()
