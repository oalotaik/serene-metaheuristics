"""Does context matter under the *exploratory* hosts (SA, Tabu)?

On ALNS the context features added nothing. But SERENE-MH shone most under SA/Tabu
(which accept worsening moves), where picking the right operator by search stage
might matter more. Here we ablate context on/off (and exploration on/off) under SA
and Tabu, on the 20 TSPLIB instances, with linear gating.

Key contrasts: ctx vs noctx within each host, for each exploration setting,
Wilcoxon paired by instance. Tables -> results/tables/tsp_context_sa_tabu/.

    python scripts/run_context_sa_tabu.py
"""

import os

import pandas as pd
from scipy.stats import wilcoxon

from serene_mh.core import SimulatedAnnealing, Tabu, Roulette
from serene_mh.controller import SereneMH
from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
OUT = "results/tables/tsp_context_sa_tabu"
FIGURES = "results/figures"


def _serene(use_context, exploration):
    return lambda a: SereneMH(a, 3, 1, use_context=use_context, exploration=exploration)


HOSTS = {
    "SA": lambda: SimulatedAnnealing(max_evals=MAX_EVALS),
    "Tabu": lambda: Tabu(),
}
SELECTORS = {
    "ctx-expl": _serene(True, 0.3),
    "noctx-expl": _serene(False, 0.3),
    "ctx-greedy": _serene(True, 0.0),
    "noctx-greedy": _serene(False, 0.0),
    "roulette": lambda a: Roulette(a),
}
CONTRASTS = [("ctx-expl", "noctx-expl"), ("ctx-greedy", "noctx-greedy")]

CONFIGS = [
    Config(f"{s}|{h}", smake, hmake)
    for h, hmake in HOSTS.items() for s, smake in SELECTORS.items()
]


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"context @ SA/Tabu: {len(instances)} TSPLIB x {len(CONFIGS)} configs x {len(SEEDS)} seeds")

    study = run_study(instances, CONFIGS, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)
    df = enrich(study, optima)

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    rank_table(df, "auc").to_csv(os.path.join(OUT, "ranks_auc.csv"), index=False)

    rows = []
    for value in ("auc", "gap"):
        for host in HOSTS:
            sub = df[df.host == host]
            piv = sub.pivot_table(index="instance", columns="selector", values=value)
            for ctx, noctx in CONTRASTS:
                _, p = wilcoxon(piv[ctx], piv[noctx])
                winner = ctx if piv[ctx].mean() < piv[noctx].mean() else noctx
                rows.append({"host": host, "metric": value, "with_context": ctx, "no_context": noctx,
                             "winner": winner, "p": p,
                             "mean_ctx": piv[ctx].mean(), "mean_noctx": piv[noctx].mean()})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(os.path.join(OUT, "context_contrasts.csv"), index=False)
    print("wrote tables to", OUT)

    print("\nmean AUC by (host, selector):")
    print(df.groupby(["host", "selector"])["auc"].mean().unstack("selector").round(4).to_string())
    print("\ncontext contrasts (does context help under SA/Tabu?):")
    print(contrasts.round(4).to_string(index=False))

    os.makedirs(FIGURES, exist_ok=True)
    for host in HOSTS:
        order = [f"{s}|{host}" for s in ("ctx-greedy", "noctx-greedy", "ctx-expl", "noctx-expl", "roulette")]
        plot_regret_curves(study, order, os.path.join(FIGURES, f"tsp_context_{host.lower()}"),
                           references=optima, title=f"Context on/off under {host} (TSPLIB)")


if __name__ == "__main__":
    main()
