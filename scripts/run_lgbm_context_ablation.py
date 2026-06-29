"""Does context help once the gate is a LightGBM (vs the linear gate, where it didn't)?

The earlier ablation found context added nothing - but that used the *linear*
gating score, which cannot exploit context x action interactions. A LightGBM gate
can, so we re-test context on/off under LightGBM gating, for both exploration
settings. Under the ALNS host on the 20 TSPLIB instances:

  lgbm-ctx-expl / lgbm-noctx-expl       LightGBM gating, context on/off, nu=0.3
  lgbm-ctx-greedy / lgbm-noctx-greedy   LightGBM gating, context on/off, nu=0
  linear-ctx-greedy                     linear gating reference (prior ablation winner)
  roulette                              reference

Writes tables to results/tables/tsp_lgbm_context/ and a figure. The key contrasts
are ctx vs noctx within each exploration setting (Wilcoxon, blocked by instance).

    python scripts/run_lgbm_context_ablation.py
"""

import os

import pandas as pd
from scipy.stats import wilcoxon

from serene_mh.core import RecordToRecord, Roulette
from serene_mh.controller import SereneMH
from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table, within_host_vs
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
OUT = "results/tables/tsp_lgbm_context"
FIGURES = "results/figures"


def _alns():
    return RecordToRecord(deviation=0.02)


def _serene(use_context, exploration, gating="lgbm"):
    return lambda a: SereneMH(a, slate_size=3, n_exec=1, gating=gating,
                              use_context=use_context, exploration=exploration)


CONFIGS = [
    Config("lgbm-ctx-expl|ALNS", _serene(True, 0.3), _alns),
    Config("lgbm-noctx-expl|ALNS", _serene(False, 0.3), _alns),
    Config("lgbm-ctx-greedy|ALNS", _serene(True, 0.0), _alns),
    Config("lgbm-noctx-greedy|ALNS", _serene(False, 0.0), _alns),
    Config("linear-ctx-greedy|ALNS", _serene(True, 0.0, gating="linear"), _alns),
    Config("roulette|ALNS", lambda a: Roulette(a), _alns),
]

CONTRASTS = [("lgbm-ctx-expl", "lgbm-noctx-expl"), ("lgbm-ctx-greedy", "lgbm-noctx-greedy")]


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"lgbm context ablation: {len(instances)} TSPLIB x {len(CONFIGS)} configs x {len(SEEDS)} seeds")

    study = run_study(instances, CONFIGS, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)
    df = enrich(study, optima)

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    for value in ("auc", "gap"):
        rank_table(df, value).to_csv(os.path.join(OUT, f"ranks_{value}.csv"), index=False)
        within_host_vs(df, value, reference="lgbm-ctx-greedy").to_csv(
            os.path.join(OUT, f"vs_lgbm_ctx_greedy_{value}.csv"), index=False)

    # explicit context-vs-no-context contrasts (paired by instance)
    rows = []
    for value in ("auc", "gap"):
        piv = df.pivot_table(index="instance", columns="selector", values=value)
        for ctx, noctx in CONTRASTS:
            _, p = wilcoxon(piv[ctx], piv[noctx])
            winner = ctx if piv[ctx].mean() < piv[noctx].mean() else noctx
            rows.append({"metric": value, "with_context": ctx, "no_context": noctx,
                         "winner": winner, "p": p,
                         "mean_ctx": piv[ctx].mean(), "mean_noctx": piv[noctx].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "context_contrasts.csv"), index=False)
    print("wrote tables to", OUT)

    print("\nmean AUC and gap by config (lower=better):")
    print(df.groupby("selector")[["auc", "gap"]].mean().sort_values("auc").round(4).to_string())
    print("\ncontext contrasts (does context help under LightGBM gating?):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    os.makedirs(FIGURES, exist_ok=True)
    order = ["lgbm-ctx-greedy|ALNS", "lgbm-noctx-greedy|ALNS", "lgbm-ctx-expl|ALNS",
             "lgbm-noctx-expl|ALNS", "linear-ctx-greedy|ALNS"]
    plot_regret_curves(study, order, os.path.join(FIGURES, "tsp_lgbm_context"),
                       references=optima, title="Context on/off under LightGBM gating (TSPLIB, ALNS)")


if __name__ == "__main__":
    main()
