"""CFLP ablation: which parts of SERENE-MH drive the medium-tier win?

On TSP, none of the "fancy" parts earned their keep - context added nothing,
Thompson exploration HURT, and the whole win was the surrogate-gated greedy slate
(Sessions 15-17). CFLP is the first testbed where the full contextual-TS
controller beats the AOS baselines (Session 20). This ablation asks whether that
win is again just the gated slate, or whether context / exploration finally pay.

Under each exploratory host (ALNS / SA / Tabu - the ones where serene wins; LS is
the known null host), compare the full controller against one-component-removed
variants, plus the roulette / uniform / greedy-mean anchors:

  serene-full     context + gated slate (3->1) + exploration (nu=0.3)   the real thing
  serene-noctx    context OFF                                           does context matter?
  serene-nogate   gated slate OFF (slate_size = n_exec = 1)             does gating matter?
  serene-noexpl   exploration OFF (nu = 0)                              does TS help or hurt?
  serene-plainTS  context OFF and gating OFF                            strip to a plain TS bandit
  roulette        ALNS-native adaptive selection                       reference
  uniform         random operator choice                               control
  greedy-mean     raw running-mean greedy                              the TSP-killer comparator

Everything is compared to serene-full, within each host, blocked by instance, at
the 25/50/75/100% budget fractions (the efficiency cutoffs) and AUC.

    python scripts/run_cflp_ablations.py           # medium tier (12x 25x50), 3 seeds, 300 evals
    python scripts/run_cflp_ablations.py --full     # all 37 instances, 5 seeds, 500 evals

The --full run mirrors the main study's budget (run_cflp_study.py --full) and is
the one that locks the context/gating significance the n=12 medium run left
directional. It is a ~6 h job - intended for an overnight run. CSVs ->
results/tables/cflp_{medium,full}_ablations/.
"""

import os
import sys

import pandas as pd

from serene_mh.core import (GreedyMean, Roulette, UniformRandom,
                            Greedy, SimulatedAnnealing, Tabu, RecordToRecord)
from serene_mh.controller import SereneMH
from serene_mh.problems import load_cflp_benchmarks, load_cflp_optima, CAP_INSTANCES
from serene_mh.experiments import Config, run_study
from serene_mh.experiments.analysis import enrich, rank_table, within_host_vs

MEDIUM_TIER = ["cap81", "cap82", "cap83", "cap84",
               "cap91", "cap92", "cap93", "cap94",
               "cap101", "cap102", "cap103", "cap104"]
CUTOFFS = ["q25", "q50", "q75", "q100"]

# one-component-removed variants of SERENE-MH (kwargs to SereneMH)
VARIANTS = {
    "serene-full":    dict(slate_size=3, n_exec=1, use_context=True,  exploration=0.3),
    "serene-noctx":   dict(slate_size=3, n_exec=1, use_context=False, exploration=0.3),
    "serene-nogate":  dict(slate_size=1, n_exec=1, use_context=True,  exploration=0.3),
    "serene-noexpl":  dict(slate_size=3, n_exec=1, use_context=True,  exploration=0.0),
    "serene-plainTS": dict(slate_size=1, n_exec=1, use_context=False, exploration=0.3),
}


def hosts(max_evals):
    return {
        "ALNS": lambda: RecordToRecord(deviation=0.02),
        "SA": lambda: SimulatedAnnealing(max_evals=max_evals),
        "Tabu": lambda: Tabu(tenure=20),
    }


def build_configs(max_evals):
    configs = []
    for hname, make_host in hosts(max_evals).items():
        for vname, kw in VARIANTS.items():
            configs.append(Config(f"{vname}|{hname}",
                                  lambda a, kw=kw: SereneMH(a, **kw), make_host))
        configs.append(Config(f"roulette|{hname}", lambda a: Roulette(a), make_host))
        configs.append(Config(f"uniform|{hname}", lambda a: UniformRandom(a), make_host))
        configs.append(Config(f"greedy-mean|{hname}", lambda a: GreedyMean(a, epsilon=0.0), make_host))
    return configs


def main(full=False):
    if full:
        names, seeds, max_evals, out = CAP_INSTANCES, list(range(5)), 500, "results/tables/cflp_full_ablations"
    else:
        names, seeds, max_evals, out = MEDIUM_TIER, list(range(3)), 300, "results/tables/cflp_medium_ablations"

    instances = load_cflp_benchmarks(names)
    optima = {k: v for k, v in load_cflp_optima().items() if k in instances}
    configs = build_configs(max_evals)
    print(f"CFLP ablation ({'FULL 37' if full else 'medium 25x50'}): {len(instances)} instances x "
          f"{len(configs)} configs x {len(seeds)} seeds, {max_evals} evals")

    study = run_study(instances, configs, seeds=seeds, max_evals=max_evals, references=optima)
    df = enrich(study, optima)
    opt = df["instance"].map(optima)
    for q in CUTOFFS:
        df[f"g{q[1:]}"] = df[q] / opt - 1.0

    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    pd.concat([rank_table(df, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "ranks.csv"), index=False)
    # each variant/baseline vs serene-full, within host, at each cutoff + AUC
    pd.concat([within_host_vs(df, v, reference="serene-full") for v in ["auc"] + CUTOFFS],
              ignore_index=True).to_csv(os.path.join(out, "vs_full.csv"), index=False)
    gap_means = df.groupby(["host", "selector"])[["g25", "g50", "g75", "g100", "auc"]].mean()
    gap_means.reset_index().to_csv(os.path.join(out, "gap_by_budget_fraction.csv"), index=False)
    print(f"wrote ablation tables to {out}/")

    print("\nmean gap-to-optimum (%) at 25/50/75/100% + AUC, by host/variant (sorted by AUC within host):")
    show = df.groupby(["host", "selector"])[["g25", "g50", "g75", "g100", "auc"]].mean()
    show[["g25", "g50", "g75", "g100"]] *= 100
    for host in ["ALNS", "SA", "Tabu"]:
        print(f"\n{host}:")
        print(show.loc[host].sort_values("auc").round(3).to_string())


if __name__ == "__main__":
    main(full="--full" in sys.argv[1:])
