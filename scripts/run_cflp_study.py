"""Paper-1 CFLP study: the full LS/SA/Tabu/ALNS x selector grid, plus the
GreedyMean comparator, on OR-Library cap instances with exact optima.

CFLP is the testbed where the method should finally earn its keep: every
candidate re-solves the assignment transportation LP (~10 ms), so evaluation is
genuinely expensive and there is no cheap delta - exactly the regime the
surrogate-gated slate is built for, and the regime cheap-eval TSP could not
provide. THE question: does SERENE-MH (gating / context) beat the 12-line
`greedy-mean` here, and beat the AOS baselines within every host?

EFFICIENCY is the claim, so the headline is quality at TIGHT BUDGETS: gap-to-
optimum at 25 / 50 / 75 / 100 % of the evaluation budget (q25/q50/q75/q100), with
within-host significance (serene vs each baseline, blocked by instance) at each
cutoff, plus AUC for the overall convergence area. Final gap is host context, not
the verdict. Budget is deliberately tight (expensive evals = sample-efficiency
regime), not the 1500 used for cheap TSP.

Writes per-run results + the budget-fraction summary tables under
results/tables/cflp_*/.

    python scripts/run_cflp_study.py            # pilot: small tier (13x 16x50), 3 seeds, 300 evals
    python scripts/run_cflp_study.py --medium   # medium tier (12x 25x50), 3 seeds, 300 evals
    python scripts/run_cflp_study.py --full     # full: 37 instances, 5 seeds, 500 evals
"""

import os
import sys

import pandas as pd

from serene_mh.core import GreedyMean, Greedy, SimulatedAnnealing, Tabu, RecordToRecord
from serene_mh.problems import load_cflp_benchmarks, load_cflp_optima, CAP_INSTANCES
from serene_mh.experiments import Config, run_study, standard_configs
from serene_mh.experiments.analysis import enrich, rank_table, within_host_vs
from serene_mh.experiments.plots import plot_regret_curves

# the small (16x50) tier - the quick pilot subset
SMALL_TIER = ["cap41", "cap42", "cap43", "cap44", "cap51",
              "cap61", "cap62", "cap63", "cap64",
              "cap71", "cap72", "cap73", "cap74"]

# the medium (25x50) tier - one step up from the pilot
MEDIUM_TIER = ["cap81", "cap82", "cap83", "cap84",
               "cap91", "cap92", "cap93", "cap94",
               "cap101", "cap102", "cap103", "cap104"]

# the budget fractions that define "efficiency" for this project
CUTOFFS = ["q25", "q50", "q75", "q100"]


def hosts(max_evals, tabu_tenure=20, rrt_deviation=0.02):
    """The four acceptance hosts, matching presets.standard_configs."""
    return {
        "LS": lambda: Greedy(),
        "SA": lambda: SimulatedAnnealing(max_evals=max_evals),
        "Tabu": lambda: Tabu(tenure=tabu_tenure),
        "ALNS": lambda: RecordToRecord(deviation=rrt_deviation),
    }


def main(tier="small"):
    if tier == "full":
        names, seeds, max_evals, out = CAP_INSTANCES, list(range(5)), 500, "results/tables/cflp_main"
    elif tier == "medium":
        names, seeds, max_evals, out = MEDIUM_TIER, list(range(3)), 300, "results/tables/cflp_medium"
    else:
        names, seeds, max_evals, out = SMALL_TIER, list(range(3)), 300, "results/tables/cflp_pilot"

    instances = load_cflp_benchmarks(names)
    optima = {k: v for k, v in load_cflp_optima().items() if k in instances}

    # full host x selector grid (20) + GreedyMean under each host (4) = 24 configs
    configs = standard_configs(max_evals)
    for hname, make in hosts(max_evals).items():
        configs.append(Config(f"greedy-mean|{hname}", lambda a: GreedyMean(a, epsilon=0.0), make))

    print(f"CFLP {tier.upper()}: {len(instances)} instances x {len(configs)} configs "
          f"x {len(seeds)} seeds, {max_evals} evals  (eval = transportation LP)")

    study = run_study(instances, configs, seeds=seeds, max_evals=max_evals, references=optima)
    df = enrich(study, optima)
    # gap-to-optimum at each budget fraction = (best-so-far at that fraction)/optimum - 1
    opt = df["instance"].map(optima)
    for q in CUTOFFS:
        df[f"g{q[1:]}"] = df[q] / opt - 1.0

    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "results.csv"), index=False)

    # ranks (blocked by instance) for AUC and each budget-fraction quality
    pd.concat([rank_table(df, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "ranks.csv"), index=False)
    # within-host serene-vs-each-baseline significance at each budget fraction (+ AUC)
    pd.concat([within_host_vs(df, v) for v in ["auc"] + CUTOFFS], ignore_index=True).to_csv(
        os.path.join(out, "within_host_efficiency.csv"), index=False)
    # mean gap-to-optimum (%) at each budget fraction, by config
    gap_means = (df.groupby(["host", "selector"])[["g25", "g50", "g75", "g100", "auc"]].mean() *
                 pd.Series({"g25": 100, "g50": 100, "g75": 100, "g100": 100, "auc": 1}))
    gap_means.reset_index().to_csv(os.path.join(out, "gap_by_budget_fraction.csv"), index=False)
    print(f"wrote results + efficiency tables to {out}/")

    print("\nmean gap-to-optimum (%) at 25/50/75/100% of budget, by config (sorted by 25%):")
    show = df.groupby(["host", "selector"])[["g25", "g50", "g75", "g100"]].mean() * 100
    print(show.sort_values("g25").round(3).to_string())

    figures = "results/figures"
    os.makedirs(figures, exist_ok=True)
    order = [f"{s}|ALNS" for s in ("serene", "greedy-mean", "roulette", "exp3", "uniform")]
    label = {"full": "full", "medium": "25x50 tier", "small": "16x50 tier"}[tier]
    plot_regret_curves(study, order, os.path.join(figures, f"cflp_{tier}_alns_crossover"),
                       references=optima,
                       title=f"OR-Library CFLP ({label}), ALNS host")


if __name__ == "__main__":
    args = sys.argv[1:]
    selected = "full" if "--full" in args else "medium" if "--medium" in args else "small"
    main(tier=selected)
