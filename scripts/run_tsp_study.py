"""Main Paper-1 TSP study: the full LS/SA/Tabu/ALNS x selector grid on the 20
symmetric TSPLIB instances, with true optima.

Writes per-run results and summary tables (ranks, within-host means, within-host
Wilcoxon) as CSVs under results/tables/tsp_main/, and a per-host crossover figure
under results/figures/. Reproducible:

    python scripts/run_tsp_study.py
"""

import os

from serene_mh.problems import load_benchmarks, load_optima, PAPER1_TSP
from serene_mh.experiments import run_study, standard_configs
from serene_mh.experiments.analysis import save_summary
from serene_mh.experiments.plots import plot_regret_curves

MAX_EVALS = 1500
SEEDS = list(range(5))
TABLES = "results/tables/tsp_main"
FIGURES = "results/figures"


def main():
    instances = load_benchmarks(PAPER1_TSP)
    optima = {k: v for k, v in load_optima().items() if k in instances}
    print(f"{len(instances)} TSPLIB instances x 20 configs x {len(SEEDS)} seeds, {MAX_EVALS} evals")

    configs = standard_configs(MAX_EVALS)
    study = run_study(instances, configs, seeds=SEEDS, max_evals=MAX_EVALS, references=optima)

    df = save_summary(study, optima, TABLES)
    print(f"wrote {len(df)} result rows + summary tables to {TABLES}/")

    os.makedirs(FIGURES, exist_ok=True)
    order = [f"{s}|ALNS" for s in ("serene", "roulette", "exp3", "ucb1", "uniform")]
    paths = plot_regret_curves(
        study, order, os.path.join(FIGURES, "tsp_main_alns_crossover"),
        references=optima, title="TSPLIB (20 symmetric), ALNS host",
    )
    print("wrote figure", paths)


if __name__ == "__main__":
    main()
