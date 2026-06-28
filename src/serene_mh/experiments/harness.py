"""Experiment harness: run many (instance x config x seed) searches and collect
the results in a tidy table plus per-run convergence curves.

A *config* is one thing we compare: an operator selector paired with an
acceptance criterion (e.g. "serene under SA"). Presets that build the full set of
base metaheuristics x selectors live in `presets.py`.

Fairness: for a given (instance, seed) every config is run from the *same* random
seed, so they all start from the same solution and face the same operator
randomness - the only thing that differs is the config itself.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core import build_actions, run_search


@dataclass
class Config:
    """One comparable configuration.

    `make_selector(actions)` builds a fresh selector for the given action list;
    `make_acceptance()` builds a fresh acceptance criterion. Both must return new
    objects each call (they carry per-run state).
    """

    name: str
    make_selector: Callable
    make_acceptance: Callable


def convergence_curve(history, max_evals) -> np.ndarray:
    """Best objective found so far as a function of evaluations spent.

    Returns an array of length `max_evals`; entry e is the best objective after
    e+1 evaluations. This puts every run on a common x-axis (evaluations), which
    is the budget that matters for sample efficiency.
    """
    curve = np.full(max_evals, np.nan)
    for row in history:
        e = row["n_evals"] - 1
        if 0 <= e < max_evals:
            curve[e] = row["best_obj"]
    # forward-fill: between recorded points the best-so-far is unchanged
    last = np.nan
    for e in range(max_evals):
        if np.isnan(curve[e]):
            curve[e] = last
        else:
            last = curve[e]
    # any leading gap (before the first record) takes the first known value
    first_valid = next((v for v in curve if not np.isnan(v)), np.nan)
    return np.where(np.isnan(curve), first_valid, curve)


@dataclass
class Study:
    """The collected results of a run_study call."""

    results: object              # pandas DataFrame: one row per (instance, config, seed)
    curves: dict                 # (instance, config, seed) -> convergence curve
    max_evals: int
    configs: list = field(default_factory=list)

    def save(self, out_dir):
        """Write the results table (CSV) and the curves (compressed npz)."""
        import os

        os.makedirs(out_dir, exist_ok=True)
        self.results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
        if self.curves:
            keys = ["||".join(map(str, k)) for k in self.curves]
            np.savez_compressed(
                os.path.join(out_dir, "curves.npz"),
                **{k: v for k, v in zip(keys, self.curves.values())},
            )
        return out_dir


def run_study(instances, configs, seeds, max_evals, references=None, record_curves=True):
    """Run every (instance, config, seed) combination and gather the results.

    Parameters
    ----------
    instances : dict[str, Problem] or Problem
        The instance(s) to run on.
    configs : list[Config]
        The configurations to compare.
    seeds : iterable[int]
        Random seeds (repetitions).
    max_evals : int
        Evaluation budget per run.
    references : dict[str, float] or None
        Optional best-known/optimal objective per instance, for a gap column.
    """
    import pandas as pd

    if not isinstance(instances, dict):
        instances = {getattr(instances, "name", "instance"): instances}
    references = references or {}

    rows = []
    curves = {}
    for inst_name, problem in instances.items():
        for cfg in configs:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                actions = build_actions(problem.operators())
                selector = cfg.make_selector(actions)
                acceptance = cfg.make_acceptance()
                result = run_search(
                    problem, selector, acceptance, max_evals, rng, record=record_curves
                )
                row = {
                    "instance": inst_name,
                    "config": cfg.name,
                    "seed": seed,
                    "best": result.best_objective,
                    "n_evals": result.n_evals,
                    "n_iterations": result.n_iterations,
                }
                ref = references.get(inst_name)
                if ref:
                    row["gap"] = (result.best_objective - ref) / ref
                rows.append(row)
                if record_curves:
                    curves[(inst_name, cfg.name, seed)] = convergence_curve(
                        result.history, max_evals
                    )

    return Study(results=pd.DataFrame(rows), curves=curves, max_evals=max_evals, configs=list(configs))
