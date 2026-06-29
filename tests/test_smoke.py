"""Smoke test for the core engine.

We define a tiny throwaway problem (OneMax) and run the search with every
acceptance criterion and every baseline selector, checking that the search
actually improves the objective and respects the evaluation budget.

OneMax is only a sanity vehicle - the real problems (TSP, CVRP, restoration)
arrive in later phases. Run with:  python -m pytest -q   (or  python tests/test_smoke.py)
"""

import numpy as np

from serene_mh.core import (
    Solution,
    Problem,
    Operator,
    build_actions,
    Greedy,
    SimulatedAnnealing,
    Tabu,
    RecordToRecord,
    UniformRandom,
    Roulette,
    UCB1,
    EXP3,
    GreedyMean,
    run_search,
)

LENGTH = 50  # length of the binary vector


class OneMax(Problem):
    """Minimise the number of zeros in a length-LENGTH binary vector (optimum 0)."""

    name = "onemax"

    def initial_solution(self, rng):
        bits = rng.integers(0, 2, size=LENGTH)
        return Solution(data=bits)

    def evaluate(self, solution):
        return float(LENGTH - int(solution.data.sum()))

    def operators(self):
        return [FlipBits()]

    def signature(self, solution):
        return tuple(int(b) for b in solution.data)


class FlipBits(Operator):
    """Flip k randomly chosen bits."""

    name = "flip"

    def param_settings(self):
        return [{"k": 1}, {"k": 2}, {"k": 3}]

    def apply(self, solution, rng, k=1):
        bits = solution.data.copy()
        idx = rng.choice(len(bits), size=k, replace=False)
        bits[idx] = 1 - bits[idx]
        return Solution(data=bits)


SELECTORS = [UniformRandom, Roulette, UCB1, EXP3, GreedyMean]


def _make_acceptances():
    return [
        Greedy(),
        SimulatedAnnealing(t_start=2.0, cooling=0.99),
        Tabu(tenure=10),
        RecordToRecord(deviation=0.1),
    ]


def test_runs_improve_and_respect_budget():
    problem = OneMax()
    for selector_cls in SELECTORS:
        for acceptance in _make_acceptances():
            rng = np.random.default_rng(0)
            selector = selector_cls(build_actions(problem.operators()))
            result = run_search(problem, selector, acceptance, max_evals=500, rng=rng)

            # budget is respected
            assert result.n_evals == 500, (selector.name, acceptance.name, result.n_evals)
            # a random start has ~25 zeros; a working search should get well below that
            assert result.best_objective < LENGTH / 2, (selector.name, acceptance.name, result.best_objective)
            # history has one row per iteration
            assert len(result.history) == result.n_iterations


if __name__ == "__main__":
    test_runs_improve_and_respect_budget()
    print("smoke test passed")
