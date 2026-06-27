"""Tests for the SERENE-MH controller.

These check the controller *works and is wired correctly* (runs, learns, respects
the evaluation budget, supports the slate/gating mechanics and prior warm-start).
They are not performance claims - the comparison against baselines belongs to the
Track B / Track C experiments on real problems.
"""

import numpy as np

from serene_mh.core import Solution, Problem, Operator, build_actions, Greedy, run_search
from serene_mh.controller import SereneMH, average_priors

LENGTH = 50


class OneMax(Problem):
    """Minimise the number of zeros in a length-LENGTH binary vector (optimum 0)."""

    name = "onemax"

    def initial_solution(self, rng):
        return Solution(data=rng.integers(0, 2, size=LENGTH))

    def evaluate(self, solution):
        return float(LENGTH - int(solution.data.sum()))

    def operators(self):
        return [FlipBits()]


class FlipBits(Operator):
    name = "flip"

    def param_settings(self):
        return [{"k": 1}, {"k": 2}, {"k": 3}]

    def apply(self, solution, rng, k=1):
        bits = solution.data.copy()
        idx = rng.choice(len(bits), size=k, replace=False)
        bits[idx] = 1 - bits[idx]
        return Solution(data=bits)


def _run(slate_size, n_exec, max_evals=500, seed=0, prior_mean=None):
    problem = OneMax()
    selector = SereneMH(
        build_actions(problem.operators()),
        slate_size=slate_size,
        n_exec=n_exec,
        prior_mean=prior_mean,
    )
    rng = np.random.default_rng(seed)
    result = run_search(problem, selector, Greedy(), max_evals=max_evals, rng=rng)
    return selector, result


def test_plain_contextual_bandit_improves():
    selector, result = _run(slate_size=1, n_exec=1)
    assert result.n_evals == 500
    assert len(result.history) == result.n_iterations
    assert result.best_objective < LENGTH / 3  # clearly improved from a ~25 start


def test_surrogate_gating_runs_within_budget():
    # propose 3, evaluate only the best-looking 1 -> still one eval per iteration
    selector, result = _run(slate_size=3, n_exec=1)
    assert result.n_evals == 500
    assert result.best_objective < LENGTH / 3


def test_multi_exec_budget_accounting():
    # evaluate 2 per iteration -> about half as many iterations, budget still exact
    selector, result = _run(slate_size=3, n_exec=2)
    assert result.n_evals == 500
    assert len(result.history) == result.n_iterations
    assert result.n_iterations < 300  # ~2 evals per iteration


def test_export_and_warmstart():
    selector, _ = _run(slate_size=1, n_exec=1)
    mean = selector.export_mean()
    assert mean.shape == (selector.d,)

    # averaging two priors keeps the shape; warm-starting a fresh run works
    prior = average_priors([mean, mean])
    assert prior.shape == (selector.d,)
    _, result = _run(slate_size=1, n_exec=1, seed=1, prior_mean=prior)
    assert result.best_objective < LENGTH / 3
