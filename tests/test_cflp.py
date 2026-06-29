"""Tests for the CFLP problem core: the transportation (min-cost-flow) eval is
correct on a hand-checkable instance, and a search runs and stays feasible."""

import numpy as np

from serene_mh.core import Solution, Greedy, UniformRandom, build_actions, run_search
from serene_mh.problems.cflp import CFLP, random_cflp_instance


def _cost(problem, mask):
    sol = Solution(data=np.array(mask, dtype=bool))
    return problem.evaluate(sol), sol.feasible


def test_eval_matches_hand_computation():
    # 2 facilities, 2 customers; demand 5 each; each facility can serve both.
    # f0 cheap for c0, f1 cheap for c1 (per-unit costs).
    p = CFLP(fixed_cost=[100, 100], capacity=[10, 10], demand=[5, 5],
             unit_cost=[[1, 10], [10, 1]])
    assert _cost(p, [1, 1]) == (210.0, True)   # 200 fixed + 5 + 5 served
    assert _cost(p, [1, 0]) == (155.0, True)   # 100 fixed + 5*1 + 5*10
    assert _cost(p, [0, 1]) == (155.0, True)
    cost, feasible = _cost(p, [0, 0])
    assert not feasible                        # no open capacity -> penalty


def test_search_runs_and_stays_feasible():
    rng = np.random.default_rng(0)
    inst = random_cflp_instance(10, 40, rng)
    start = inst.evaluate(inst.initial_solution(rng))
    selector = UniformRandom(build_actions(inst.operators()))
    result = run_search(inst, selector, Greedy(), max_evals=300, rng=rng)
    assert result.n_evals == 300
    assert result.best_objective <= start      # never worse than all-open
    assert result.best.feasible
