"""Tests for the CFLP problem core: the transportation (min-cost-flow) eval is
correct on a hand-checkable instance, a search runs and stays feasible, the
operator portfolio produces valid solutions, and the OR-Library parser + exact
solver agree with the evaluator."""

import numpy as np

from serene_mh.core import Solution, Greedy, UniformRandom, build_actions, run_search
from serene_mh.problems.cflp import CFLP, random_cflp_instance
from serene_mh.problems.cflp_benchmarks import parse_cflp, solve_cflp_exact

# A tiny OR-Library-format instance == the hand instance below:
# 2 facilities (cap 10, fixed 100), 2 customers (demand 5), serve-all costs chosen
# so unit_cost = [[1, 10], [10, 1]].
_TINY_CAP = """2 2
10 100.
10 100.
5
5 50
5
50 5
"""


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


def test_operators_produce_valid_solutions():
    # The whole portfolio: every operator returns a boolean mask of the right
    # length, never mutates its input, and never closes every facility.
    rng = np.random.default_rng(1)
    inst = random_cflp_instance(8, 30, rng)
    actions = build_actions(inst.operators())
    assert len(actions) == 10  # 6 single-setting ops + multiflip(2 settings) + destroy_repair(2 settings)
    sol = inst.initial_solution(rng)
    for action in actions:
        before = sol.data.copy()
        cand = action.operator.apply(sol, rng, **action.params)
        assert cand.data.dtype == bool
        assert len(cand.data) == inst.m
        assert cand.data.any()                 # at least one facility open
        assert np.array_equal(sol.data, before)  # input untouched


def test_greedy_operators_are_structure_aware():
    # GreedyOpen on a 1-facility solution should open the facility that best serves
    # the customers, not a random one. Customer 0 is cheap from f1, customer 1 from
    # f2; starting with only f0 open, the higher-demand customer should drive choice.
    p = CFLP(fixed_cost=[10, 10, 10], capacity=[100, 100, 100], demand=[1, 9],
             unit_cost=[[5, 5], [1, 9], [9, 1]])  # f1 cheap for c0, f2 cheap for c1
    from serene_mh.problems.cflp import GreedyOpen
    sol = Solution(data=np.array([True, False, False]))
    cand = GreedyOpen(p.demand, p.unit_cost).apply(sol, np.random.default_rng(0))
    # c1 has demand 9 and is much cheaper from f2 -> opening f2 gives the most gain
    assert cand.data[2] and not cand.data[1]


def test_orlib_parser_matches_hand_instance():
    p = parse_cflp(_TINY_CAP, name="tiny")
    assert p.m == 2 and p.n == 2
    assert np.allclose(p.unit_cost, [[1, 10], [10, 1]])
    assert np.array_equal(p.capacity, [10, 10])
    assert np.array_equal(p.demand, [5, 5])
    # evaluator on the parsed instance reproduces the hand-checked costs
    both = Solution(data=np.array([True, True]))
    assert p.evaluate(both) == 210.0


def test_exact_solver_matches_evaluator():
    p = parse_cflp(_TINY_CAP, name="tiny")
    opt, mask, status = solve_cflp_exact(p)
    assert status == "Optimal"
    assert opt == 155.0                          # one facility open is optimal
    # the exact solver's open-set, scored by the flow evaluator, equals the optimum
    assert p.evaluate(Solution(data=mask)) == opt
