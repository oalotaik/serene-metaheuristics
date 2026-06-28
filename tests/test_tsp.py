"""Tests for the TSP problem module: operators stay valid permutations, search
improves the tour, the controller runs on it, distances are correct, and the
TSPLIB parser reads a small instance."""

import numpy as np

from serene_mh.core import Greedy, UniformRandom, build_actions, run_search
from serene_mh.controller import SereneMH
from serene_mh.problems import TSP, TwoOpt, OrOpt, Swap, tour_length, random_euclidean_instance
from serene_mh.problems.tsplib import load_tsplib


def test_operators_keep_valid_permutation():
    rng = np.random.default_rng(0)
    problem = random_euclidean_instance(20, rng)
    sol = problem.initial_solution(rng)
    for op in [TwoOpt(), OrOpt(), Swap()]:
        for settings in op.param_settings():
            for _ in range(20):
                cand = op.apply(sol, rng, **settings)
                assert sorted(int(c) for c in cand.data) == list(range(problem.size))
                # operators must not mutate the input tour
                assert len(sol.data) == problem.size


def test_local_search_improves_tour():
    rng = np.random.default_rng(1)
    problem = random_euclidean_instance(40, rng)
    identity_len = tour_length(np.arange(problem.size), problem.D)
    selector = UniformRandom(build_actions(problem.operators()))
    result = run_search(problem, selector, Greedy(), max_evals=2000, rng=rng)
    assert result.n_evals == 2000
    assert 0 < result.best_objective < identity_len
    assert sorted(int(c) for c in result.best.data) == list(range(problem.size))


def test_serene_runs_on_tsp():
    rng = np.random.default_rng(2)
    problem = random_euclidean_instance(40, rng)
    identity_len = tour_length(np.arange(problem.size), problem.D)
    selector = SereneMH(build_actions(problem.operators()), slate_size=3, n_exec=1)
    result = run_search(problem, selector, Greedy(), max_evals=2000, rng=rng)
    assert result.best_objective < identity_len
    assert sorted(int(c) for c in result.best.data) == list(range(problem.size))


def test_tour_length_on_unit_square():
    # four corners of a 10x10 square; optimal cycle is the perimeter = 40
    coords = [(0, 0), (0, 10), (10, 10), (10, 0)]
    from serene_mh.problems import euclidean_matrix

    D = euclidean_matrix(coords)
    assert D[0, 1] == 10 and D[0, 3] == 10
    assert D[0, 2] == 14  # round(sqrt(200)) = 14
    assert tour_length([0, 1, 2, 3], D) == 40


def test_load_tsplib_euc2d(tmp_path):
    content = """NAME : square4
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 10
3 10 10
4 10 0
EOF
"""
    path = tmp_path / "square4.tsp"
    path.write_text(content)
    problem = load_tsplib(str(path))
    assert isinstance(problem, TSP)
    assert problem.size == 4
    assert problem.name == "square4"
    assert tour_length([0, 1, 2, 3], problem.D) == 40
