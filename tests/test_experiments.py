"""Tests for the experiment harness: a small study runs end to end, convergence
curves are well-formed, the statistics return sane shapes, and the plots write
files."""

import numpy as np
import pytest

from serene_mh.problems import random_euclidean_instance
from serene_mh.experiments import run_study, standard_configs, convergence_curve
from serene_mh.experiments import stats, plots, metrics

MAX_EVALS = 300


def _instances():
    return {
        "rand-A": random_euclidean_instance(30, np.random.default_rng(10)),
        "rand-B": random_euclidean_instance(30, np.random.default_rng(11)),
    }


@pytest.fixture(scope="module")
def study():
    configs = standard_configs(MAX_EVALS, hosts=["LS", "SA"], selectors=["uniform", "serene"])
    return run_study(_instances(), configs, seeds=[0, 1, 2], max_evals=MAX_EVALS)


def test_convergence_curve_is_monotone_nonincreasing():
    history = [
        {"n_evals": 2, "best_obj": 100.0},
        {"n_evals": 5, "best_obj": 80.0},
        {"n_evals": 9, "best_obj": 80.0},
    ]
    curve = convergence_curve(history, max_evals=10)
    assert curve.shape == (10,)
    assert np.all(np.diff(curve) <= 1e-9)  # best-so-far never increases
    assert curve[-1] == 80.0


def test_study_runs_and_records(study):
    # 2 instances x 4 configs x 3 seeds
    assert len(study.results) == 2 * 4 * 3
    assert set(study.results.columns) >= {"instance", "config", "seed", "best"}
    for curve in study.curves.values():
        assert curve.shape == (MAX_EVALS,)
        assert np.all(np.diff(curve) <= 1e-9)


def test_statistics_shapes(study):
    matrix = stats.block_matrix(study)
    assert matrix.shape[0] == 2 * 3  # (instance, seed) blocks
    assert matrix.shape[1] == 4      # configs

    ranks = stats.average_ranks(matrix)
    assert len(ranks) == 4
    assert abs(ranks.mean() - (4 + 1) / 2) < 1e-6  # ranks average to (k+1)/2

    stat, p = stats.friedman_test(matrix)
    assert 0.0 <= p <= 1.0

    cd = stats.nemenyi_cd(num_configs=4, num_blocks=matrix.shape[0])
    assert cd > 0

    table = stats.pairwise_wilcoxon(matrix)
    assert {"a", "b", "p", "p_holm", "significant"} <= set(table.columns)


def test_plots_write_files(study, tmp_path):
    lc = plots.plot_learning_curves(study, "rand-A", tmp_path / "lc.png")
    fq = plots.plot_final_quality(study, tmp_path / "fq.png")
    assert (tmp_path / "lc.png").exists()
    assert (tmp_path / "fq.png").exists()


def test_per_curve_metrics():
    # a curve that drops from 100 to 10 over 100 evaluations
    curve = np.linspace(100.0, 10.0, 100)
    q = metrics.quality_at_fractions(curve)
    assert q[1.0] == 10.0                # final = best at 100%
    assert q[0.25] > q[0.5] > q[0.75] > q[1.0]  # improves over the budget
    # evals-to-target: first point at or below 55 is around the midpoint
    assert metrics.evals_to_target(curve, target=55.0) == 51
    assert np.isinf(metrics.evals_to_target(curve, target=5.0))  # never reached
    # a faster-dropping curve has smaller regret area
    fast = np.concatenate([np.full(10, 100.0), np.full(90, 10.0)])
    slow = np.concatenate([np.full(90, 100.0), np.full(10, 10.0)])
    assert metrics.regret_auc(fast, 10.0, 100.0) < metrics.regret_auc(slow, 10.0, 100.0)


def test_add_metrics_columns(study):
    enriched = metrics.add_metrics(study)
    for col in ("q25", "q50", "q75", "q100", "auc", "ttt", "target"):
        assert col in enriched.columns
    assert len(enriched) == len(study.results)
    # final-budget quality must match the recorded best
    assert np.allclose(enriched["q100"], enriched["best"])


def test_average_ranks_orders_dominant_config_first():
    # synthetic blocks x configs where "good" is always best (lowest)
    import pandas as pd

    matrix = pd.DataFrame(
        {"good": [1.0, 2.0, 1.5, 1.0], "mid": [3.0, 3.0, 2.5, 2.0], "bad": [5.0, 4.0, 6.0, 5.0]}
    )
    ranks = stats.average_ranks(matrix, lower_is_better=True)
    assert ranks.index[0] == "good"
    assert ranks["good"] < ranks["mid"] < ranks["bad"]
