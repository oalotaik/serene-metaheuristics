"""Statistical comparison of configs across many (instance, seed) blocks.

This is the standard apparatus for comparing several algorithms over many
problems (Demsar, 2006):

  - build a *blocks x configs* table of results (one row per instance-seed,
    one column per config), where every config saw the same blocks;
  - average ranks per config;
  - a Friedman test for "are the configs different at all?";
  - the Nemenyi critical difference (CD) for "which pairs differ?";
  - and, as an alternative, pairwise Wilcoxon signed-rank tests with Holm
    correction.

Everything assumes *lower is better* by default (minimisation).
"""

import numpy as np


def block_matrix(study, value="best", aggregate=None):
    """Reshape a Study's results into a blocks x configs DataFrame.

    Each block is one (instance, seed) pair; each column is a config. With every
    config run on the same seeds, the blocks are paired - exactly what the
    Friedman / Wilcoxon tests need.
    """
    import pandas as pd

    df = study.results if hasattr(study, "results") else study
    table = df.pivot_table(index=["instance", "seed"], columns="config", values=value)
    if aggregate == "instance":
        table = table.groupby("instance").mean()
    return table.dropna()


def average_ranks(matrix, lower_is_better=True):
    """Average rank of each config across blocks (rank 1 = best in a block)."""
    ranks = matrix.rank(axis=1, ascending=lower_is_better)
    return ranks.mean(axis=0).sort_values()


def friedman_test(matrix):
    """Friedman test across configs. Returns (statistic, p_value).

    A small p-value means the configs are not all equivalent.
    """
    from scipy.stats import friedmanchisquare

    columns = [matrix[c].to_numpy() for c in matrix.columns]
    stat, p = friedmanchisquare(*columns)
    return stat, p


def nemenyi_cd(num_configs, num_blocks, alpha=0.05):
    """Critical difference for the Nemenyi post-hoc test.

    Two configs differ significantly if their average ranks differ by at least
    this much. The critical value comes from the studentized-range distribution,
    so any number of configs and any alpha are supported.
    """
    from scipy.stats import studentized_range

    k = num_configs
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return q * np.sqrt(k * (k + 1) / (6.0 * num_blocks))


def pairwise_wilcoxon(matrix, lower_is_better=True, alpha=0.05):
    """Pairwise Wilcoxon signed-rank tests with Holm-corrected p-values.

    Returns a DataFrame with one row per config pair: raw p, Holm-adjusted p,
    whether it is significant at `alpha`, and which config is better.
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    configs = list(matrix.columns)
    pairs = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            a, b = configs[i], configs[j]
            xa, xb = matrix[a].to_numpy(), matrix[b].to_numpy()
            if np.allclose(xa, xb):
                p = 1.0
            else:
                p = wilcoxon(xa, xb).pvalue
            better = a if (xa.mean() < xb.mean()) == lower_is_better else b
            pairs.append({"a": a, "b": b, "p": p, "better": better})

    pairs.sort(key=lambda r: r["p"])  # Holm: order by ascending p
    m = len(pairs)
    prev = 0.0
    for rank, row in enumerate(pairs):
        adj = min(1.0, (m - rank) * row["p"])
        adj = max(adj, prev)  # enforce monotonic non-decreasing adjusted p
        row["p_holm"] = adj
        row["significant"] = adj < alpha
        prev = adj
    return pd.DataFrame(pairs)
