"""Turn a Study into tidy results and summary tables, and write them to CSV.

Config names follow the "selector|host" convention from presets, so we can split
them into a `selector` and `host` column and do the within-host comparisons that
are the right way to judge SERENE-MH (an operator-selection layer is judged
against other selectors under the *same* host).
"""

import os

import pandas as pd
from scipy.stats import wilcoxon

from . import metrics, stats


def split_config(df):
    """Add `selector` and `host` columns from the "selector|host" config name."""
    parts = df["config"].str.split("|", expand=True)
    df = df.copy()
    df["selector"] = parts[0]
    df["host"] = parts[1] if parts.shape[1] > 1 else "host"
    return df


def enrich(study, optima=None):
    """Per-run results with sample-efficiency metrics, gap-to-optimum and config split."""
    df = metrics.add_metrics(study, references=optima)
    if optima:
        df["gap"] = df["best"] / df["instance"].map(optima) - 1.0
    return split_config(df)


def rank_table(df, value, aggregate="instance"):
    """Average ranks per config (blocked), plus the Friedman p and Nemenyi CD."""
    matrix = stats.block_matrix(df, value=value, aggregate=aggregate)
    ranks = stats.average_ranks(matrix)
    _, p = stats.friedman_test(matrix)
    cd = stats.nemenyi_cd(matrix.shape[1], matrix.shape[0])
    out = ranks.rename("avg_rank").reset_index()
    out["metric"] = value
    out["friedman_p"] = p
    out["nemenyi_cd"] = cd
    out["n_blocks"] = matrix.shape[0]
    return out


def within_host_vs(df, value, reference="serene"):
    """Wilcoxon of `reference` vs each other selector, within each host, blocked by
    instance (mean over seeds). Holm-corrected within each host. Lower value = better.
    """
    rows = []
    for host, sub in df.groupby("host"):
        pivot = sub.pivot_table(index="instance", columns="selector", values=value)
        if reference not in pivot.columns:
            continue
        others = [c for c in pivot.columns if c != reference]
        tests = []
        for other in others:
            _, p = wilcoxon(pivot[reference], pivot[other])
            winner = reference if pivot[reference].mean() < pivot[other].mean() else other
            tests.append((other, p, winner))
        tests.sort(key=lambda r: r[1])  # Holm
        m, prev = len(tests), 0.0
        for rank, (other, p, winner) in enumerate(tests):
            p_holm = max(min(1.0, (m - rank) * p), prev)
            prev = p_holm
            rows.append({
                "host": host, "metric": value, "reference": reference, "baseline": other,
                "winner": winner, "p": p, "p_holm": p_holm, "significant": p_holm < 0.05,
            })
    return pd.DataFrame(rows)


def save_summary(study, optima, out_dir, metrics_for_means=("gap", "auc", "q25", "ttt")):
    """Write the full results table and the summary tables to `out_dir` as CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    df = enrich(study, optima)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    rank_values = ["auc"] + (["gap"] if optima else [])
    ranks = pd.concat([rank_table(df, v) for v in rank_values], ignore_index=True)
    ranks.to_csv(os.path.join(out_dir, "ranks.csv"), index=False)

    cols = [c for c in metrics_for_means if c in df.columns]
    means = df.groupby(["host", "selector"])[cols].mean().reset_index()
    means.to_csv(os.path.join(out_dir, "within_host_means.csv"), index=False)

    wil = pd.concat([within_host_vs(df, v) for v in rank_values], ignore_index=True)
    wil.to_csv(os.path.join(out_dir, "within_host_wilcoxon.csv"), index=False)
    return df
