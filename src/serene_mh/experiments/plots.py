"""Plots for a Study.

The headline plot is the *learning curve*: best objective versus number of
evaluations, which is the axis sample efficiency is about. We also provide a
final-quality comparison across configs.

Uses a non-interactive backend so it works in scripts and tests without a display.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _curves_for(study, instance, config):
    """All seed curves for one (instance, config), stacked into an array."""
    rows = [v for (inst, cfg, seed), v in study.curves.items() if inst == instance and cfg == config]
    return np.vstack(rows) if rows else None


def plot_learning_curves(study, instance, out_path, configs=None, show_band=True):
    """Seed-averaged best-vs-evaluations curve per config, for one instance."""
    configs = configs or [c.name for c in study.configs]
    x = np.arange(1, study.max_evals + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for config in configs:
        stack = _curves_for(study, instance, config)
        if stack is None:
            continue
        mean = stack.mean(axis=0)
        ax.plot(x, mean, label=config, linewidth=1.5)
        if show_band and stack.shape[0] > 1:
            err = stack.std(axis=0)
            ax.fill_between(x, mean - err, mean + err, alpha=0.15)

    ax.set_xlabel("evaluations")
    ax.set_ylabel("best objective so far")
    ax.set_title(f"Learning curves - {instance}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_final_quality(study, out_path, value="best", lower_is_better=True):
    """Box plot of final quality per config.

    With several instances, each instance's results are normalised to the best
    config mean on that instance (ratio >= 1), so they can share one axis.
    """
    df = study.results.copy()
    instances = df["instance"].unique()
    if len(instances) > 1:
        def _norm(group):
            per_config = group.groupby("config")[value].transform("mean")
            best = group[value].min() if lower_is_better else group[value].max()
            group = group.copy()
            group["_plot"] = group[value] / best
            return group
        df = df.groupby("instance", group_keys=False).apply(_norm)
        ycol, ylabel = "_plot", "ratio to best config (per instance)"
    else:
        ycol, ylabel = value, value

    order = df.groupby("config")[ycol].mean().sort_values(ascending=lower_is_better).index
    data = [df[df["config"] == c][ycol].to_numpy() for c in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(list(order), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title("Final quality by config")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
