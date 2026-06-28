"""Publication-ready plots for a Study.

All figures use a consistent, journal-friendly style (serif fonts, colourblind-
safe palette, distinct markers so lines stay readable in black-and-white,
despined axes, light grid) and are saved as both a vector PDF (for the paper)
and a 300-dpi PNG (for quick viewing).

Main figures
------------
- plot_regret_curves : normalised regret vs evaluations, aggregated across
  instances and seeds - the sample-efficiency / crossover figure.
- plot_learning_curves : raw best objective vs evaluations for a single instance.
- plot_final_quality : final-quality comparison across configs.

Uses a non-interactive backend so it works in scripts and tests without a display.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colourblind-safe palette + matching markers.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

PUB_STYLE = {
    "figure.figsize": (6.5, 4.0),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
}


def _pub_context():
    return plt.rc_context(PUB_STYLE)


def _save(fig, out_path):
    """Save as both PDF (vector) and PNG (300 dpi), regardless of the given ext."""
    stem, _ = os.path.splitext(str(out_path))
    paths = []
    for ext in (".pdf", ".png"):
        p = stem + ext
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _instance_refs(study, references):
    """Per-instance best-found and worst-start, for normalising across instances."""
    references = references or {}
    best, worst = {}, {}
    for (inst, _c, _s), curve in study.curves.items():
        best[inst] = min(best.get(inst, np.inf), float(curve.min()))
        worst[inst] = max(worst.get(inst, -np.inf), float(curve[0]))
    for inst in best:
        if inst in references:
            best[inst] = float(references[inst])
    return best, worst


def plot_regret_curves(study, configs, out_path, references=None, title=None, logx=False):
    """Normalised regret to best-found vs evaluations, averaged over instances+seeds.

    Regret is (best-so-far - reference) / (worst-start - reference) per instance,
    so curves from different-sized instances are comparable on one axis. The shaded
    band is +/- one standard error across runs. This is the figure that shows the
    sample-efficiency crossover.
    """
    best, worst = _instance_refs(study, references)
    x = np.arange(1, study.max_evals + 1)

    with _pub_context():
        fig, ax = plt.subplots()
        for i, config in enumerate(configs):
            stack = []
            for (inst, cfg, _s), curve in study.curves.items():
                if cfg != config:
                    continue
                span = max(worst[inst] - best[inst], 1e-12)
                stack.append((curve - best[inst]) / span)
            if not stack:
                continue
            stack = np.vstack(stack)
            mean = stack.mean(axis=0)
            se = stack.std(axis=0) / np.sqrt(stack.shape[0])
            color = PALETTE[i % len(PALETTE)]
            ax.plot(
                x, mean, label=config, color=color,
                marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(x) // 10), markersize=5,
            )
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Normalised regret to best-found")
        ax.set_ylim(bottom=0)
        if title:
            ax.set_title(title)
        ax.legend(ncol=2)
        return _save(fig, out_path)


def plot_learning_curves(study, instance, out_path, configs=None, show_band=True, title=None):
    """Seed-averaged best-vs-evaluations curve per config, for one instance."""
    configs = configs or [c.name for c in study.configs]
    x = np.arange(1, study.max_evals + 1)

    with _pub_context():
        fig, ax = plt.subplots()
        for i, config in enumerate(configs):
            rows = [v for (inst, cfg, _s), v in study.curves.items()
                    if inst == instance and cfg == config]
            if not rows:
                continue
            stack = np.vstack(rows)
            mean = stack.mean(axis=0)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(
                x, mean, label=config, color=color,
                marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(x) // 10), markersize=5,
            )
            if show_band and stack.shape[0] > 1:
                se = stack.std(axis=0) / np.sqrt(stack.shape[0])
                ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best objective so far")
        ax.set_title(title or f"Learning curves - {instance}")
        ax.legend(ncol=2)
        return _save(fig, out_path)


def plot_final_quality(study, out_path, value="best", lower_is_better=True, title=None):
    """Box plot of final quality per config.

    With several instances, each instance's results are normalised to the best
    config mean on that instance (ratio >= 1) so they can share one axis.
    """
    df = study.results.copy()
    instances = df["instance"].unique()
    if len(instances) > 1:
        def _norm(group):
            best = group[value].min() if lower_is_better else group[value].max()
            group = group.copy()
            group["_plot"] = group[value] / best
            return group
        df = df.groupby("instance", group_keys=False).apply(_norm)
        ycol, ylabel = "_plot", "Ratio to best config (per instance)"
    else:
        ycol, ylabel = value, value.capitalize()

    order = df.groupby("config")[ycol].mean().sort_values(ascending=lower_is_better).index
    data = [df[df["config"] == c][ycol].to_numpy() for c in order]

    with _pub_context():
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(list(order), rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        return _save(fig, out_path)
