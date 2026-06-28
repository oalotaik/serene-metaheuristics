"""Sample-efficiency metrics computed from convergence curves.

The budget is always measured in *evaluations* (calls to problem.evaluate), so
"time-to-target" means *evaluations-to-target*. These metrics capture how fast a
run improves, not just where it ends up - which is the whole point of SERENE-MH.

Per-curve metrics
-----------------
- quality_at_fractions : best objective reached after 25 / 50 / 75 / 100 % of the
  evaluation budget (final quality is the 100 % point).
- evals_to_target      : how many evaluations until the best-so-far reaches a
  target quality (lower = converges faster).
- regret_auc           : area under the normalised convergence curve in [0, 1]
  (lower = better and faster); rewards reaching good solutions early.

Study-level
-----------
- add_metrics : attach all of the above as columns on a Study's results table,
  using a consistent per-instance reference (the provided optimum, or the best
  objective found anywhere in the study for that instance).
"""

import numpy as np

DEFAULT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def quality_at_fractions(curve, fractions=DEFAULT_FRACTIONS):
    """Best objective reached after each given fraction of the budget."""
    n = len(curve)
    out = {}
    for f in fractions:
        idx = max(0, min(n - 1, round(f * n) - 1))
        out[f] = float(curve[idx])
    return out


def evals_to_target(curve, target):
    """Evaluations until best-so-far first reaches `target` (minimisation).

    Returns the 1-based evaluation count, or np.inf if the target is never met.
    """
    hits = np.where(curve <= target)[0]
    return int(hits[0] + 1) if len(hits) else np.inf


def regret_auc(curve, best_ref, worst_ref):
    """Mean normalised regret over the run, in [0, 1] (lower = better/faster).

    Each point is scaled to (curve - best_ref) / (worst_ref - best_ref) and
    clipped to [0, 1], then averaged over evaluations. A run that drops to good
    values early has a small area; a slow run has a large area.
    """
    span = max(worst_ref - best_ref, 1e-12)
    norm = np.clip((curve - best_ref) / span, 0.0, 1.0)
    return float(norm.mean())


def add_metrics(study, fractions=DEFAULT_FRACTIONS, target_tol=0.01, references=None):
    """Return a copy of study.results with sample-efficiency columns added.

    Columns: q25/q50/q75/q100 (quality at budget fractions), `auc` (regret area),
    `ttt` (evals-to-target; capped at max_evals+1 if never reached), `target`.

    `references` maps instance -> known optimum; when absent, the per-instance
    reference is the best objective found anywhere in the study for that instance.
    """
    import pandas as pd

    references = references or {}

    # per-instance best/worst references from the curves
    inst_best, inst_worst = {}, {}
    for (inst, _cfg, _seed), curve in study.curves.items():
        inst_best[inst] = min(inst_best.get(inst, np.inf), float(curve.min()))
        inst_worst[inst] = max(inst_worst.get(inst, -np.inf), float(curve[0]))

    never_reached = study.max_evals + 1
    rows = []
    for (inst, cfg, seed), curve in study.curves.items():
        best_ref = float(references.get(inst, inst_best[inst]))
        worst_ref = inst_worst[inst]
        target = best_ref * (1.0 + target_tol)
        ttt = evals_to_target(curve, target)
        row = {
            "instance": inst,
            "config": cfg,
            "seed": seed,
            "auc": regret_auc(curve, best_ref, worst_ref),
            "ttt": never_reached if np.isinf(ttt) else ttt,
            "target": target,
        }
        for f, v in quality_at_fractions(curve, fractions).items():
            row[f"q{int(round(f * 100))}"] = v
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    return study.results.merge(metrics_df, on=["instance", "config", "seed"], how="left")
