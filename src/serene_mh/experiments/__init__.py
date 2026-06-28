"""Experiment harness, statistics and plots for comparing operator selectors
across the base metaheuristics.
"""

from .harness import Config, Study, run_study, convergence_curve
from .presets import standard_configs
from . import stats
from . import plots
from . import metrics

__all__ = [
    "Config",
    "Study",
    "run_study",
    "convergence_curve",
    "standard_configs",
    "stats",
    "plots",
    "metrics",
]
