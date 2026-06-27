"""The SERENE-MH operator-selection controller.

`SereneMH` is a drop-in replacement for the baseline selectors in
serene_mh.core.selectors: it has the same `select` / `update` interface and so
runs inside the same `run_search` loop. The difference is that it chooses
operators with a *contextual* policy that learns online from search telemetry.
"""

from .serene import SereneMH, average_priors

__all__ = ["SereneMH", "average_priors"]
