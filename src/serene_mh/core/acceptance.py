"""Acceptance criteria.

Given the current solution and a candidate, an acceptance criterion decides
whether the search should move to the candidate. This single choice is what
turns the generic loop into Local Search, Simulated Annealing or Tabu Search.

All criteria assume minimisation (lower objective is better).

How the four "base metaheuristics" map onto this code:
  - LS   = Greedy acceptance
  - SA   = SimulatedAnnealing acceptance
  - Tabu = Tabu acceptance
  - ALNS = the Roulette selector (see selectors.py) + destroy/repair operators,
           usually run under RecordToRecord or SimulatedAnnealing acceptance.
"""

import math
from collections import deque


class AcceptanceCriterion:
    """Base class. Subclasses implement `accept`; some also use `step`."""

    name: str = "acceptance"

    def accept(self, state, candidate, rng) -> bool:
        raise NotImplementedError

    def step(self, state) -> None:
        """Called once per iteration, e.g. to cool the temperature."""

    def reset(self) -> None:
        """Restore any internal state to its starting point (new run)."""


class Greedy(AcceptanceCriterion):
    """Hill climbing / Local Search: accept only improving moves."""

    name = "LS"

    def __init__(self, accept_equal: bool = False):
        self.accept_equal = accept_equal

    def accept(self, state, candidate, rng) -> bool:
        if self.accept_equal:
            return candidate.objective <= state.incumbent.objective
        return candidate.objective < state.incumbent.objective


class SimulatedAnnealing(AcceptanceCriterion):
    """Accept improving moves always; accept a worsening move with probability
    exp(-delta / T). The temperature T cools geometrically each iteration, so
    worse moves are common early and rare late.

    Two design choices keep this usable across very different problems:

    * RELATIVE deltas. We divide the worsening `delta` by the incumbent's
      magnitude, so `t_start` means roughly "the fractional worsening tolerated
      early on" regardless of whether tour lengths are ~50 or ~50,000. (A fixed
      absolute temperature silently collapses SA into greedy on large-objective
      problems - the bug we hit on TSP.)

    * BUDGET-AWARE cooling. If you pass `max_evals`, the cooling rate is set so
      the temperature glides from `t_start` down to `t_min` over the whole run,
      instead of needing a hand-tuned per-iteration rate.
    """

    name = "SA"

    def __init__(
        self,
        t_start: float = 0.1,
        t_min: float = 1e-4,
        cooling: float | None = None,
        max_evals: int | None = None,
        relative: bool = True,
    ):
        self.t_start = t_start
        self.t_min = t_min
        self.relative = relative
        if cooling is None:
            # glide t_start -> t_min across the budget (approx. one eval per step)
            cooling = (t_min / t_start) ** (1.0 / max_evals) if max_evals else 0.999
        self.cooling = cooling
        self.t = t_start

    def reset(self) -> None:
        self.t = self.t_start

    def accept(self, state, candidate, rng) -> bool:
        delta = candidate.objective - state.incumbent.objective
        if delta <= 0:
            return True
        scale = max(abs(state.incumbent.objective), 1e-12) if self.relative else 1.0
        return rng.random() < math.exp(-(delta / scale) / max(self.t, self.t_min))

    def step(self, state) -> None:
        self.t = max(self.t * self.cooling, self.t_min)


class Tabu(AcceptanceCriterion):
    """Short-term memory + bounded worsening (Tabu adapted to sample-based search).

    Classic Tabu Search scans a whole neighbourhood and moves to the best
    non-tabu neighbour, which may be worse but is usually only slightly worse.
    Our loop samples only a few candidates per step, so "accept any non-tabu
    candidate, however bad" collapses into a random walk. We therefore keep the
    tabu memory (anti-cycling) and the aspiration rule, but only allow a
    *bounded* worsening move (within `max_worse`, relative to the incumbent) -
    enough to climb out of local optima without wandering off.

    `tenure` is how many recent solutions to forbid; `max_worse` is the largest
    fractional worsening accepted for a non-tabu, non-improving move.
    """

    name = "Tabu"

    def __init__(self, tenure: int = 20, max_worse: float = 0.05):
        self.tenure = tenure
        self.max_worse = max_worse
        self.recent = deque(maxlen=tenure)

    def reset(self) -> None:
        self.recent.clear()

    def accept(self, state, candidate, rng) -> bool:
        sig = state.problem.signature(candidate)
        aspiration = candidate.objective < state.best.objective
        if sig in self.recent and not aspiration:
            return False  # tabu and not good enough to override
        delta = candidate.objective - state.incumbent.objective
        if delta <= 0 or aspiration:
            self.recent.append(sig)
            return True
        # worsening, non-tabu: accept only within a bounded tolerance
        if delta <= self.max_worse * max(abs(state.incumbent.objective), 1e-12):
            self.recent.append(sig)
            return True
        return False


class RecordToRecord(AcceptanceCriterion):
    """Record-to-record travel: accept a candidate if it is not much worse than
    the best found so far (within `deviation` of it, relative to its magnitude).
    Simple and nearly parameter-free; a common acceptance rule for ALNS.
    """

    name = "RRT"

    def __init__(self, deviation: float = 0.02):
        self.deviation = deviation

    def accept(self, state, candidate, rng) -> bool:
        threshold = state.best.objective + self.deviation * abs(state.best.objective)
        return candidate.objective <= threshold
