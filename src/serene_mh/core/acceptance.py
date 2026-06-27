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
    exp(-delta / T), where delta is how much worse it is. The temperature T
    cools geometrically (T <- T * cooling) each iteration, so worse moves are
    common early and rare late.
    """

    name = "SA"

    def __init__(self, t_start: float = 1.0, cooling: float = 0.999, t_min: float = 1e-6):
        self.t_start = t_start
        self.cooling = cooling
        self.t_min = t_min
        self.t = t_start

    def reset(self) -> None:
        self.t = self.t_start

    def accept(self, state, candidate, rng) -> bool:
        delta = candidate.objective - state.incumbent.objective
        if delta <= 0:
            return True
        return rng.random() < math.exp(-delta / max(self.t, self.t_min))

    def step(self, state) -> None:
        self.t = max(self.t * self.cooling, self.t_min)


class Tabu(AcceptanceCriterion):
    """Accept the candidate unless we visited it very recently (it is 'tabu').
    Always accept if it is the best solution found so far (the 'aspiration' rule).

    This lets the search move to worse solutions to escape local optima while
    avoiding immediate cycling. `tenure` is how many recent solutions to forbid.
    """

    name = "Tabu"

    def __init__(self, tenure: int = 15):
        self.tenure = tenure
        self.recent = deque(maxlen=tenure)

    def reset(self) -> None:
        self.recent.clear()

    def accept(self, state, candidate, rng) -> bool:
        sig = state.problem.signature(candidate)
        is_tabu = sig in self.recent
        aspiration = candidate.objective < state.best.objective
        if is_tabu and not aspiration:
            return False
        self.recent.append(sig)
        return True


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
