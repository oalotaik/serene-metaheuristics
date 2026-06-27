"""Core data types: a Solution and the Problem base class.

A *problem* (TSP, CVRP, restoration, ...) only needs to know how to:
  1. build a starting solution,
  2. compute the objective value of a solution (the expensive step we count),
  3. list the operators (moves) that can be applied to its solutions.

Everything else - the search loop, operator selection, acceptance - is generic
and lives in the other core modules, so a new problem plugs in without touching
the engine.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    """One candidate solution.

    `data` is whatever representation the problem uses (e.g. a list of cities for
    TSP, or an arc-to-period assignment for restoration). `objective` is its cost,
    filled in when the solution is evaluated; it is None until then.

    Convention: we always *minimise* the objective. A maximisation problem should
    return a negated value from `evaluate`.
    """

    data: Any
    objective: float | None = None
    feasible: bool = True
    info: dict = field(default_factory=dict)


class Problem:
    """Base class for an optimisation problem. Subclasses override these methods."""

    name: str = "problem"

    def initial_solution(self, rng) -> Solution:
        """Return a starting Solution. `rng` is a numpy random Generator."""
        raise NotImplementedError

    def evaluate(self, solution: Solution) -> float:
        """Return the objective value of `solution` (lower is better).

        This is the 'expensive' computation whose calls the search budget counts,
        so keep it free of side effects.
        """
        raise NotImplementedError

    def operators(self) -> list:
        """Return the list of Operator objects usable on this problem."""
        raise NotImplementedError

    def signature(self, solution: Solution):
        """A hashable fingerprint of a solution, used by Tabu search to remember
        recently visited solutions.

        Default: the rounded objective (a weak but always-available fingerprint).
        Problems with a natural discrete representation should override this with
        something more precise (e.g. a tuple of the tour) for a stronger memory.
        """
        if solution.objective is None:
            return None
        return round(solution.objective, 6)
