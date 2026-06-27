"""Problem-agnostic search machinery for SERENE-MH.

Convenience re-exports so callers can do e.g.

    from serene_mh.core import run_search, SimulatedAnnealing, Roulette
"""

from .problem import Solution, Problem
from .operators import Operator, Action, build_actions
from .acceptance import (
    AcceptanceCriterion,
    Greedy,
    SimulatedAnnealing,
    Tabu,
    RecordToRecord,
)
from .selectors import (
    OperatorSelector,
    UniformRandom,
    Roulette,
    UCB1,
    EXP3,
)
from .engine import (
    SearchState,
    SearchResult,
    run_search,
    relative_improvement_reward,
)

__all__ = [
    "Solution",
    "Problem",
    "Operator",
    "Action",
    "build_actions",
    "AcceptanceCriterion",
    "Greedy",
    "SimulatedAnnealing",
    "Tabu",
    "RecordToRecord",
    "OperatorSelector",
    "UniformRandom",
    "Roulette",
    "UCB1",
    "EXP3",
    "SearchState",
    "SearchResult",
    "run_search",
    "relative_improvement_reward",
]
