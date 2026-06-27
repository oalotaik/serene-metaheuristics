"""The generic single-solution metaheuristic loop.

`run_search` ties together a problem, an operator selector and an acceptance
criterion. It is deliberately problem-agnostic: every problem-specific detail
lives behind the Problem / Operator interfaces.

The budget is measured in *objective evaluations* (calls to problem.evaluate),
because being efficient in that count is the whole point of SERENE-MH.

The loop each iteration:
  1. ask the selector for an action and build the candidate it produces,
  2. evaluate the candidate (this is the expensive step we count),
  3. work out the reward the action earned,
  4. ask the acceptance criterion whether to move to the candidate,
  5. update the best-so-far,
  6. let the selector learn from the reward,
  7. record a row of history.
"""

from collections import deque
from dataclasses import dataclass, field

from .problem import Solution


@dataclass
class SearchState:
    """Everything the loop tracks about the current run.

    `telemetry()` returns a small snapshot of progress. Baseline selectors ignore
    it, but SERENE-MH will later use it as the context features for its policy.
    """

    problem: object
    incumbent: Solution
    best: Solution
    n_evals: int = 0
    max_evals: int = 0
    iteration: int = 0
    stagnation: int = 0  # iterations since `best` last improved
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=25))
    recent_accepts: deque = field(default_factory=lambda: deque(maxlen=25))

    def telemetry(self) -> dict:
        frac_budget = self.n_evals / self.max_evals if self.max_evals else 0.0
        recent_improvement = (
            sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
        )
        accept_rate = (
            sum(self.recent_accepts) / len(self.recent_accepts) if self.recent_accepts else 0.0
        )
        return {
            "frac_budget": frac_budget,
            "incumbent_obj": self.incumbent.objective,
            "best_obj": self.best.objective,
            "stagnation": self.stagnation,
            "recent_improvement": recent_improvement,
            "accept_rate": accept_rate,
        }


@dataclass
class SearchResult:
    """What a run returns."""

    best: Solution
    best_objective: float
    n_evals: int
    n_iterations: int
    history: list = field(default_factory=list)
    selector: str = ""
    acceptance: str = ""


def relative_improvement_reward(incumbent: Solution, candidate: Solution) -> float:
    """Reward in (roughly) [0, 1]: how much the candidate improved on the
    incumbent, relative to the incumbent's magnitude. Worsening moves earn 0.

    Using a *relative* improvement keeps rewards on a similar scale across very
    different problems and across the early/late stages of one run.
    """
    base = max(abs(incumbent.objective), 1e-12)
    gain = (incumbent.objective - candidate.objective) / base
    return max(0.0, gain)


def run_search(
    problem,
    selector,
    acceptance,
    max_evals,
    rng,
    reward_fn=relative_improvement_reward,
    record=True,
):
    """Run the metaheuristic until `max_evals` objective evaluations are spent.

    Returns a SearchResult with the best solution found and a per-iteration
    history (used later for convergence plots and learning curves).
    """
    # Start from an initial solution and evaluate it (this counts as one eval).
    incumbent = problem.initial_solution(rng)
    incumbent.objective = problem.evaluate(incumbent)
    state = SearchState(
        problem=problem, incumbent=incumbent, best=incumbent, n_evals=1, max_evals=max_evals
    )
    acceptance.reset()
    selector.reset()

    history = []
    while state.n_evals < max_evals:
        # 1. choose an action and build the candidate it produces
        index = selector.select(state, rng)
        action = selector.actions[index]
        candidate = action.operator.apply(state.incumbent, rng, **action.params)

        # 2. evaluate the candidate (the expensive, counted step)
        candidate.objective = problem.evaluate(candidate)
        state.n_evals += 1

        # 3. reward the action, measured against the incumbent it started from
        reward = reward_fn(state.incumbent, candidate)

        # 4. accept or reject
        accepted = acceptance.accept(state, candidate, rng)
        if accepted:
            state.incumbent = candidate

        # 5. track the best solution found so far
        improved_best = candidate.objective < state.best.objective
        if improved_best:
            state.best = candidate
            state.stagnation = 0
        else:
            state.stagnation += 1

        # 6. let the selector learn
        selector.update(index, reward, info={"accepted": accepted, "new_best": improved_best})

        # 7. bookkeeping
        state.recent_rewards.append(reward)
        state.recent_accepts.append(1 if accepted else 0)
        acceptance.step(state)
        state.iteration += 1
        if record:
            history.append(
                {
                    "iteration": state.iteration,
                    "n_evals": state.n_evals,
                    "action": action.label,
                    "reward": reward,
                    "accepted": accepted,
                    "incumbent_obj": state.incumbent.objective,
                    "best_obj": state.best.objective,
                }
            )

    return SearchResult(
        best=state.best,
        best_objective=state.best.objective,
        n_evals=state.n_evals,
        n_iterations=state.iteration,
        history=history,
        selector=selector.name,
        acceptance=acceptance.name,
    )
