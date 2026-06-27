"""The generic single-solution metaheuristic loop.

`run_search` ties together a problem, an operator selector and an acceptance
criterion. It is deliberately problem-agnostic: every problem-specific detail
lives behind the Problem / Operator interfaces.

The budget is measured in *objective evaluations* (calls to problem.evaluate),
because being efficient in that count is the whole point of SERENE-MH.

Each iteration the selector returns a small *slate* of actions to actually try.
Baseline selectors return a single action; the SERENE-MH controller proposes
several candidates, gates them with a surrogate, and returns only the few worth
a real evaluation. The engine then:
  1. applies each action in the slate to the current incumbent,
  2. evaluates each resulting candidate (the expensive, counted step),
  3. rewards each action by how much it improved on the incumbent (this reward
     is independent of whether the move is later accepted - that is exactly the
     'counterfactual credit' SERENE-MH relies on),
  4. takes the best candidate of the slate and asks the acceptance criterion
     whether to move there,
  5. updates the best-so-far and lets the selector learn from all the outcomes.
"""

from collections import deque
from dataclasses import dataclass, field

from .problem import Solution


@dataclass
class Outcome:
    """The result of trying one action during an iteration."""

    index: int          # which action (its position in selector.actions)
    action: object      # the Action object
    candidate: Solution # the solution it produced (already evaluated)
    reward: float       # improvement over the incumbent it started from
    accepted: bool = False  # True only for the slate's chosen candidate, if accepted
    new_best: bool = False  # did this candidate become the new best-so-far?


@dataclass
class SearchState:
    """Everything the loop tracks about the current run.

    `telemetry()` returns a small snapshot of progress. Baseline selectors ignore
    it, but the SERENE-MH controller uses it as the context features for its policy.
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
        incumbent_gap = 0.0
        if self.best.objective is not None and abs(self.best.objective) > 1e-12:
            incumbent_gap = (self.incumbent.objective - self.best.objective) / abs(self.best.objective)
        return {
            "frac_budget": frac_budget,
            "recent_improvement": recent_improvement,
            "accept_rate": accept_rate,
            "stagnation": self.stagnation,
            "incumbent_gap": incumbent_gap,
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
        start_incumbent = state.incumbent  # all slate moves are measured from here

        # 1. ask the selector which action(s) to actually try this iteration
        slate = selector.select(state, rng)

        # 2. apply and evaluate each action in the slate
        outcomes = []
        for index in slate:
            action = selector.actions[index]
            candidate = action.operator.apply(start_incumbent, rng, **action.params)
            candidate.objective = problem.evaluate(candidate)
            state.n_evals += 1
            reward = reward_fn(start_incumbent, candidate)
            outcomes.append(Outcome(index=index, action=action, candidate=candidate, reward=reward))
            if state.n_evals >= max_evals:
                break

        # 3. take the best candidate of the slate; let acceptance decide on it
        best_outcome = min(outcomes, key=lambda o: o.candidate.objective)
        accepted = acceptance.accept(state, best_outcome.candidate, rng)
        best_outcome.accepted = accepted
        if accepted:
            state.incumbent = best_outcome.candidate

        # 4. update best-so-far across every candidate we evaluated this iteration
        improved_best = False
        for o in outcomes:
            if o.candidate.objective < state.best.objective:
                state.best = o.candidate
                o.new_best = True
                improved_best = True
        state.stagnation = 0 if improved_best else state.stagnation + 1

        # 5. let the selector learn from all outcomes
        selector.update(state, outcomes, rng)

        # 6. bookkeeping
        state.recent_rewards.append(max(o.reward for o in outcomes))
        state.recent_accepts.append(1 if accepted else 0)
        acceptance.step(state)
        state.iteration += 1
        if record:
            history.append(
                {
                    "iteration": state.iteration,
                    "n_evals": state.n_evals,
                    "actions": [o.action.label for o in outcomes],
                    "best_reward": max(o.reward for o in outcomes),
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
