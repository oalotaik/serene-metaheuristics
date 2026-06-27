"""Operator-selection strategies (adaptive operator selection, AOS).

Given the list of available actions, a selector decides which one(s) to try next
and learns from the reward each action earns. These are the *baselines* that
SERENE-MH (in serene_mh.controller) is compared against.

Contract (shared with SERENE-MH so they are interchangeable in the engine):
  - `select(state, rng) -> list[int]` returns the indices of the actions to try
    this iteration. The baselines here always return a single index; SERENE-MH
    may return a few. Using indices into `self.actions` keeps bookkeeping simple.
  - `update(state, outcomes, rng)` receives the engine's Outcome objects (one per
    action tried), each carrying `.index` and `.reward`, and learns from them.

`rng` is a numpy random Generator throughout. Rewards are assumed to lie roughly
in [0, 1].
"""

import math


class OperatorSelector:
    """Base class. Subclasses implement `select` and usually `update`."""

    name: str = "selector"

    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)

    def select(self, state, rng) -> list[int]:
        raise NotImplementedError

    def update(self, state, outcomes, rng) -> None:
        """Learn from the outcomes of the actions tried. Default: do nothing."""

    def reset(self) -> None:
        """Restore internal state to its starting point (new run)."""


class UniformRandom(OperatorSelector):
    """Pick an action uniformly at random. The simplest possible baseline."""

    name = "uniform"

    def select(self, state, rng) -> list[int]:
        return [int(rng.integers(self.n))]


class Roulette(OperatorSelector):
    """Classic ALNS adaptive operator selection.

    Each action has a weight; we pick proportional to weight, then nudge the
    chosen action's weight toward the reward it just earned:

        w <- (1 - reaction) * w + reaction * reward

    `reaction` controls how fast weights adapt.
    """

    name = "roulette"

    def __init__(self, actions, reaction: float = 0.1, w_min: float = 1e-3):
        super().__init__(actions)
        self.reaction = reaction
        self.w_min = w_min
        self.weights = [1.0] * self.n

    def reset(self) -> None:
        self.weights = [1.0] * self.n

    def select(self, state, rng) -> list[int]:
        total = sum(self.weights)
        probs = [w / total for w in self.weights]
        return [int(rng.choice(self.n, p=probs))]

    def update(self, state, outcomes, rng) -> None:
        for o in outcomes:
            w = self.weights[o.index]
            self.weights[o.index] = max(self.w_min, (1 - self.reaction) * w + self.reaction * o.reward)


class UCB1(OperatorSelector):
    """UCB1 bandit: trade off trying high-reward actions against rarely-tried ones.

    Plays each action once, then repeatedly picks the action with the highest
        mean_reward + c * sqrt(2 * ln(t) / count).
    Larger `c` means more exploration.
    """

    name = "ucb1"

    def __init__(self, actions, c: float = 1.0):
        super().__init__(actions)
        self.c = c
        self.counts = [0] * self.n
        self.means = [0.0] * self.n
        self.t = 0

    def reset(self) -> None:
        self.counts = [0] * self.n
        self.means = [0.0] * self.n
        self.t = 0

    def select(self, state, rng) -> list[int]:
        # Make sure every action is tried once before comparing scores.
        for i in range(self.n):
            if self.counts[i] == 0:
                return [i]
        t = self.t + 1
        scores = [
            self.means[i] + self.c * math.sqrt(2 * math.log(t) / self.counts[i])
            for i in range(self.n)
        ]
        return [max(range(self.n), key=lambda i: scores[i])]

    def update(self, state, outcomes, rng) -> None:
        for o in outcomes:
            self.t += 1
            self.counts[o.index] += 1
            # Running mean: new_mean = old_mean + (reward - old_mean) / count.
            self.means[o.index] += (o.reward - self.means[o.index]) / self.counts[o.index]


class EXP3(OperatorSelector):
    """EXP3 bandit, suited to rewards that drift over the run.

    Keeps exponential weights; samples action i with a probability that mixes the
    weights with a little uniform exploration (rate `gamma`), and updates using an
    importance-weighted reward so rarely-picked actions still get fair credit.
    """

    name = "exp3"

    def __init__(self, actions, gamma: float = 0.1):
        super().__init__(actions)
        self.gamma = gamma
        self.weights = [1.0] * self.n
        self._probs = [1.0 / self.n] * self.n

    def reset(self) -> None:
        self.weights = [1.0] * self.n
        self._probs = [1.0 / self.n] * self.n

    def select(self, state, rng) -> list[int]:
        total = sum(self.weights)
        self._probs = [(1 - self.gamma) * w / total + self.gamma / self.n for w in self.weights]
        return [int(rng.choice(self.n, p=self._probs))]

    def update(self, state, outcomes, rng) -> None:
        for o in outcomes:
            p = self._probs[o.index]
            estimated_reward = o.reward / max(p, 1e-12)
            self.weights[o.index] *= math.exp(self.gamma * estimated_reward / self.n)
        # Rescale occasionally so the weights cannot overflow.
        biggest = max(self.weights)
        if biggest > 1e6:
            self.weights = [w / biggest for w in self.weights]
