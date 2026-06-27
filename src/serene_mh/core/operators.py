"""Operators (moves) and how they become 'actions' for the selector.

An *operator* turns a solution into a neighbouring one (2-opt, a random relocate,
a destroy-and-repair, ...). An operator can expose a few discrete 'knob settings'
- its parameters. Each (operator, setting) combination is one *action*: the unit
the operator-selection layer chooses between.
"""

from dataclasses import dataclass, field


class Operator:
    """Base class for a move. Subclasses set `name` and implement `apply`."""

    name: str = "operator"

    def param_settings(self) -> list[dict]:
        """The discrete parameter settings this operator offers.

        Default: a single setting with no parameters. An operator like
        'remove k arcs' would return e.g. [{"k": 1}, {"k": 2}, {"k": 3}].
        """
        return [{}]

    def apply(self, solution, rng, **params):
        """Return a NEW candidate Solution built from `solution`.

        Must not modify `solution` in place - the engine keeps using it.
        """
        raise NotImplementedError


@dataclass
class Action:
    """A concrete choice: an operator together with one parameter setting."""

    operator: Operator
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """A short human-readable name, handy for logs and plots."""
        if not self.params:
            return self.operator.name
        knobs = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.operator.name}({knobs})"


def build_actions(operators) -> list[Action]:
    """Expand every operator's parameter settings into a flat list of actions.

    This flat list is exactly the set of 'arms' an operator selector chooses among.
    """
    actions = []
    for op in operators:
        for params in op.param_settings():
            actions.append(Action(op, dict(params)))
    return actions
