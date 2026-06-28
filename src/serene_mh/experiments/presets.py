"""Ready-made config sets for the experiments.

The Paper 1 comparison must cover every base metaheuristic in the plan
(LS / SA / Tabu / ALNS) crossed with every operator selector (the baselines plus
SERENE-MH). We express each base metaheuristic as an *acceptance host* and pair
it with each selector:

  - LS   -> Greedy acceptance
  - SA   -> SimulatedAnnealing acceptance
  - Tabu -> Tabu acceptance
  - ALNS -> RecordToRecord acceptance (the destroy/repair operators live in the
            problem's portfolio, so "roulette | ALNS" is the canonical ALNS, and
            "serene | ALNS" is SERENE-MH driving the same ALNS machinery)

`standard_configs` returns the full host x selector grid; subset it with the
`hosts` / `selectors` arguments for quick runs.
"""

from .harness import Config
from ..core import Greedy, SimulatedAnnealing, Tabu, RecordToRecord
from ..core import UniformRandom, Roulette, UCB1, EXP3
from ..controller import SereneMH


def _hosts(max_evals, tabu_tenure, rrt_deviation):
    return {
        "LS": lambda: Greedy(),
        "SA": lambda: SimulatedAnnealing(max_evals=max_evals),
        "Tabu": lambda: Tabu(tenure=tabu_tenure),
        "ALNS": lambda: RecordToRecord(deviation=rrt_deviation),
    }


def _selectors(slate_size, n_exec):
    return {
        "uniform": lambda actions: UniformRandom(actions),
        "roulette": lambda actions: Roulette(actions),
        "ucb1": lambda actions: UCB1(actions),
        "exp3": lambda actions: EXP3(actions),
        "serene": lambda actions: SereneMH(actions, slate_size=slate_size, n_exec=n_exec),
    }


def standard_configs(
    max_evals,
    hosts=None,
    selectors=None,
    tabu_tenure=20,
    rrt_deviation=0.02,
    slate_size=3,
    n_exec=1,
):
    """The full grid of base metaheuristics x operator selectors.

    `hosts` / `selectors`, if given, restrict to those names (handy for fast runs).
    Config names look like "serene|SA" or "roulette|ALNS".
    """
    all_hosts = _hosts(max_evals, tabu_tenure, rrt_deviation)
    all_selectors = _selectors(slate_size, n_exec)
    host_names = hosts or list(all_hosts)
    selector_names = selectors or list(all_selectors)

    configs = []
    for hname in host_names:
        for sname in selector_names:
            configs.append(
                Config(
                    name=f"{sname}|{hname}",
                    make_selector=all_selectors[sname],
                    make_acceptance=all_hosts[hname],
                )
            )
    return configs
