"""The Capacitated Facility Location Problem (CFLP).

A solution is the set of *open* facilities (a boolean vector). The objective is

    total fixed cost of open facilities  +  optimal cost of serving all customers

where the serving cost is found by solving the transportation problem: assign
each customer's (splittable) demand to open facilities, respecting capacities, at
minimum cost. That transportation solve is the "expensive evaluation" we count -
crucially there is NO cheap delta: opening or closing a facility re-solves the
whole assignment, so a surrogate that predicts a move's value without solving is
genuinely useful (the regime TSP could not provide).

Eval uses networkx min-cost flow (a fast, exact algorithm for the transportation
LP). Infeasible open-sets (open capacity < total demand) get a penalty.
"""

import numpy as np
import networkx as nx

from ..core.problem import Solution, Problem
from ..core.operators import Operator


# networkx network-simplex needs integer edge weights to be guaranteed to
# terminate (float weights can cycle / hang), so we scale unit costs to integers.
_COST_SCALE = 1000


def assignment_cost(open_idx, capacities, demands, unit_cost):
    """Minimum cost to serve all demand from the open facilities (transportation
    problem), or None if the open capacity cannot cover total demand.

    `unit_cost[i, j]` is the cost per unit of demand served from facility i to
    customer j. Demand is splittable. Solved as an integer-weighted min-cost flow
    (costs scaled by _COST_SCALE for exact, terminating network simplex).
    """
    total = int(round(demands.sum()))
    if sum(int(capacities[i]) for i in open_idx) < total:
        return None  # not enough open capacity -> infeasible

    G = nx.DiGraph()
    G.add_node("S", demand=-total)
    for j, d in enumerate(demands):
        G.add_node(("c", j), demand=int(round(d)))
    for i in open_idx:
        G.add_edge("S", ("f", i), capacity=int(capacities[i]), weight=0)
        for j in range(len(demands)):
            G.add_edge(("f", i), ("c", j), weight=int(round(unit_cost[i, j] * _COST_SCALE)))
    try:
        return nx.min_cost_flow_cost(G) / _COST_SCALE
    except nx.NetworkXUnfeasible:
        return None


class CFLP(Problem):
    """Capacitated facility location (splittable demand).

    Parameters
    ----------
    fixed_cost : (m,) array       fixed cost of opening each facility
    capacity   : (m,) int array   capacity of each facility
    demand     : (n,) int array   demand of each customer
    unit_cost  : (m, n) array     per-unit serving cost facility i -> customer j
    """

    def __init__(self, fixed_cost, capacity, demand, unit_cost, name="cflp"):
        self.fixed_cost = np.asarray(fixed_cost, dtype=float)
        self.capacity = np.asarray(capacity)
        self.demand = np.asarray(demand)
        self.unit_cost = np.asarray(unit_cost, dtype=float)
        self.m = len(self.fixed_cost)   # facilities
        self.n = len(self.demand)       # customers
        self.name = name
        # a feasible fallback cost (everything open) used as the infeasible penalty base
        self._penalty = float(self.fixed_cost.sum() + self.unit_cost.max() * self.demand.sum()) * 10.0

    def initial_solution(self, rng) -> Solution:
        # start with every facility open: always feasible if the instance is.
        return Solution(data=np.ones(self.m, dtype=bool))

    def evaluate(self, solution: Solution) -> float:
        open_mask = solution.data
        open_idx = [i for i in range(self.m) if open_mask[i]]
        if not open_idx:
            solution.feasible = False
            return self._penalty
        serve = assignment_cost(open_idx, self.capacity, self.demand, self.unit_cost)
        if serve is None:
            solution.feasible = False
            return self._penalty
        solution.feasible = True
        return float(self.fixed_cost[open_idx].sum() + serve)

    def operators(self) -> list:
        return [CloseOne(), OpenOne(), Swap()]

    def signature(self, solution: Solution):
        return tuple(bool(b) for b in solution.data)


# ---------------------------------------------------------------------- operators
class CloseOne(Operator):
    """Close one currently-open facility (intensify: cut fixed cost)."""

    name = "close"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        if len(open_idx) > 1:  # keep at least one open
            data[int(rng.choice(open_idx))] = False
        return Solution(data=data)


class OpenOne(Operator):
    """Open one currently-closed facility (diversify / restore feasibility)."""

    name = "open"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        closed_idx = np.flatnonzero(~data)
        if len(closed_idx) > 0:
            data[int(rng.choice(closed_idx))] = True
        return Solution(data=data)


class Swap(Operator):
    """Close one open facility and open one closed facility."""

    name = "swap"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        closed_idx = np.flatnonzero(~data)
        if len(open_idx) > 0 and len(closed_idx) > 0:
            data[int(rng.choice(open_idx))] = False
            data[int(rng.choice(closed_idx))] = True
        return Solution(data=data)


# ----------------------------------------------------------------- test instances
def random_cflp_instance(m, n, rng, name=None):
    """A small random CFLP instance with guaranteed feasibility (all-open covers)."""
    coords_f = rng.uniform(0, 100, size=(m, 2))
    coords_c = rng.uniform(0, 100, size=(n, 2))
    dist = np.sqrt(((coords_f[:, None, :] - coords_c[None, :, :]) ** 2).sum(-1))
    demand = rng.integers(1, 10, size=n)
    # total capacity comfortably exceeds total demand
    capacity = np.full(m, int(np.ceil(2.0 * demand.sum() / m)))
    fixed_cost = rng.uniform(50, 150, size=m)
    unit_cost = dist  # per-unit serving cost ~ distance
    return CFLP(fixed_cost, capacity, demand, unit_cost, name=name or f"rand-cflp-{m}x{n}")
