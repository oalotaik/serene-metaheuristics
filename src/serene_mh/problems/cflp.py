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
# A large scale keeps the rounding error far below any reported gap precision (so
# "best found" never spuriously dips below the exact optimum) while staying within
# safe integer magnitudes.
_COST_SCALE = 100000


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
        # A heterogeneous, stage-dependent portfolio so that *which* operator to
        # apply genuinely matters as the search progresses:
        #   - random moves (close/open/swap, multi-flip) = cheap diversifiers,
        #   - structure-aware greedy moves (greedy drop/open/swap) = intensifiers
        #     that exploit the cost structure,
        #   - destroy-repair = a strong restructuring move for escaping optima.
        # The greedy/destroy operators use only CHEAP O(m*n) cost-structure proxies
        # (nearest-open assignment, ignoring capacity) - they never solve the
        # transportation LP. That keeps the engine's single evaluate() call the
        # only expensive, *counted* operation, which is exactly what makes the
        # surrogate-gating sample-efficiency story meaningful on CFLP.
        return [
            CloseOne(),
            OpenOne(),
            Swap(),
            MultiFlip(),
            GreedyDrop(self.capacity, self.demand, self.unit_cost),
            GreedyOpen(self.demand, self.unit_cost),
            GreedySwap(self.demand, self.unit_cost),
            DestroyRepair(self.capacity, self.demand, self.unit_cost),
        ]

    def signature(self, solution: Solution):
        return tuple(bool(b) for b in solution.data)


# ----------------------------------------------------------- cheap cost-structure
# These proxies estimate facility usefulness in O(m*n) WITHOUT solving the
# transportation LP, by assigning each customer's full demand to its cheapest open
# facility (ignoring capacity). They let the greedy/destroy operators be
# structure-aware while staying cheap - the LP is reserved for the counted eval.
def _facility_load(open_mask, demand, unit_cost):
    """Demand each open facility would serve under capacity-free nearest-open
    assignment (a utilisation proxy). Closed facilities get 0."""
    load = np.zeros(len(open_mask))
    open_idx = np.flatnonzero(open_mask)
    if len(open_idx) == 0:
        return load
    pick = open_idx[unit_cost[open_idx].argmin(axis=0)]  # cheapest open facility / customer
    np.add.at(load, pick, demand)
    return load


def _open_gain(open_mask, demand, unit_cost):
    """For each CLOSED facility, the demand-weighted reduction in nearest-open
    serving cost if it were opened. Open facilities get -inf, so argmax always
    returns a closed facility."""
    if open_mask.any():
        best = unit_cost[open_mask].min(axis=0)            # current cheapest open cost / customer
    else:
        best = np.full(unit_cost.shape[1], np.inf)
    reduction = np.maximum(0.0, best[None, :] - unit_cost)  # (m, n)
    gain = (reduction * demand[None, :]).sum(axis=1)        # (m,)
    gain[open_mask] = -np.inf
    return gain


# ---------------------------------------------------------------------- operators
class CloseOne(Operator):
    """Close one random open facility (cheap intensify: cut a fixed cost)."""

    name = "close"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        if len(open_idx) > 1:  # keep at least one open
            data[int(rng.choice(open_idx))] = False
        return Solution(data=data)


class OpenOne(Operator):
    """Open one random closed facility (cheap diversify / restore feasibility)."""

    name = "open"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        closed_idx = np.flatnonzero(~data)
        if len(closed_idx) > 0:
            data[int(rng.choice(closed_idx))] = True
        return Solution(data=data)


class Swap(Operator):
    """Close one random open facility and open one random closed facility."""

    name = "swap"

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        closed_idx = np.flatnonzero(~data)
        if len(open_idx) > 0 and len(closed_idx) > 0:
            data[int(rng.choice(open_idx))] = False
            data[int(rng.choice(closed_idx))] = True
        return Solution(data=data)


class MultiFlip(Operator):
    """Flip k random facilities (open<->closed): a diversifying kick whose strength
    grows with k. Most useful once the search has stalled in a local optimum."""

    name = "multiflip"

    def param_settings(self):
        return [{"k": 2}, {"k": 3}]

    def apply(self, solution, rng, k=2):
        data = solution.data.copy()
        k = min(k, len(data))
        idx = rng.choice(len(data), size=k, replace=False)
        data[idx] = ~data[idx]
        if not data.any():  # never leave everything closed
            data[int(rng.integers(len(data)))] = True
        return Solution(data=data)


class GreedyDrop(Operator):
    """Close the least-utilised open facility (cheap nearest-open load proxy) whose
    removal still leaves enough open capacity for total demand. A structure-aware
    intensifier: trims fixed cost without obviously breaking feasibility."""

    name = "greedy_drop"

    def __init__(self, capacity, demand, unit_cost):
        self.capacity = np.asarray(capacity)
        self.demand = np.asarray(demand)
        self.unit_cost = np.asarray(unit_cost, dtype=float)
        self.total = float(self.demand.sum())

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        if len(open_idx) <= 1:
            return Solution(data=data)
        load = _facility_load(data, self.demand, self.unit_cost)
        open_cap = self.capacity[data].sum()
        for i in open_idx[np.argsort(load[open_idx])]:  # least-loaded first
            if open_cap - self.capacity[i] >= self.total:
                data[i] = False
                break
        return Solution(data=data)


class GreedyOpen(Operator):
    """Open the closed facility that most cuts the nearest-open serving cost
    (demand-weighted gain proxy). A structure-aware diversifier that relieves the
    most expensively served customers."""

    name = "greedy_open"

    def __init__(self, demand, unit_cost):
        self.demand = np.asarray(demand)
        self.unit_cost = np.asarray(unit_cost, dtype=float)

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        if data.all():
            return Solution(data=data)
        gain = _open_gain(data, self.demand, self.unit_cost)
        data[int(np.argmax(gain))] = True
        return Solution(data=data)


class GreedySwap(Operator):
    """Relocate capacity: close the least-utilised open facility and open the
    highest-gain closed one (both cheap proxies)."""

    name = "greedy_swap"

    def __init__(self, demand, unit_cost):
        self.demand = np.asarray(demand)
        self.unit_cost = np.asarray(unit_cost, dtype=float)

    def apply(self, solution, rng, **params):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        closed_idx = np.flatnonzero(~data)
        if len(open_idx) == 0 or len(closed_idx) == 0:
            return Solution(data=data)
        load = _facility_load(data, self.demand, self.unit_cost)
        drop = int(open_idx[np.argmin(load[open_idx])])
        gain = _open_gain(data, self.demand, self.unit_cost)
        add = int(np.argmax(gain))
        data[drop] = False
        data[add] = True
        return Solution(data=data)


class DestroyRepair(Operator):
    """ALNS-style restructuring: close a cluster of q open facilities (a random
    seed plus its nearest neighbours in service pattern), then greedily reopen by
    serving-cost gain until the open capacity can cover total demand. A strong
    escape move - bigger q restructures more of the solution."""

    name = "destroy_repair"

    def __init__(self, capacity, demand, unit_cost):
        self.capacity = np.asarray(capacity)
        self.demand = np.asarray(demand)
        self.unit_cost = np.asarray(unit_cost, dtype=float)
        self.total = float(self.demand.sum())

    def param_settings(self):
        return [{"q": 2}, {"q": 3}]

    def apply(self, solution, rng, q=2):
        data = solution.data.copy()
        open_idx = np.flatnonzero(data)
        q = min(q, max(0, len(open_idx) - 1))  # keep at least one open
        if q == 0:
            return Solution(data=data)
        seed = int(rng.choice(open_idx))
        # cluster = the seed + its most similar open facilities (close cost rows =
        # they serve the same customers at similar prices), so we destroy a
        # coherent region rather than scattered facilities.
        dist = np.sqrt(((self.unit_cost - self.unit_cost[seed]) ** 2).sum(axis=1))
        others = sorted((i for i in open_idx if i != seed), key=lambda i: dist[i])
        for i in [seed] + others[:q - 1]:
            data[i] = False
        # repair: reopen by serving-cost gain until capacity can cover demand
        while self.capacity[data].sum() < self.total and not data.all():
            gain = _open_gain(data, self.demand, self.unit_cost)
            data[int(np.argmax(gain))] = True
        if not data.any():
            data[int(rng.integers(len(data)))] = True
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
