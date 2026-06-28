"""The Travelling Salesman Problem (symmetric, distance-matrix based).

A solution is a *tour*: a permutation of the city indices 0..n-1, understood
cyclically (the last city connects back to the first). The objective is the
total length of that cycle - a cheap O(n) computation, which is why TSP is our
"cheap-evaluation" contrast case for the method paper.

The three operators (2-opt, Or-opt, swap) are generic permutation moves: they
only shuffle the tour and never look at the distances, so they could be reused
for any permutation problem.
"""

import math

import numpy as np

from ..core.problem import Solution, Problem
from ..core.operators import Operator


# --------------------------------------------------------------------------- data
def euclidean_matrix(coords, rounded: bool = True) -> np.ndarray:
    """Full distance matrix from 2-D coordinates.

    `rounded` uses the TSPLIB EUC_2D convention (distances rounded to the nearest
    integer); set it False for exact Euclidean distances.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    if rounded:
        dist = np.round(dist)
    return dist


def tour_length(tour, distance_matrix) -> float:
    """Length of the cyclic tour (sum of edges, including the return edge)."""
    tour = np.asarray(tour)
    return float(distance_matrix[tour, np.roll(tour, -1)].sum())


def nearest_neighbor_tour(distance_matrix, start: int = 0) -> np.ndarray:
    """Greedy nearest-neighbour construction: a decent starting tour."""
    D = distance_matrix
    n = len(D)
    visited = [start]
    used = np.zeros(n, dtype=bool)
    used[start] = True
    current = start
    for _ in range(n - 1):
        dists = D[current].copy()
        dists[used] = np.inf
        nxt = int(np.argmin(dists))
        visited.append(nxt)
        used[nxt] = True
        current = nxt
    return np.array(visited)


class TSP(Problem):
    """Symmetric TSP defined by a distance matrix.

    `start` controls the initial tour: "nearest" (nearest-neighbour) or "random".
    `coords` (optional) is kept around only for plotting.
    """

    def __init__(self, distance_matrix, name: str = "tsp", start: str = "nearest", coords=None):
        self.D = np.asarray(distance_matrix, dtype=float)
        self.size = len(self.D)
        self.name = name
        self.start = start
        self.coords = None if coords is None else np.asarray(coords, dtype=float)

    def initial_solution(self, rng) -> Solution:
        if self.start == "nearest":
            tour = nearest_neighbor_tour(self.D, start=int(rng.integers(self.size)))
        else:
            tour = rng.permutation(self.size)
        return Solution(data=np.asarray(tour))

    def evaluate(self, solution: Solution) -> float:
        return tour_length(solution.data, self.D)

    def operators(self) -> list:
        # A heterogeneous portfolio so that *which* operator to use actually
        # matters by context: intensifiers (2-opt, Or-opt, swap), diversifiers
        # (double-bridge, segment-shuffle) for escaping local optima, and an
        # ALNS-style destroy-and-repair. DestroyRepair needs the distances.
        return [
            TwoOpt(),
            OrOpt(),
            Swap(),
            DoubleBridge(),
            SegmentShuffle(),
            DestroyRepair(self.D),
        ]

    def signature(self, solution: Solution):
        return tuple(int(c) for c in solution.data)


# ---------------------------------------------------------------------- operators
class TwoOpt(Operator):
    """Reverse the tour segment between two random positions (the 2-opt move).

    This removes two edges and reconnects the tour the other way - the classic
    move for untangling crossings.
    """

    name = "2opt"

    def apply(self, solution, rng, **params):
        tour = solution.data
        n = len(tour)
        i, j = sorted(int(x) for x in rng.choice(n, size=2, replace=False))
        new = tour.copy()
        new[i:j + 1] = new[i:j + 1][::-1]
        return Solution(data=new)


class OrOpt(Operator):
    """Move a short contiguous segment (length 1-3) to a random new position."""

    name = "oropt"

    def param_settings(self):
        return [{"seg": 1}, {"seg": 2}, {"seg": 3}]

    def apply(self, solution, rng, seg=1):
        tour = solution.data
        n = len(tour)
        if seg >= n:
            seg = 1
        i = int(rng.integers(n - seg + 1))      # where the segment starts
        segment = tour[i:i + seg]
        rest = np.concatenate([tour[:i], tour[i + seg:]])
        j = int(rng.integers(len(rest) + 1))    # where to reinsert it
        new = np.concatenate([rest[:j], segment, rest[j:]])
        return Solution(data=new)


class Swap(Operator):
    """Swap the cities at two random positions."""

    name = "swap"

    def apply(self, solution, rng, **params):
        tour = solution.data.copy()
        i, j = (int(x) for x in rng.choice(len(tour), size=2, replace=False))
        tour[i], tour[j] = tour[j], tour[i]
        return Solution(data=tour)


class DoubleBridge(Operator):
    """The classic 4-opt 'double bridge' perturbation: cut the tour into four
    pieces A,B,C,D and reconnect them as A,C,B,D. A strong *diversifying* move
    that escapes 2-opt local optima - most useful when the search has stalled.
    """

    name = "double_bridge"

    def apply(self, solution, rng, **params):
        tour = solution.data
        n = len(tour)
        if n < 4:
            return Solution(data=tour.copy())
        p1, p2, p3 = sorted(int(x) for x in rng.choice(range(1, n), size=3, replace=False))
        new = np.concatenate([tour[:p1], tour[p2:p3], tour[p1:p2], tour[p3:]])
        return Solution(data=new)


class SegmentShuffle(Operator):
    """Randomly shuffle a short window of the tour - a small diversifying kick."""

    name = "segment_shuffle"

    def param_settings(self):
        return [{"length": 5}, {"length": 10}]

    def apply(self, solution, rng, length=5):
        tour = solution.data.copy()
        n = len(tour)
        length = min(length, n)
        i = int(rng.integers(n - length + 1))
        segment = tour[i:i + length].copy()
        rng.shuffle(segment)
        tour[i:i + length] = segment
        return Solution(data=tour)


class DestroyRepair(Operator):
    """ALNS-style move: remove q cities at random, then greedily reinsert each at
    its cheapest position (cheapest-insertion repair). Needs the distance matrix.
    """

    name = "destroy_repair"

    def __init__(self, distance_matrix):
        self.D = np.asarray(distance_matrix, dtype=float)

    def param_settings(self):
        return [{"q": 2}, {"q": 3}, {"q": 5}]

    def apply(self, solution, rng, q=2):
        D = self.D
        tour = [int(c) for c in solution.data]
        n = len(tour)
        q = min(q, n - 1)
        remove_pos = set(int(p) for p in rng.choice(n, size=q, replace=False))
        removed = [tour[p] for p in remove_pos]
        remaining = [tour[p] for p in range(n) if p not in remove_pos]
        for city in removed:
            best_pos, best_inc = 0, math.inf
            for p in range(len(remaining)):
                prev = remaining[p - 1]            # cyclic: position before p
                nxt = remaining[p]
                inc = D[prev, city] + D[city, nxt] - D[prev, nxt]
                if inc < best_inc:
                    best_inc, best_pos = inc, p
            remaining.insert(best_pos, city)
        return Solution(data=np.array(remaining))


# ----------------------------------------------------------------- test instances
def random_euclidean_instance(n, rng, box: float = 1000.0, rounded: bool = True, start="nearest"):
    """A random Euclidean TSP instance on an n-point square - handy for tests."""
    coords = rng.uniform(0.0, box, size=(n, 2))
    return TSP(euclidean_matrix(coords, rounded), name=f"rand-eucl-{n}", start=start, coords=coords)
