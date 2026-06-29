"""Download standard OR-Library capacitated facility location (CFLP) instances
and compute their exact optima.

We use the classic **cap41-cap134** set: 37 instances in three size tiers -
16x50 (cap41-74), 25x50 (cap81-104) and 50x50 (cap111-134) facilities x
customers. Many instances blocked by size give the statistical power (and the
hierarchical-prior / transfer story) the method paper needs, just like the 20
TSPLIB instances do for TSP.

Why we compute optima instead of hardcoding them: OR-Library ships only the
instances, not the optimal values, and the cap problems use **splittable
(divisible) demand** - exactly our CFLP model (assignment = transportation LP).
So we can solve each instance to optimality with PuLP+CBC (a binary open-decision
MILP with continuous assignment) and write the result to optima.json. This is
reproducible, matches our evaluation model exactly, and the CBC value for cap41
(1040444.37) agrees with the published optimum (1040444.375) - a check on both
the parser and the min-cost-flow evaluator. The larger capa/b/c instances
(100x1000) use a different file layout and are too big for the inner-loop
expensive-eval regime, so they are intentionally left out.

Run it:
    python -m serene_mh.problems.cflp_benchmarks            # download + solve all
    python -m serene_mh.problems.cflp_benchmarks cap41 cap131
"""

import json
import os
import sys
import urllib.request

import numpy as np

from .cflp import CFLP


# --- the classic OR-Library capacitated set, by size tier
_TIER_16x50 = ["cap41", "cap42", "cap43", "cap44", "cap51",
               "cap61", "cap62", "cap63", "cap64",
               "cap71", "cap72", "cap73", "cap74"]
_TIER_25x50 = ["cap81", "cap82", "cap83", "cap84",
               "cap91", "cap92", "cap93", "cap94",
               "cap101", "cap102", "cap103", "cap104"]
_TIER_50x50 = ["cap111", "cap112", "cap113", "cap114",
               "cap121", "cap122", "cap123", "cap124",
               "cap131", "cap132", "cap133", "cap134"]
CAP_INSTANCES = _TIER_16x50 + _TIER_25x50 + _TIER_50x50

_BASE_URL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}.txt"


# ----------------------------------------------------------------------- parsing
def parse_cflp(text, name="cflp"):
    """Parse an OR-Library `capXX` instance into a CFLP.

    Format: `m n`, then m lines of `capacity fixed_cost`, then for each customer a
    `demand` followed by m "serve-all" costs (the cost of serving that customer's
    *entire* demand from facility i). We convert to our per-unit `unit_cost`
    (serve-all / demand) so the CFLP's transportation eval is in demand units.
    """
    tokens = text.split()
    if any(t.lower() == "capacity" for t in tokens[:6]):
        raise ValueError(f"{name}: capa/b/c-style format is not supported")
    it = iter(tokens)
    m = int(float(next(it)))
    n = int(float(next(it)))

    capacity = np.empty(m)
    fixed = np.empty(m)
    for i in range(m):
        capacity[i] = float(next(it))
        fixed[i] = float(next(it))

    demand = np.empty(n)
    serve_all = np.empty((m, n))  # serve_all[i, j] = cost to serve all of j from i
    for j in range(n):
        demand[j] = float(next(it))
        for i in range(m):
            serve_all[i, j] = float(next(it))

    unit_cost = serve_all / demand[None, :]  # per-unit serving cost
    return CFLP(fixed, capacity.astype(int), demand.astype(int), unit_cost, name=name)


def load_cflp_file(path, name=None):
    """Load a CFLP from a downloaded OR-Library `capXX.txt` file."""
    with open(path) as fh:
        text = fh.read()
    return parse_cflp(text, name=name or os.path.splitext(os.path.basename(path))[0])


# --------------------------------------------------------------------- downloading
def download_cflp(names=None, dest="data/raw/cflp", timeout=30, overwrite=False, verbose=True):
    """Download the given instances (default: the cap41-134 set) into `dest`.

    Skips files that already exist (unless `overwrite`). Returns the list of saved
    instance paths. Optima are computed separately via `build_optima`.
    """
    names = list(names) if names else list(CAP_INSTANCES)
    os.makedirs(dest, exist_ok=True)

    saved, failed = [], []
    for name in names:
        path = os.path.join(dest, name + ".txt")
        if os.path.exists(path) and not overwrite:
            if verbose:
                print(f"  skip  {name} (exists)")
            saved.append(path)
            continue
        url = _BASE_URL.format(name=name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "serene-mh/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            parse_cflp(text, name=name)  # validate it parses before saving
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            if verbose:
                print(f"  FAIL  {name}: {exc}")
            continue
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
        saved.append(path)
        if verbose:
            print(f"  got   {name}  ({len(text):,} chars)")

    if verbose:
        print(f"\nsaved {len(saved)} instance(s) to {dest}; {len(failed)} failed.")
        if failed:
            print("failed:", ", ".join(failed))
    return saved


# ------------------------------------------------------------------- exact solver
def solve_cflp_exact(cflp, msg=False, time_limit=None):
    """Solve a CFLP to optimality with PuLP+CBC (splittable demand).

    Binary open decisions y_i, continuous assignment fractions x_ij in [0, 1]:
        min  sum_i f_i y_i + sum_ij (unit_cost_ij * demand_j) x_ij
        s.t. sum_i x_ij = 1                 (serve every customer)
             sum_j demand_j x_ij <= cap_i y_i   (capacity, only if open)
    Returns (optimal_objective, open_mask). This is the exact ground-truth
    reference for small instances (and how optima.json is built).
    """
    import pulp

    m, n = cflp.m, cflp.n
    f = cflp.fixed_cost
    cap = cflp.capacity
    d = cflp.demand
    uc = cflp.unit_cost

    prob = pulp.LpProblem(cflp.name, pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)
         for i in range(m) for j in range(n)}

    prob += (pulp.lpSum(f[i] * y[i] for i in range(m))
             + pulp.lpSum(uc[i, j] * d[j] * x[(i, j)] for i in range(m) for j in range(n)))
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(m)) == 1
    for i in range(m):
        prob += pulp.lpSum(d[j] * x[(i, j)] for j in range(n)) <= cap[i] * y[i]

    solver = pulp.PULP_CBC_CMD(msg=1 if msg else 0, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    open_mask = np.array([pulp.value(y[i]) > 0.5 for i in range(m)], dtype=bool)
    return float(pulp.value(prob.objective)), open_mask, status


def build_optima(names=None, dest="data/raw/cflp", time_limit=None, verbose=True):
    """Solve each downloaded instance exactly and write `dest/optima.json`.

    Returns the {name: optimum} map. Skips instances not present on disk; warns on
    any that CBC could not prove optimal (e.g. if a time limit was hit).
    """
    names = list(names) if names else list(CAP_INSTANCES)
    optima = {}
    for name in names:
        path = os.path.join(dest, name + ".txt")
        if not os.path.exists(path):
            if verbose:
                print(f"  skip  {name} (not downloaded)")
            continue
        cflp = load_cflp_file(path, name=name)
        opt, _mask, status = solve_cflp_exact(cflp, time_limit=time_limit)
        optima[name] = opt
        if verbose:
            flag = "" if status == "Optimal" else f"  <-- {status}!"
            print(f"  {name:>7}  ({cflp.m}x{cflp.n})  optimum {opt:,.3f}{flag}")

    with open(os.path.join(dest, "optima.json"), "w") as fh:
        json.dump(optima, fh, indent=2, sort_keys=True)
    if verbose:
        print(f"\nwrote {len(optima)} optima to {os.path.join(dest, 'optima.json')}")
    return optima


# ------------------------------------------------------------------------ loading
def load_cflp_optima(dest="data/raw/cflp"):
    """Load the instance -> optimal objective map written by build_optima."""
    with open(os.path.join(dest, "optima.json")) as fh:
        return json.load(fh)


def load_cflp_benchmarks(names=None, dest="data/raw/cflp"):
    """Load downloaded CFLP instances as a {name: CFLP} dict, ready for run_study.

    Defaults to the cap41-134 set; skips any name not present on disk.
    """
    names = names if names is not None else CAP_INSTANCES
    instances = {}
    for name in names:
        path = os.path.join(dest, name + ".txt")
        if os.path.exists(path):
            instances[name] = load_cflp_file(path, name=name)
    return instances


if __name__ == "__main__":
    requested = sys.argv[1:] or None
    download_cflp(requested)
    print("\nsolving for optima (CBC)...")
    build_optima(requested)
