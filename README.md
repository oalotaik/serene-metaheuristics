# SERENE-MH

**Sample-Efficient Reinforcement-Enhanced Neighborhood Exploration for Metaheuristics**

A learning-based **adaptive operator selection (AOS)** layer for single-solution
metaheuristics. SERENE-MH decides, at each iteration, *which* neighborhood
operator (and discrete parameter setting) to apply, with the explicit goal of
reaching a host metaheuristic's reachable quality in **as few expensive objective
evaluations as possible**. It is intended for combinatorial optimization problems
where evaluating a candidate is costly — e.g. each move requires re-solving an
embedded subproblem — so the number of true objective evaluations, not wall-clock,
is the scarce resource.

This repository accompanies a two-paper program. **Paper 1 (the method)** presents
SERENE-MH as a general framework and demonstrates it on two contrasting testbeds:
the **Travelling Salesman Problem** (TSPLIB; cheap *O(n)* evaluation — a control
case) and the **Capacitated Facility Location Problem** (OR-Library; expensive
evaluation — each open/close re-solves a transportation LP). **Paper 2 (an
application)** applies the method to max-min-fair post-disruption network
restoration and cites Paper 1; that component is under development.

---

## Method in brief

SERENE-MH replaces the ad-hoc roulette weights of classical ALNS with a small
contextual policy over the operator portfolio:

1. **Contextual Thompson-sampling bandit.** Operators × parameter settings are the
   "arms". A Bayesian-linear value model, conditioned on a low-dimensional **search
   state** (fraction of budget used, recent-improvement rate, acceptance rate,
   stagnation, incumbent gap), is sampled to score the arms — so operator choice
   adapts to the *stage* of the search.
2. **Surrogate-gated slate.** Each iteration the policy proposes a small slate of
   candidate moves; the surrogate (the policy's posterior mean, or an optional
   LightGBM regressor) ranks them, and **only the few most promising are passed to
   the true, expensive objective**. This is the central sample-efficiency lever.
3. **Counterfactual operator credit.** Each tried operator is rewarded by its
   improvement over the start-of-iteration incumbent, *independent* of whether the
   move is later accepted by the SA/Tabu/record-to-record gate — decoupling
   operator credit from the acceptance criterion.
4. **Cross-instance warm-start.** Learned operator values can be exported and used
   as a prior on related instances (a hierarchical / empirical-Bayes transfer
   story).

The controller is deliberately **classical and lightweight** (Bayesian-linear
Thompson sampling + optional gradient-boosted surrogate); there are no neural
networks. It hosts the four standard single-solution metaheuristics as
interchangeable **acceptance criteria** — Local Search (greedy), Simulated
Annealing, Tabu Search, and a record-to-record variant used as the ALNS host.

### How the method is evaluated

SERENE-MH is an operator-selection layer; it cannot exceed the host
metaheuristic's quality ceiling, and its only claim is **efficiency**. The
evaluation protocol reflects that:

- **Budget is the number of true objective evaluations** (deterministic,
  reproducible); wall-clock is secondary.
- **Within-host, selector-vs-selector.** With the same acceptance criterion and the
  same operator portfolio, does SERENE-MH reach the host's reachable quality
  *faster* than the standard AOS baselines (uniform random, ALNS roulette, UCB1,
  EXP3) and a greedy mean-tracking comparator?
- **Efficiency is read at tight budgets:** optimality gap at **25 / 50 / 75 / 100 %**
  of the evaluation budget, plus the normalized regret area-under-curve (AUC).
- **Statistics:** within-host Wilcoxon signed-rank (blocked by instance,
  Holm-corrected) for pairwise claims; Friedman + Nemenyi for the full grid.
- **Optimality gaps** (to known or exactly computed optima) are reported as host
  context, not as the controller's verdict.

---

## Testbeds and benchmark data

| Problem | Benchmark set | Evaluation cost | Optima | Role |
|---|---|---|---|---|
| **CFLP** | OR-Library `cap41`–`cap134` (16/25/50 facilities × 50 customers) | **Expensive** — each candidate re-solves a transportation LP (min-cost flow, ~10 ms) | Computed exactly via PuLP+CBC | Primary expensive-evaluation testbed |
| **TSP** | TSPLIB (20 symmetric instances, `eil51`–`lin318`) | Cheap — *O(n)* tour length | Published optima | Cheap-evaluation control case |

**CFLP formulation.** A solution is the set of open facilities (a binary vector).
The objective is the fixed cost of the open facilities plus the optimal cost of
serving all (splittable) customer demand from them — the latter obtained by solving
the **transportation problem** exactly via network min-cost flow. There is no cheap
move delta: opening or closing a facility re-solves the whole assignment, which is
precisely the regime in which surrogate gating is worthwhile. Infeasible open-sets
(open capacity below total demand) incur a penalty. OR-Library "serve-all" costs are
converted to per-unit costs (`c_ij / d_j`); facility-open decisions make the exact
problem a MILP, solved to optimality with CBC to provide ground-truth gaps.

---

## Installation

Python **3.13**. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell; on Unix use: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # installs the `serene_mh` package (editable)
pytest -q                         # optional: run the test suite
```

The numerical/solver stack is open-source: NumPy/SciPy/pandas/NetworkX,
scikit-learn + LightGBM (the controller), PuLP+CBC (LP/MILP), Matplotlib.

---

## Reproducing the results

**1. Fetch the benchmark data** (one-off; writes under `data/raw/`):

```bash
# OR-Library CFLP: download cap41–134 and compute exact optima with CBC -> data/raw/cflp/
python -m serene_mh.problems.cflp_benchmarks

# TSPLIB TSP: download the 20 symmetric instances + optima -> data/raw/tsplib/
python -m serene_mh.problems.benchmarks
```

The `cap41`–`cap134` instances and their CBC-computed `optima.json` are also
committed under `data/raw/cflp/`, so the CFLP studies run without network access.

**2. Run the studies** (each writes tidy CSVs under `results/tables/<study>/`):

```bash
# CFLP: full host × selector grid + greedy-mean comparator. Three tiers:
python scripts/run_cflp_study.py             # pilot  : 16×50 tier, 3 seeds, 300 evals
python scripts/run_cflp_study.py --medium    # medium : 25×50 tier, 3 seeds, 300 evals
python scripts/run_cflp_study.py --full      # full   : 37 instances, 5 seeds, 500 evals

# CFLP component ablation (which parts of SERENE-MH carry the win), exploratory hosts:
python scripts/run_cflp_ablations.py

# TSP control study (the cheap-evaluation contrast case):
python scripts/run_tsp_study.py
```

**3. Read the outputs.** Each study directory contains:

- `results.csv` — one row per (instance, config, seed) with the efficiency metrics
  (`q25`/`q50`/`q75`/`q100`, gap at each fraction, `auc`, evals-to-target, final gap);
- `ranks.csv` — Friedman average ranks (+ Nemenyi critical difference) per metric;
- `within_host_efficiency.csv` / `vs_full.csv` — within-host Wilcoxon tests;
- `gap_by_budget_fraction.csv` — mean optimality gap (%) at 25/50/75/100 % of budget.

Publication-ready figures (vector PDF + 300-dpi PNG) are written to
`results/figures/` and are regenerable from the committed scripts and data.

---

## Current findings (summary)

On **OR-Library CFLP** (expensive evaluation), SERENE-MH delivers significant
**within-host** efficiency gains over the standard AOS baselines at tight budgets,
under the exploratory hosts (Simulated Annealing, Tabu, ALNS); the component
ablation shows that **contextual state, Thompson exploration, and surrogate gating
each contribute**. On **TSPLIB TSP** (cheap evaluation, one dominant operator) those
gains collapse to the surrogate-gated slate alone — a deliberate contrast that
characterizes *when* the machinery pays. Numbers and significance tests live in
`results/tables/`. (Studies at full scale are ongoing; see `PROJECT_LOG.md` for the
live record.)

---

## Repository layout

```
src/serene_mh/
  core/         # problem-agnostic search machinery
                #   problem.py (Solution/Problem), operators.py, acceptance.py
                #   (LS/SA/Tabu/RRT), selectors.py (AOS baselines), engine.py (run_search)
  controller/   # SERENE-MH: serene.py (contextual TS + gated slate), surrogate.py (LightGBM)
  problems/     # tsp.py + tsplib.py + benchmarks.py (TSPLIB);
                # cflp.py + cflp_benchmarks.py (OR-Library + exact CBC optima)
  experiments/  # harness.py (multi-instance/seed runner), metrics.py, stats.py,
                # analysis.py (tidy tables + within-host tests), plots.py, presets.py
scripts/        # reproducible study drivers (run_cflp_study.py, run_cflp_ablations.py, ...)
data/raw/       # benchmark instances + optima (TSPLIB, OR-Library CFLP)
results/        # tables/ (CSV deliverables) and figures/ (PDF + PNG)
tests/          # pytest suite
```

A note on scope: `serene_mh.core` is problem-agnostic, so adding a new
optimization problem means implementing one `Problem` subclass (initial solution,
objective, operator portfolio) — the search engine, AOS baselines, controller, and
experiment harness are reused unchanged.

---

## References

The Paper 2 case study follows the problem of Bin Obaid, Almoghathawi & Algafri,
*"Max-Min Fair Restoration of Infrastructure Networks,"* **Mathematics** 13 (2025)
3112 — used as a faithful, comparable application benchmark, not reimplemented.

## Citation

```bibtex
@misc{Alotaik_SERENE_MH,
  title  = {SERENE-MH: Sample-Efficient Reinforcement-Enhanced Neighborhood
            Exploration for Metaheuristics},
  author = {Alotaik, O.},
  year   = {2025},
  note   = {Working paper / software, https://github.com/oalotaik/serene-metaheuristics}
}
```
