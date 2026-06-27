"""SERENE-MH: a sample-efficient operator-selection layer for metaheuristics.

The package is organised so that the generic search machinery is separate from
any specific problem:

  serene_mh.core      - problem-agnostic pieces: the Solution/Problem types,
                        operators, acceptance criteria, baseline operator
                        selectors, and the search loop (`run_search`).
  serene_mh.problems  - concrete problems (TSP, CVRP, restoration) [added later].
  serene_mh.controller- the SERENE-MH operator selector [added later].
  serene_mh.experiments - experiment harness and metrics [added later].
"""

__version__ = "0.0.1"
