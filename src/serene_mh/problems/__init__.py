"""Concrete optimisation problems that plug into the SERENE-MH core.

Each problem is a `serene_mh.core.Problem` subclass plus its operators; nothing
here touches the search engine. TSP and CVRP are the Paper 1 testbeds; the
restoration problem (Paper 2) is added later.
"""

from .tsp import (
    TSP,
    TwoOpt,
    OrOpt,
    Swap,
    DoubleBridge,
    SegmentShuffle,
    DestroyRepair,
    tour_length,
    euclidean_matrix,
    nearest_neighbor_tour,
    random_euclidean_instance,
)
from .tsplib import load_tsplib
from .benchmarks import (
    download_tsplib,
    load_benchmarks,
    load_optima,
    OPTIMA,
    PAPER1,
    PAPER1_TSP,
    PAPER1_ATSP,
)

__all__ = [
    "TSP",
    "TwoOpt",
    "OrOpt",
    "Swap",
    "DoubleBridge",
    "SegmentShuffle",
    "DestroyRepair",
    "tour_length",
    "euclidean_matrix",
    "nearest_neighbor_tour",
    "random_euclidean_instance",
    "load_tsplib",
    "download_tsplib",
    "load_benchmarks",
    "load_optima",
    "OPTIMA",
    "PAPER1",
    "PAPER1_TSP",
    "PAPER1_ATSP",
]
