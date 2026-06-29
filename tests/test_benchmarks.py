"""Tests for the benchmark metadata and loaders.

These avoid the network: they check the curated optima/lists are well-formed and,
if the instances have already been downloaded, that they load and match their
dimensions. Downloading itself is exercised manually, not in the test suite.
"""

import os

import pytest

from serene_mh.problems import benchmarks as bm

DEST = "data/raw/tsplib"


def test_curated_sets_are_well_formed():
    assert len(bm.TSP_OPTIMA) == 20
    assert len(bm.ATSP_OPTIMA) == 19
    assert len(bm.OPTIMA) == 39  # no key collisions between the two
    assert set(bm.TSP_OPTIMA).isdisjoint(bm.ATSP_OPTIMA)
    assert all(isinstance(v, int) and v > 0 for v in bm.OPTIMA.values())
    assert bm.PAPER1 == bm.PAPER1_TSP + bm.PAPER1_ATSP
    assert len(bm.PAPER1) == 39


def test_kind_and_extension_routing():
    assert bm._kind("berlin52") == "tsp"
    assert bm._kind("br17") == "atsp"
    assert bm._ext("kroA100") == ".tsp"
    assert bm._ext("ftv33") == ".atsp"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEST, "berlin52.tsp")),
    reason="benchmarks not downloaded",
)
def test_loaded_instances_match_dimensions():
    import numpy as np

    instances = bm.load_benchmarks()
    assert instances["berlin52"].size == 52
    assert all(p.size > 0 for p in instances.values())

    # optima cover everything we loaded
    optima = bm.load_optima(DEST)
    assert all(name in optima for name in instances)

    # asymmetric instances must load a genuinely asymmetric distance matrix
    atsp = [n for n in instances if n in bm.ATSP_OPTIMA]
    if atsp:
        D = instances[atsp[0]].D
        assert not np.allclose(D, D.T)
