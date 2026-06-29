"""Download standard TSPLIB benchmark instances (and their known optima).

At this stage we use a mixed set to see how SERENE-MH behaves across problem
structure: 20 symmetric TSP instances and all 19 asymmetric ATSP instances in
TSPLIB. Each has a documented optimal tour length, so we can report true gaps.

Files are saved to data/raw/tsplib/<name>.tsp (symmetric) or <name>.atsp
(asymmetric); the optima for the requested set are written to optima.json.

Run it:
    python -m serene_mh.problems.benchmarks                 # the whole mixed set
    python -m serene_mh.problems.benchmarks berlin52 ftv33  # just these

Note: the operators in tsp.py were designed for symmetric TSP. They still run on
ATSP (the objective sums the *directed* tour cost correctly), but 2-opt's segment
reversal is not ideal on asymmetric instances - good enough for exploration, not
a specialised ATSP solver.
"""

import gzip
import json
import os
import sys
import urllib.request

# --- symmetric TSP: 20 instances spanning ~50-320 cities -> optimal tour length
TSP_OPTIMA = {
    "eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538, "pr76": 108159,
    "rat99": 1211, "kroA100": 21282, "kroC100": 20749, "rd100": 7910, "eil101": 629,
    "lin105": 14379, "pr124": 59030, "bier127": 118282, "ch130": 6110, "ch150": 6528,
    "kroA150": 26524, "u159": 42080, "rat195": 2323, "kroA200": 29368, "lin318": 42029,
}

# --- asymmetric ATSP: all 19 TSPLIB instances -> optimal tour length
ATSP_OPTIMA = {
    "br17": 39, "ftv33": 1286, "ftv35": 1473, "ftv38": 1530, "p43": 5620,
    "ftv44": 1613, "ftv47": 1776, "ry48p": 14422, "ft53": 6905, "ftv55": 1608,
    "ftv64": 1839, "ft70": 38673, "ftv70": 1950, "kro124p": 36230, "ftv170": 2755,
    "rbg323": 1326, "rbg358": 1163, "rbg403": 2465, "rbg443": 2720,
}

OPTIMA = {**TSP_OPTIMA, **ATSP_OPTIMA}

# per-kind download sources: (url template, is_gzipped), tried in order
_SOURCES = {
    "tsp": [
        ("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz", True),
        ("https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp", False),
    ],
    "atsp": [
        ("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/{name}.atsp.gz", True),
    ],
}


def _kind(name):
    return "atsp" if name in ATSP_OPTIMA else "tsp"


def _ext(name):
    return ".atsp" if _kind(name) == "atsp" else ".tsp"


def _city_count(name):
    """The number embedded in a TSPLIB name ~ its city count (e.g. kroA100 -> 100)."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


PAPER1_TSP = sorted(TSP_OPTIMA, key=_city_count)
PAPER1_ATSP = sorted(ATSP_OPTIMA, key=_city_count)
PAPER1 = PAPER1_TSP + PAPER1_ATSP


def _fetch(name, timeout=30):
    """Return the instance text for `name`, trying each source until one works."""
    last_error = None
    for url_template, is_gzip in _SOURCES[_kind(name)]:
        url = url_template.format(name=name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "serene-mh/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
            if is_gzip:
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="replace")
            if "NODE_COORD_SECTION" in text or "EDGE_WEIGHT_SECTION" in text:
                return text
            last_error = ValueError("downloaded file is not a TSPLIB instance")
        except Exception as exc:  # noqa: BLE001 - try the next source
            last_error = exc
    raise RuntimeError(f"could not download '{name}': {last_error}")


def download_tsplib(names=None, dest="data/raw/tsplib", timeout=30, overwrite=False, verbose=True):
    """Download the given instances (default: the mixed Paper-1 set) into `dest`.

    Skips files that already exist (unless `overwrite`). Always (re)writes
    optima.json for the requested set. Returns the list of saved instance paths.
    """
    names = list(names) if names else list(PAPER1)
    os.makedirs(dest, exist_ok=True)

    saved, failed = [], []
    for name in names:
        path = os.path.join(dest, name + _ext(name))
        if os.path.exists(path) and not overwrite:
            if verbose:
                print(f"  skip  {name} (exists)")
            saved.append(path)
            continue
        try:
            text = _fetch(name, timeout=timeout)
        except RuntimeError as exc:
            failed.append(name)
            if verbose:
                print(f"  FAIL  {name}: {exc}")
            continue
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
        saved.append(path)
        if verbose:
            print(f"  got   {name}  ({len(text):,} chars)")

    optima = {n: OPTIMA[n] for n in names if n in OPTIMA}
    with open(os.path.join(dest, "optima.json"), "w") as fh:
        json.dump(optima, fh, indent=2, sort_keys=True)

    if verbose:
        print(f"\nsaved {len(saved)} instance(s) to {dest}; {len(failed)} failed.")
        if failed:
            print("failed:", ", ".join(failed))
    return saved


def load_optima(dest="data/raw/tsplib"):
    """Load the instance -> optimal tour length map written by download_tsplib."""
    with open(os.path.join(dest, "optima.json")) as fh:
        return json.load(fh)


def load_benchmarks(names=None, dest="data/raw/tsplib", start="nearest"):
    """Load downloaded instances as a {name: TSP} dict, ready for run_study.

    Defaults to the curated Paper-1 set; skips any name not present on disk.
    Works for both symmetric (.tsp) and asymmetric (.atsp) files.
    """
    from .tsplib import load_tsplib

    names = names if names is not None else PAPER1
    instances = {}
    for name in names:
        path = os.path.join(dest, name + _ext(name))
        if os.path.exists(path):
            instances[name] = load_tsplib(path, start=start)
    return instances


if __name__ == "__main__":
    requested = sys.argv[1:] or None
    download_tsplib(requested)
