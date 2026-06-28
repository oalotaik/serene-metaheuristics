"""Loader for TSPLIB instance files (the standard TSP benchmark format).

Supports the common symmetric instances:
  - coordinate based: EUC_2D, CEIL_2D, ATT, GEO
  - explicit matrices: FULL_MATRIX, LOWER_DIAG_ROW
Each distance type follows the exact TSPLIB rounding rules so that tour lengths
match published optimal values. Unsupported types raise a clear error.

Usage:
    from serene_mh.problems.tsplib import load_tsplib
    problem = load_tsplib("data/raw/tsplib/berlin52.tsp")
"""

import math

import numpy as np

from .tsp import TSP, euclidean_matrix


# ----------------------------------------------------------- distance functions
def _euc_2d(a, b):
    return round(math.hypot(a[0] - b[0], a[1] - b[1]))


def _ceil_2d(a, b):
    return math.ceil(math.hypot(a[0] - b[0], a[1] - b[1]))


def _att(a, b):
    # TSPLIB "pseudo-Euclidean" distance (used by att48).
    r = math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10.0)
    t = round(r)
    return t + 1 if t < r else t


def _geo_radians(value):
    # TSPLIB interprets coordinates as degrees.minutes, not decimal degrees.
    deg = int(value)
    minutes = value - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def _geo_matrix(coords):
    n = len(coords)
    lat = [_geo_radians(c[0]) for c in coords]
    lon = [_geo_radians(c[1]) for c in coords]
    rrr = 6378.388
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            q1 = math.cos(lon[i] - lon[j])
            q2 = math.cos(lat[i] - lat[j])
            q3 = math.cos(lat[i] + lat[j])
            d = int(rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
            D[i, j] = D[j, i] = d
    return D


def _matrix_from_coords(coords, weight_type):
    if weight_type == "EUC_2D":
        return euclidean_matrix(coords, rounded=True)
    if weight_type == "CEIL_2D":
        return np.ceil(euclidean_matrix(coords, rounded=False))
    if weight_type == "GEO":
        return _geo_matrix(coords)
    # ATT: small enough to fill with a double loop
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _att(coords[i], coords[j])
    return D


# ----------------------------------------------------------------- explicit form
def _matrix_from_explicit(numbers, n, fmt):
    D = np.zeros((n, n))
    it = iter(numbers)
    if fmt == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                D[i, j] = next(it)
    elif fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(i + 1):
                v = next(it)
                D[i, j] = D[j, i] = v
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
    return D


# ------------------------------------------------------------------------ loader
def load_tsplib(path, start: str = "nearest") -> TSP:
    """Parse a .tsp file into a TSP problem."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh]

    header = {}
    section = None
    coords = {}
    explicit_numbers = []
    name = "tsplib"

    for ln in lines:
        if not ln or ln == "EOF":
            continue
        upper = ln.upper()
        # header line "KEY : VALUE"
        if ":" in ln and section is None:
            key, _, value = ln.partition(":")
            header[key.strip().upper()] = value.strip()
            continue
        if upper in ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"):
            section = "coords"
            continue
        if upper == "EDGE_WEIGHT_SECTION":
            section = "weights"
            continue
        if section == "coords":
            parts = ln.split()
            idx = int(parts[0]) - 1  # TSPLIB indexes from 1
            coords[idx] = (float(parts[1]), float(parts[2]))
        elif section == "weights":
            explicit_numbers.extend(float(x) for x in ln.split())

    n = int(header.get("DIMENSION"))
    weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    name = header.get("NAME", "tsplib")

    coord_array = None
    if coords:
        coord_array = np.array([coords[i] for i in range(n)], dtype=float)

    if weight_type == "EXPLICIT":
        fmt = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
        D = _matrix_from_explicit(explicit_numbers, n, fmt)
    elif weight_type in ("EUC_2D", "CEIL_2D", "ATT", "GEO"):
        if coord_array is None:
            raise ValueError(f"{name}: {weight_type} needs a NODE_COORD_SECTION")
        D = _matrix_from_coords(coord_array, weight_type)
    else:
        raise ValueError(f"{name}: unsupported EDGE_WEIGHT_TYPE '{weight_type}'")

    return TSP(D, name=name, start=start, coords=coord_array)
