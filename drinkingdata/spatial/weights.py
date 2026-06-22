"""Spatial weights ``W`` from hydrologic / institutional connectivity (§2.2).

The paper builds ``W`` from **hydrologic/hydraulic connectivity**, not Euclidean
distance: a shared HUC-8 boundary, an NHDPlus downstream relationship, or a
shared managed-groundwater sub-basin (SGMA / GCD). Row-standardized baseline,
with k-NN / inverse-distance / flow-only / basin-only variants for the
robustness battery (§2.6).

This module provides a small, dependency-light :class:`SpatialWeights` value
object (just a dense matrix + unit ordering) and constructors for each rule. The
contiguity constructor imports ``libpysal`` lazily; everything else is numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def row_standardize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-standardize a weights matrix (rows that sum to ~0 are left at 0)."""
    W = np.asarray(W, dtype=float)
    row_sums = W.sum(axis=1, keepdims=True)
    safe = np.where(row_sums < eps, 1.0, row_sums)
    out = W / safe
    out[(row_sums < eps).ravel()] = 0.0
    return out


@dataclass
class SpatialWeights:
    """A spatial weights matrix tied to an explicit unit ordering.

    Attributes
    ----------
    W : (n, n) ndarray
        ``W[i, j]`` is the weight unit ``order[i]`` places on unit ``order[j]``
        (exposure of ``i`` to ``j``). Typically row-standardized.
    order : list
        Unit ids giving the row/column ordering of ``W``.
    """

    W: np.ndarray
    order: List

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        self.order = list(self.order)
        n = len(self.order)
        if self.W.shape != (n, n):
            raise ValueError(f"W shape {self.W.shape} != ({n}, {n})")
        self._pos = {u: i for i, u in enumerate(self.order)}

    @property
    def n(self) -> int:
        return len(self.order)

    def standardized(self) -> "SpatialWeights":
        return SpatialWeights(row_standardize(self.W), self.order)

    def lag(self, values: pd.Series) -> pd.Series:
        """Spatial lag ``W x`` for one cross-section indexed by unit id."""
        x = np.array([float(values.get(u, 0.0)) for u in self.order])
        return pd.Series(self.W @ x, index=self.order)

    def lag_panel(self, dose: pd.Series, unit_level: int = 0, time_level: int = 1) -> pd.Series:
        """Spatial lag of a panel dose: ``(W D)_it``, applied period-by-period.

        ``dose`` must be indexed by a (unit, time) MultiIndex. Returns a series on
        the same index. This is the spillover exposure ``WD`` of paper §2.3.
        """
        idx = dose.index
        units = np.asarray(idx.get_level_values(unit_level))
        times = np.asarray(idx.get_level_values(time_level))
        dose_vals = np.asarray(dose.values, dtype=float)
        pos = self._pos

        result = np.zeros(len(idx), dtype=float)
        for t in pd.unique(times):
            mask = times == t
            u_t = units[mask]
            d_vec = np.zeros(self.n)
            for u, v in zip(u_t, dose_vals[mask]):
                if u in pos:
                    d_vec[pos[u]] = v
            wd = self.W @ d_vec
            result[mask] = [wd[pos[u]] if u in pos else 0.0 for u in u_t]
        return pd.Series(result, index=idx, name="WD_it")


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def from_edges(
    units: Sequence,
    edges: pd.DataFrame,
    *,
    src_col: str = "src_unit_id",
    dst_col: str = "dst_unit_id",
    weight_col: Optional[str] = None,
    directed: bool = True,
    standardize: bool = True,
) -> SpatialWeights:
    """Build ``W`` from an edge list (hydrologic network or institutional links).

    The canonical hydrologic case is an NHDPlus upstream→downstream edge list:
    each row ``(src, dst)`` means flow goes from ``src`` to ``dst``, so ``dst`` is
    exposed to ``src``. Set ``directed=False`` for symmetric links (shared HUC-8
    boundary, same SGMA/GCD basin).
    """
    order = list(dict.fromkeys(units))
    pos = {u: i for i, u in enumerate(order)}
    n = len(order)
    W = np.zeros((n, n))

    e = edges.dropna(subset=[src_col, dst_col])
    for _, r in e.iterrows():
        s, d = r[src_col], r[dst_col]
        if s not in pos or d not in pos or s == d:
            continue
        w = float(r[weight_col]) if weight_col else 1.0
        W[pos[d], pos[s]] += w   # exposure of downstream d to upstream s
        if not directed:
            W[pos[s], pos[d]] += w

    sw = SpatialWeights(W, order)
    return sw.standardized() if standardize else sw


def from_groups(
    units: pd.DataFrame,
    *,
    unit_col: str = "unit_id",
    group_col: str = "basin_id",
    standardize: bool = True,
) -> SpatialWeights:
    """Institutional ``W``: complete graph within each shared group.

    Use for "same managed-groundwater sub-basin" (SGMA/GCD) or "same water
    supplier" spillovers. Units sharing ``group_col`` are mutually connected.
    """
    u = units[[unit_col, group_col]].dropna()
    order = list(dict.fromkeys(u[unit_col]))
    pos = {x: i for i, x in enumerate(order)}
    n = len(order)
    W = np.zeros((n, n))
    for _, grp in u.groupby(group_col):
        idxs = [pos[x] for x in grp[unit_col].unique() if x in pos]
        for i in idxs:
            for j in idxs:
                if i != j:
                    W[i, j] = 1.0
    sw = SpatialWeights(W, order)
    return sw.standardized() if standardize else sw


def from_knn(
    units: pd.DataFrame,
    *,
    unit_col: str = "unit_id",
    x_col: str = "x",
    y_col: str = "y",
    k: int = 6,
    standardize: bool = True,
) -> SpatialWeights:
    """k-nearest-neighbour ``W`` on centroid coordinates (robustness, §2.2)."""
    u = units[[unit_col, x_col, y_col]].dropna().reset_index(drop=True)
    order = u[unit_col].tolist()
    coords = u[[x_col, y_col]].to_numpy(dtype=float)
    n = len(order)
    k = min(k, n - 1)
    W = np.zeros((n, n))
    # pairwise squared distances
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    for i in range(n):
        nn = np.argpartition(d2[i], k)[:k]
        W[i, nn] = 1.0
    sw = SpatialWeights(W, order)
    return sw.standardized() if standardize else sw


def from_inverse_distance(
    units: pd.DataFrame,
    *,
    unit_col: str = "unit_id",
    x_col: str = "x",
    y_col: str = "y",
    cutoff: Optional[float] = None,
    power: float = 1.0,
    standardize: bool = True,
) -> SpatialWeights:
    """Inverse-distance ``W`` with optional distance cutoff (robustness, §2.2)."""
    u = units[[unit_col, x_col, y_col]].dropna().reset_index(drop=True)
    order = u[unit_col].tolist()
    coords = u[[x_col, y_col]].to_numpy(dtype=float)
    n = len(order)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    with np.errstate(divide="ignore"):
        W = np.where(dist > 0, dist ** (-power), 0.0)
    np.fill_diagonal(W, 0.0)
    if cutoff is not None:
        W[dist > cutoff] = 0.0
    sw = SpatialWeights(W, order)
    return sw.standardized() if standardize else sw


def from_contiguity(units_gdf, *, unit_col: str = "unit_id", kind: str = "queen",
                    standardize: bool = True) -> SpatialWeights:
    """Polygon contiguity ``W`` (Queen/Rook) via ``libpysal`` (fallback only).

    Hydrologic connectivity (``from_edges``) is preferred for the analytic path;
    contiguity is a fallback when a flow network is unavailable.
    """
    from libpysal.weights import Queen, Rook  # lazy heavy dep

    g = units_gdf[[unit_col, "geometry"]].dropna().reset_index(drop=True)
    order = g[unit_col].tolist()
    builder = {"queen": Queen, "rook": Rook}[kind]
    w = builder.from_dataframe(g, use_index=False)
    n = len(order)
    W = np.zeros((n, n))
    for i, neigh in w.neighbors.items():
        for j in neigh:
            W[i, j] = 1.0
    sw = SpatialWeights(W, order)
    return sw.standardized() if standardize else sw
