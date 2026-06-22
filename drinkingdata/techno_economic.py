"""Techno-economic benchmark and the econ-vs-engineering gap (paper §2.5).

Two pieces feed the central deliverable — comparing the *causal* estimate with
the *engineering* prediction:

1. **TE-Direct benchmark.** The literature-derived direct water use of a facility:

       TE-Direct_j = WUE_j · φ_j · Capacity_j · Util_j · Δt,     φ_j = 1 / PUE_j

   summed over facilities ``j`` (WUE = water-use efficiency L/kWh, φ = inverse
   power-usage-effectiveness, Capacity in kW, Util the utilization fraction, Δt
   the period length in hours).

2. **Gap decomposition.** The headline result decomposes

       ΔΔ = Δ^econ − Δ^te

   into interpretable channels: a **spillover** term (present in the econometric
   estimate, absent from the per-facility engineering number), a **marginal-vs-
   average indirect** term (the econometric estimate prices water at the marginal
   generator, the techno-economic at the average), and a residual **direct-effect
   gap**. Reported by sector and Aqueduct water-stress class.

Pure-numpy/pandas; unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def te_direct(
    facilities: pd.DataFrame,
    *,
    wue_col: str = "wue",
    pue_col: str = "pue",
    capacity_col: str = "capacity_kw",
    util_col: str = "utilization",
    hours_col: str = "hours",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Techno-economic direct water by facility, aggregated over ``group_cols``.

    Each facility's contribution is ``WUE · (1/PUE) · Capacity · Util · hours``.
    Returns the grouping keys (default: ``unit_id`` if present, else a single
    total row) plus ``te_direct_water``.
    """
    df = facilities.copy()
    phi = 1.0 / df[pue_col]
    df["_te"] = df[wue_col] * phi * df[capacity_col] * df[util_col] * df[hours_col]

    if group_cols is None:
        group_cols = [c for c in ("unit_id",) if c in df.columns]

    if not group_cols:
        return pd.DataFrame({"te_direct_water": [df["_te"].sum()]})
    return (
        df.groupby(group_cols, as_index=False)["_te"].sum()
        .rename(columns={"_te": "te_direct_water"})
    )


@dataclass
class GapDecomposition:
    """Decomposition of ΔΔ = Δ^econ − Δ^te into channels (§2.5)."""

    delta_econ: float
    delta_te: float
    spillover: float
    marginal_vs_average_indirect: float
    direct_effect_gap: float

    @property
    def total_gap(self) -> float:
        return self.delta_econ - self.delta_te

    def check(self, tol: float = 1e-6) -> bool:
        """Verify the channels sum to the total gap (accounting identity)."""
        recomposed = (
            self.spillover
            + self.marginal_vs_average_indirect
            + self.direct_effect_gap
        )
        return abs(recomposed - self.total_gap) <= tol

    def as_dict(self) -> dict:
        return {
            "delta_econ": self.delta_econ,
            "delta_te": self.delta_te,
            "total_gap": self.total_gap,
            "spillover": self.spillover,
            "marginal_vs_average_indirect": self.marginal_vs_average_indirect,
            "direct_effect_gap": self.direct_effect_gap,
        }


def decompose_gap(
    *,
    delta_econ: float,
    delta_te: float,
    spillover: float,
    indirect_marginal: float,
    indirect_average: float,
) -> GapDecomposition:
    """Build the gap decomposition (§2.5).

    The three channels:

    * ``spillover`` — the indirect/spillover ATT ``θ`` captured econometrically
      but absent from the per-facility techno-economic number.
    * ``marginal_vs_average_indirect = indirect_marginal − indirect_average`` —
      the econometric estimate values grid water at the *marginal* generator,
      the techno-economic at the *average* generator.
    * ``direct_effect_gap`` — the residual, i.e. the remaining difference in the
      direct own-unit effect after removing the two channels above. Defined so
      the channels exactly recompose ``ΔΔ`` (accounting identity).
    """
    total_gap = delta_econ - delta_te
    marginal_vs_average = indirect_marginal - indirect_average
    direct_effect_gap = total_gap - spillover - marginal_vs_average
    return GapDecomposition(
        delta_econ=delta_econ,
        delta_te=delta_te,
        spillover=spillover,
        marginal_vs_average_indirect=marginal_vs_average,
        direct_effect_gap=direct_effect_gap,
    )


def decompose_gap_by_group(
    rows: pd.DataFrame,
    *,
    group_cols: list[str],
    delta_econ_col: str = "delta_econ",
    delta_te_col: str = "delta_te",
    spillover_col: str = "spillover",
    indirect_marginal_col: str = "indirect_marginal",
    indirect_average_col: str = "indirect_average",
) -> pd.DataFrame:
    """Vectorized gap decomposition per sector / water-stress class (§2.5)."""
    out = rows[group_cols].copy()
    out["total_gap"] = rows[delta_econ_col] - rows[delta_te_col]
    out["spillover"] = rows[spillover_col]
    out["marginal_vs_average_indirect"] = (
        rows[indirect_marginal_col] - rows[indirect_average_col]
    )
    out["direct_effect_gap"] = (
        out["total_gap"] - out["spillover"] - out["marginal_vs_average_indirect"]
    )
    return out
