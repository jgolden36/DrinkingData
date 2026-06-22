"""Indirect water = Σ_f ω_f^water · ΔÊ_f  (paper §2.4, outcome 4).

The *indirect* water footprint of a data center is the water embodied in the
incremental electricity it draws from the grid. Following the paper, it is

    IndirectWater_it = Σ_f  ω_f^water · ΔÊ_{f,it}

where ``ΔÊ_{f,it}`` is the **marginal** generation by fuel ``f`` attributable to
the incremental data-center load (an input from the energy companion paper) and
``ω_f^water`` is the water-consumption factor for fuel/cooling ``f`` from
Jin et al. (2019).

This module is pure pandas/numpy and unit tested. Uncertainty is propagated by
bootstrapping both inputs jointly (:func:`bootstrap_indirect_water`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def indirect_water(
    marginal_generation: pd.DataFrame,
    water_factors: pd.DataFrame,
    *,
    fuel_col: str = "fuel",
    generation_col: str = "delta_generation_mwh",
    omega_col: str = "omega_water",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute indirect water by joining ΔÊ_f with ω_f and summing over fuels.

    Parameters
    ----------
    marginal_generation
        Tidy frame with ``fuel_col``, ``generation_col`` and any grouping keys
        (e.g. ``unit_id``, ``date``).
    water_factors
        Tidy frame mapping ``fuel_col`` → ``omega_col`` (water per MWh).
    group_cols
        Keys to aggregate to (default: ``["unit_id", "date"]`` when present,
        else every non-fuel/non-generation column).

    Returns
    -------
    DataFrame with the grouping keys plus ``indirect_water`` (same volume units
    as ``ω`` × MWh).
    """
    if group_cols is None:
        preferred = [c for c in ("unit_id", "date") if c in marginal_generation.columns]
        group_cols = preferred or [
            c for c in marginal_generation.columns
            if c not in {fuel_col, generation_col}
        ]

    merged = marginal_generation.merge(
        water_factors[[fuel_col, omega_col]], on=fuel_col, how="left", validate="m:1"
    )
    missing = merged[omega_col].isna()
    if missing.any():
        bad = sorted(merged.loc[missing, fuel_col].unique())
        raise ValueError(f"No water factor (ω) for fuels: {bad}")

    merged["_iw"] = merged[generation_col] * merged[omega_col]
    out = (
        merged.groupby(group_cols, as_index=False)["_iw"].sum()
        .rename(columns={"_iw": "indirect_water"})
    )
    return out


def bootstrap_indirect_water(
    marginal_generation: pd.DataFrame,
    water_factors: pd.DataFrame,
    *,
    gen_se_col: Optional[str] = None,
    omega_se_col: Optional[str] = None,
    fuel_col: str = "fuel",
    generation_col: str = "delta_generation_mwh",
    omega_col: str = "omega_water",
    group_cols: Optional[list[str]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Bootstrap CIs for indirect water, perturbing ΔÊ_f and ω_f jointly (§2.4).

    Standard errors are taken from ``gen_se_col`` / ``omega_se_col`` when given
    (Gaussian draws); columns left as ``None`` are treated as fixed. Returns the
    point estimate and ``(1-alpha)`` percentile CI per group.
    """
    rng = np.random.default_rng(seed)
    point = indirect_water(
        marginal_generation, water_factors, fuel_col=fuel_col,
        generation_col=generation_col, omega_col=omega_col, group_cols=group_cols,
    ).set_index(_resolve_group_cols(marginal_generation, group_cols, fuel_col, generation_col))

    draws = []
    for _ in range(n_boot):
        gen = marginal_generation.copy()
        if gen_se_col and gen_se_col in gen.columns:
            gen[generation_col] = rng.normal(gen[generation_col], gen[gen_se_col].abs())
        wf = water_factors.copy()
        if omega_se_col and omega_se_col in wf.columns:
            wf[omega_col] = np.clip(
                rng.normal(wf[omega_col], wf[omega_se_col].abs()), 0, None
            )
        b = indirect_water(
            gen, wf, fuel_col=fuel_col, generation_col=generation_col,
            omega_col=omega_col, group_cols=group_cols,
        ).set_index(point.index.names)["indirect_water"]
        draws.append(b)

    mat = pd.concat(draws, axis=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    res = pd.DataFrame({
        "indirect_water": point["indirect_water"],
        "ci_low": np.percentile(mat.values, lo, axis=1),
        "ci_high": np.percentile(mat.values, hi, axis=1),
    })
    return res.reset_index()


def _resolve_group_cols(mg, group_cols, fuel_col, generation_col):
    if group_cols is not None:
        return group_cols
    preferred = [c for c in ("unit_id", "date") if c in mg.columns]
    return preferred or [c for c in mg.columns if c not in {fuel_col, generation_col}]
