"""Spatial DiD estimators: SLX → SAR → SDM (paper §2.3).

Increasing generality, all on a two-way fixed-effects DiD panel:

* **SLX-DiD** (workhorse): ``Y = α_i + δ_t + βD + θ(WD) + Xγ + (WX)η + ε``.
  Estimable by OLS/PanelOLS — just add the spatially-lagged regressors. ``β`` is
  the direct ATT and ``θ`` the spillover ATT.
* **SAR-DiD**: adds ``ρ(WY)``. Requires ML/IV (``spreg``); report LeSage–Pace
  impacts (:mod:`drinkingdata.estimation.impacts`), not raw coefficients.
* **SDM-DiD** (preferred default): adds both ``ρ(WY)`` and ``(WX)η``; nests SLX
  and SAR. Run LR/Wald/common-factor tests to discriminate SLX/SAR/SEM/SDM.

SLX is implemented here on top of the non-spatial DiD by materializing spatial
lags of the dose and controls. SAR/SDM are functional skeletons that delegate to
``spreg`` (imported lazily) — wired once panel + ``W`` are real.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..spatial.weights import SpatialWeights
from . import did


def add_spatial_lags(
    panel: pd.DataFrame,
    W: SpatialWeights,
    cols: List[str],
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    prefix: str = "W_",
) -> pd.DataFrame:
    """Append spatial lags ``W·x`` of ``cols`` as new ``prefix+col`` columns."""
    out = panel.copy()
    indexed = out.set_index([unit_col, time_col])
    for c in cols:
        lagged = W.lag_panel(indexed[c].astype(float))
        out[f"{prefix}{c}"] = lagged.values
    return out


def fit_slx_did(
    panel: pd.DataFrame,
    W: SpatialWeights,
    *,
    outcome: str,
    dose_col: str = "D_it",
    control_cols: Optional[List[str]] = None,
    lag_controls: bool = True,
    unit_col: str = "unit_id",
    time_col: str = "date",
    cluster_cols: Optional[List[str]] = None,
):
    """Fit the SLX-DiD: direct ``β`` (dose) + spillover ``θ`` (W·dose).

    Returns the fitted ``PanelOLS`` result. The spillover ATT is the coefficient
    on ``W_<dose_col>``; the direct ATT is the coefficient on ``dose_col``.
    """
    controls = control_cols or []
    lag_cols = [dose_col] + (controls if lag_controls else [])
    augmented = add_spatial_lags(
        panel, W, lag_cols, unit_col=unit_col, time_col=time_col
    )
    treatment = [dose_col, f"W_{dose_col}"]
    extra_controls = controls + ([f"W_{c}" for c in controls] if lag_controls else [])
    return did.fit_panel_did(
        augmented,
        outcome=outcome,
        treatment_cols=treatment,
        control_cols=extra_controls,
        unit_col=unit_col,
        time_col=time_col,
        cluster_cols=cluster_cols,
    )


def fit_sdm_did(
    panel: pd.DataFrame,
    W: SpatialWeights,
    *,
    outcome: str,
    dose_col: str = "D_it",
    control_cols: Optional[List[str]] = None,
    model: str = "sdm",
    unit_col: str = "unit_id",
    time_col: str = "date",
):
    """Fit SAR/SDM-DiD via ``spreg`` (ML), within-transformed for two-way FE.

    SKELETON. Wiring notes:

    1. Within-transform ``Y`` and the regressors to absorb unit & time FE
       (demean by unit and by period), since ``spreg.ML_Lag`` / ``GM_Lag`` do not
       carry panel fixed effects natively.
    2. Build the ``libpysal`` ``W`` from ``SpatialWeights`` (full sample or, for a
       balanced panel, block-diagonal across periods).
    3. ``model='sar'`` → ``spreg.ML_Lag``; ``model='sdm'`` → ``ML_Lag`` with the
       spatially-lagged regressors ``WX`` appended (``add_spatial_lags``).
    4. Convert coefficients to LeSage–Pace impacts with
       :func:`drinkingdata.estimation.impacts.lesage_pace_impacts` and simulate
       CIs — report those, not the raw ``ρ/β/θ``.
    """
    raise NotImplementedError(
        "SAR/SDM ML estimation is scaffolded but not yet wired. See the docstring "
        "for the within-transform + spreg recipe; SLX-DiD (fit_slx_did) is "
        "available now and recovers β/θ directly."
    )
