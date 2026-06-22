"""Non-spatial difference-in-differences (paper §2.3 baseline).

Thin wrapper over ``linearmodels.PanelOLS`` with two-way fixed effects and
clustered standard errors — the panel-DiD that ``TexasWaterPanelRegression.py``
implements for counties, generalized to any unit. The spatial SLX/SAR/SDM
extensions live in :mod:`drinkingdata.estimation.spatial_did`.

``linearmodels`` is imported lazily so the package imports without it.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def fit_panel_did(
    panel: pd.DataFrame,
    *,
    outcome: str,
    treatment_cols: List[str],
    control_cols: Optional[List[str]] = None,
    unit_col: str = "unit_id",
    time_col: str = "date",
    cluster_cols: Optional[List[str]] = None,
    entity_effects: bool = True,
    time_effects: bool = True,
):
    """Fit ``Y = α_i + δ_t + βD + Xγ + ε`` with clustered SEs.

    Parameters
    ----------
    panel
        Long DataFrame containing ``unit_col``, ``time_col``, ``outcome``,
        the treatment column(s) and any controls.
    treatment_cols
        Treatment regressors, e.g. ``["D_it"]`` (dose) or interaction dummies.
    cluster_cols
        Columns to cluster SEs on (default: cluster on the spatial unit, §2.3).

    Returns
    -------
    linearmodels.panel.results.PanelEffectsResults
    """
    from linearmodels.panel import PanelOLS  # lazy

    controls = control_cols or []
    df = panel.set_index([unit_col, time_col]).sort_index()
    y = df[outcome].astype(float)
    X = df[treatment_cols + controls].astype(float)

    mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)

    if cluster_cols:
        clusters = df[cluster_cols]
        return mod.fit(cov_type="clustered", clusters=clusters, debiased=True)
    # Default: cluster on the spatial unit (entity).
    return mod.fit(cov_type="clustered", cluster_entity=True, debiased=True)
