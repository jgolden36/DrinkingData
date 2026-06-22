"""Staggered spatial event study: cohort-specific {β_k, θ_k} (paper §2.3).

Builds the stacked, cohort-by-event-time design (Sun–Abraham style) used to
trace dynamic direct (``β_k``) and spillover (``θ_k``) effects over event time
``k``, with the release date treated as a *noisy proxy* for the unobserved
training window (§2.1). The pooled (non-cohort) event-time design is built and
fit here; the full cohort-interaction matrix is provided for the interaction
specification.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.panel import build_event_time, first_treatment_date


def build_event_dummies(
    panel: pd.DataFrame,
    event_time: pd.Series,
    *,
    event_min: int,
    event_max: int,
    omit: int = -1,
    unit_col: str = "unit_id",
    time_col: str = "date",
    bin_endpoints: bool = True,
) -> pd.DataFrame:
    """Pooled event-time dummies ``1{k_it = k}`` for k in [min, max] \\ {omit}.

    When ``bin_endpoints`` is set, periods beyond the window are accumulated into
    the endpoint bins (``ET_<min>`` / ``ET_<max>``) so the design stays balanced.
    Returns a DataFrame indexed by (unit, time).
    """
    et = event_time.copy()
    if bin_endpoints:
        et = et.clip(lower=event_min, upper=event_max)

    idx = panel.set_index([unit_col, time_col]).index
    et = et.reindex(idx)
    cols = {}
    for k in range(event_min, event_max + 1):
        if k == omit:
            continue
        cols[f"ET_{k}"] = (et == k).astype(float)
    return pd.DataFrame(cols, index=idx)


def fit_event_study(
    panel: pd.DataFrame,
    *,
    outcome: str,
    event_time: pd.Series,
    event_min: int,
    event_max: int,
    omit: int = -1,
    spillover_col: Optional[str] = None,
    control_cols: Optional[List[str]] = None,
    unit_col: str = "unit_id",
    time_col: str = "date",
    cluster_cols: Optional[List[str]] = None,
):
    """Fit the pooled event study (optionally with a spillover dose term).

    ``spillover_col`` (e.g. ``"W_D_it"``) lets you include the contemporaneous
    spatial spillover dose alongside the dynamic own-effect path; the full
    cohort-interacted ``{β_k, θ_k}`` design uses :func:`build_cohort_interactions`.
    """
    from linearmodels.panel import PanelOLS  # lazy

    dummies = build_event_dummies(
        panel, event_time, event_min=event_min, event_max=event_max,
        omit=omit, unit_col=unit_col, time_col=time_col,
    )
    df = panel.set_index([unit_col, time_col]).sort_index()
    X = dummies
    if spillover_col:
        X = X.join(df[[spillover_col]].astype(float))
    if control_cols:
        X = X.join(df[control_cols].astype(float))

    y = df[outcome].astype(float)
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    if cluster_cols:
        return mod.fit(cov_type="clustered", clusters=df[cluster_cols], debiased=True)
    return mod.fit(cov_type="clustered", cluster_entity=True, debiased=True)


def build_cohort_interactions(
    panel: pd.DataFrame,
    event_time: pd.Series,
    cohorts: pd.Series,
    *,
    event_min: int,
    event_max: int,
    omit: int = -1,
    unit_col: str = "unit_id",
    time_col: str = "date",
    cohort_freq: str = "Y",
) -> pd.DataFrame:
    """Cohort × event-time interaction dummies ``1{cohort=c}·1{k=k}`` (§2.3).

    The Sun–Abraham design that lets you aggregate to clean ``{β_k}`` free of
    contamination from already-treated cohorts. Cohorts are coarsened to
    ``cohort_freq`` (default yearly) to keep the design from exploding.
    """
    idx = panel.set_index([unit_col, time_col]).index
    frame = pd.DataFrame(index=idx)
    frame["et"] = event_time.reindex(idx).values

    cohort_label = (
        pd.to_datetime(cohorts).dt.to_period(cohort_freq).astype(str)
    )
    units = idx.get_level_values(unit_col)
    frame["cohort"] = [cohort_label.get(u, None) for u in units]

    out = pd.DataFrame(index=idx)
    valid_cohorts = [c for c in pd.unique(frame["cohort"]) if c is not None]
    for c in valid_cohorts:
        for k in range(event_min, event_max + 1):
            if k == omit:
                continue
            col = f"C{c}_ET{k}"
            out[col] = ((frame["cohort"] == c) & (frame["et"] == k)).astype(float)
    return out
