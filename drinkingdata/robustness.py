"""Robustness battery (paper §2.6 / "Validating Assumptions").

Scaffolding for the validation suite that every headline estimate must survive:

* **Pre-trends / parallel-trends** — joint test that pre-period event-study
  coefficients are zero, plus Rambachan–Roth (2023) sensitivity bounds.
* **Placebo tests** — randomized treatment dates, randomized treated units, and
  a non-AI data-center placebo (DCs with no hyperscaler/AI classification).
* **Spatial-weights sensitivity** — re-estimate across the ``W`` variants in
  :mod:`drinkingdata.spatial.weights` (k-NN, inverse-distance, flow-only,
  basin-only).
* **Treatment-definition sensitivity** — baseline → confirmed locations →
  verified dates.
* **Callaway–Sant'Anna** group-time ATTs.
* **Anomaly detection** — STL + ESD clustered around release dates.

The pure-statistical pieces (pre-trend Wald test, placebo resampling) are
implemented; the externally-modeled pieces (Rambachan–Roth, Callaway–Sant'Anna,
STL+ESD) are documented hooks pending their library wiring.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def pretrend_test(result, pre_event_prefixes: List[str]) -> dict:
    """Joint Wald test that pre-treatment event-study coefficients are zero.

    Parameters
    ----------
    result
        A fitted ``linearmodels`` event-study result (see
        :func:`drinkingdata.estimation.eventstudy.fit_event_study`).
    pre_event_prefixes
        The names of the pre-period event-time dummies, e.g.
        ``["ET_-6", "ET_-5", ..., "ET_-2"]`` (omit the normalized period).

    Returns
    -------
    dict with the Wald statistic, p-value and the tested names. A large p-value
    is consistent with parallel pre-trends.
    """
    present = [c for c in pre_event_prefixes if c in result.params.index]
    if not present:
        raise ValueError("None of the requested pre-period coefficients are in the model.")
    test = result.wald_test(formula=" = ".join(present) + " = 0") if False else None

    # linearmodels exposes wald_test via restriction matrices; build R explicitly.
    params = result.params
    R = pd.DataFrame(0.0, index=present, columns=params.index)
    for name in present:
        R.loc[name, name] = 1.0
    wt = result.wald_test(R.values, np.zeros(len(present)))
    return {"stat": float(wt.stat), "pval": float(wt.pval), "tested": present}


def placebo_randomize_dates(
    panel: pd.DataFrame,
    estimate_fn: Callable[[pd.DataFrame], float],
    *,
    cohort_col: str = "cohort",
    unit_col: str = "unit_id",
    n_iter: int = 500,
    seed: Optional[int] = None,
) -> dict:
    """Randomization-inference placebo: shuffle cohorts across units (§2.6).

    ``estimate_fn`` takes a (possibly re-cohorted) panel and returns the scalar
    statistic of interest (e.g. the direct ATT ``β``). The placebo p-value is the
    share of permutations with an effect at least as extreme as the observed one.
    """
    rng = np.random.default_rng(seed)
    observed = estimate_fn(panel)

    cohorts = panel.drop_duplicates(unit_col).set_index(unit_col)[cohort_col]
    units = cohorts.index.to_numpy()
    values = cohorts.to_numpy()

    null = np.empty(n_iter)
    for i in range(n_iter):
        permuted = pd.Series(rng.permutation(values), index=units)
        fake = panel.copy()
        fake[cohort_col] = fake[unit_col].map(permuted)
        null[i] = estimate_fn(fake)

    pval = float((np.abs(null) >= abs(observed)).mean())
    return {"observed": float(observed), "placebo_pval": pval, "null": null}


def spatial_weights_sensitivity(
    panel: pd.DataFrame,
    weight_specs: dict,
    estimate_fn: Callable[[pd.DataFrame, object], dict],
) -> pd.DataFrame:
    """Re-estimate across ``W`` variants and tabulate the effects (§2.6).

    ``weight_specs`` maps a label → :class:`~drinkingdata.spatial.weights.SpatialWeights`;
    ``estimate_fn(panel, W)`` returns a dict of statistics (e.g. ``{"beta":…,
    "theta":…}``). Returns one row per spec for side-by-side comparison.
    """
    rows = []
    for label, W in weight_specs.items():
        stats = estimate_fn(panel, W)
        rows.append({"weights": label, **stats})
    return pd.DataFrame(rows)
