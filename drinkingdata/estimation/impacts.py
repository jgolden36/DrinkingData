"""LeSage–Pace direct / indirect / total impacts for spatial models (§2.3).

In a spatial model with a spatial lag of the dependent variable,

    y = ρ W y + β D + θ W D + ... + ε,

the marginal effect of a unit's own treatment is **not** ``β``: it propagates
through the system via ``(I − ρW)^{-1}``. The partial-derivatives matrix for the
treatment is

    S(W) = (I − ρW)^{-1} (β I + θ W).

LeSage & Pace summarize ``S(W)`` as:

* **Direct**  = average of the diagonal of ``S(W)``   (own-unit effect, ATT ``β``)
* **Total**   = average of the row sums of ``S(W)``
* **Indirect**= Total − Direct                         (spillover, ``θ``)

This is the central spillover deliverable. The function is pure-numpy and unit
tested; for SLX (ρ = 0) it reduces to Direct = β, Indirect = mean row-sum of θW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImpactSummary:
    direct: float
    indirect: float
    total: float

    def as_dict(self) -> dict:
        return {"direct": self.direct, "indirect": self.indirect, "total": self.total}


def impact_matrix(W: np.ndarray, beta: float, *, theta: float = 0.0,
                  rho: float = 0.0) -> np.ndarray:
    """Return the partial-derivatives matrix ``S(W) = (I-ρW)^{-1}(βI + θW)``."""
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError("W must be square")
    I = np.eye(n)
    A_inv = np.linalg.inv(I - rho * W) if rho else I
    return A_inv @ (beta * I + theta * W)


def lesage_pace_impacts(W: np.ndarray, beta: float, *, theta: float = 0.0,
                        rho: float = 0.0) -> ImpactSummary:
    """Direct / indirect / total impacts of the treatment (point estimates)."""
    S = impact_matrix(W, beta, theta=theta, rho=rho)
    n = S.shape[0]
    direct = float(np.trace(S) / n)
    total = float(S.sum() / n)
    return ImpactSummary(direct=direct, indirect=total - direct, total=total)


def simulate_impact_cis(
    W: np.ndarray,
    *,
    beta: float,
    se_beta: float,
    theta: float = 0.0,
    se_theta: float = 0.0,
    rho: float = 0.0,
    se_rho: float = 0.0,
    cov: Optional[np.ndarray] = None,
    n_sim: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> dict:
    """Simulated CIs for the impacts (LeSage–Pace Monte-Carlo, §2.3).

    Draws ``(β, θ, ρ)`` from a normal centered at the estimates. Pass a 3×3
    ``cov`` to respect parameter correlation; otherwise independent draws using
    the supplied standard errors. Returns point estimates and ``(1-alpha)`` CIs
    for direct/indirect/total.
    """
    rng = np.random.default_rng(seed)
    mean = np.array([beta, theta, rho], dtype=float)
    if cov is None:
        cov = np.diag(np.array([se_beta, se_theta, se_rho]) ** 2)
    draws = rng.multivariate_normal(mean, cov, size=n_sim)

    rows = np.empty((n_sim, 3))
    for i, (b, th, r) in enumerate(draws):
        s = lesage_pace_impacts(W, b, theta=th, rho=r)
        rows[i] = (s.direct, s.indirect, s.total)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    point = lesage_pace_impacts(W, beta, theta=theta, rho=rho)
    names = ("direct", "indirect", "total")
    return {
        name: {
            "point": getattr(point, name),
            "ci_low": float(np.percentile(rows[:, j], lo)),
            "ci_high": float(np.percentile(rows[:, j], hi)),
        }
        for j, name in enumerate(names)
    }
