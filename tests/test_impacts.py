"""LeSage–Pace direct / indirect / total impacts (§2.3)."""

import numpy as np
import pytest

from drinkingdata.estimation import impacts
from drinkingdata.spatial.weights import row_standardize


def _ring_W(n=4):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1.0
        W[i, (i - 1) % n] = 1.0
    return row_standardize(W)


def test_slx_reduces_to_beta_direct_no_rho():
    W = _ring_W()
    s = impacts.lesage_pace_impacts(W, beta=2.0, theta=0.5, rho=0.0)
    # With ρ=0, S = βI + θW; diagonal is β so direct == β exactly.
    assert s.direct == pytest.approx(2.0)
    # Indirect = mean row-sum of θW = θ (rows of standardized W sum to 1).
    assert s.indirect == pytest.approx(0.5)
    assert s.total == pytest.approx(2.5)


def test_sar_total_uses_leontief_multiplier():
    W = _ring_W()
    beta, rho = 1.0, 0.4
    s = impacts.lesage_pace_impacts(W, beta=beta, theta=0.0, rho=rho)
    # For row-standardized W, total impact = β / (1 - ρ).
    assert s.total == pytest.approx(beta / (1 - rho))
    assert s.direct < s.total           # spillovers are positive here
    assert s.indirect == pytest.approx(s.total - s.direct)


def test_simulated_cis_bracket_point():
    W = _ring_W()
    res = impacts.simulate_impact_cis(
        W, beta=1.0, se_beta=0.1, theta=0.3, se_theta=0.05,
        rho=0.2, se_rho=0.05, n_sim=500, seed=0,
    )
    for name in ("direct", "indirect", "total"):
        d = res[name]
        assert d["ci_low"] <= d["point"] <= d["ci_high"]
