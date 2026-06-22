"""Techno-economic benchmark and gap decomposition (§2.5)."""

import pandas as pd
import pytest

from drinkingdata import techno_economic as te


def test_te_direct_formula():
    fac = pd.DataFrame({
        "unit_id": ["a"],
        "wue": [1.8],          # L/kWh
        "pue": [1.2],          # φ = 1/1.2
        "capacity_kw": [1000.0],
        "utilization": [0.5],
        "hours": [720.0],
    })
    out = te.te_direct(fac)
    expected = 1.8 * (1 / 1.2) * 1000.0 * 0.5 * 720.0
    assert out.loc[0, "te_direct_water"] == pytest.approx(expected)


def test_decompose_gap_is_an_identity():
    d = te.decompose_gap(
        delta_econ=100.0,
        delta_te=60.0,
        spillover=25.0,
        indirect_marginal=20.0,
        indirect_average=12.0,
    )
    assert d.total_gap == pytest.approx(40.0)
    assert d.marginal_vs_average_indirect == pytest.approx(8.0)
    # residual closes the identity: 40 = 25 + 8 + direct_effect_gap
    assert d.direct_effect_gap == pytest.approx(7.0)
    assert d.check()


def test_decompose_gap_by_group_matches_scalar():
    rows = pd.DataFrame({
        "sector": ["municipal", "mining"],
        "delta_econ": [100.0, 50.0],
        "delta_te": [60.0, 40.0],
        "spillover": [25.0, 5.0],
        "indirect_marginal": [20.0, 10.0],
        "indirect_average": [12.0, 8.0],
    })
    out = te.decompose_gap_by_group(rows, group_cols=["sector"])
    muni = out.set_index("sector").loc["municipal"]
    assert muni["total_gap"] == pytest.approx(40.0)
    assert muni["direct_effect_gap"] == pytest.approx(7.0)
