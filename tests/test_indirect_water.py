"""Indirect water = Σ_f ω_f · ΔÊ_f (§2.4)."""

import numpy as np
import pandas as pd
import pytest

from drinkingdata.indirect import water


def test_indirect_water_sums_over_fuels():
    gen = pd.DataFrame({
        "unit_id": ["a", "a", "b"],
        "date": ["2023-01", "2023-01", "2023-01"],
        "fuel": ["gas", "coal", "gas"],
        "delta_generation_mwh": [100.0, 50.0, 200.0],
    })
    factors = pd.DataFrame({"fuel": ["gas", "coal"], "omega_water": [2.0, 5.0]})
    out = water.indirect_water(gen, factors).set_index(["unit_id", "date"])
    # a: 100*2 + 50*5 = 450 ; b: 200*2 = 400
    assert out.loc[("a", "2023-01"), "indirect_water"] == pytest.approx(450.0)
    assert out.loc[("b", "2023-01"), "indirect_water"] == pytest.approx(400.0)


def test_missing_factor_raises():
    gen = pd.DataFrame({
        "unit_id": ["a"], "date": ["2023-01"],
        "fuel": ["nuclear"], "delta_generation_mwh": [10.0],
    })
    factors = pd.DataFrame({"fuel": ["gas"], "omega_water": [2.0]})
    with pytest.raises(ValueError):
        water.indirect_water(gen, factors)


def test_bootstrap_brackets_point_estimate():
    gen = pd.DataFrame({
        "unit_id": ["a", "a"],
        "date": ["2023-01", "2023-01"],
        "fuel": ["gas", "coal"],
        "delta_generation_mwh": [100.0, 50.0],
        "gen_se": [10.0, 5.0],
    })
    factors = pd.DataFrame({
        "fuel": ["gas", "coal"],
        "omega_water": [2.0, 5.0],
        "omega_se": [0.2, 0.5],
    })
    res = water.bootstrap_indirect_water(
        gen, factors, gen_se_col="gen_se", omega_se_col="omega_se",
        n_boot=400, seed=1,
    )
    row = res.iloc[0]
    assert row["ci_low"] <= row["indirect_water"] <= row["ci_high"]
    assert row["ci_low"] < row["ci_high"]
