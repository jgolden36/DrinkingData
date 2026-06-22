"""Panel construction: dose, cohorts, event time (§2.1, §2.3)."""

import numpy as np
import pandas as pd
import pytest

from drinkingdata.data import panel


def _panel_skeleton(units, months):
    rows = [(u, m) for u in units for m in months]
    return pd.DataFrame(rows, columns=["unit_id", "date"])


def test_transform_capacity_variants():
    s = pd.Series([0.0, 3.0, -1.0])
    np.testing.assert_allclose(panel.transform_capacity(s, "level"), [0, 3, 0])
    np.testing.assert_allclose(panel.transform_capacity(s, "log1p"), np.log1p([0, 3, 0]))
    np.testing.assert_allclose(panel.transform_capacity(s, "sqrt"), [0, np.sqrt(3), 0])
    with pytest.raises(ValueError):
        panel.transform_capacity(s, "nope")


def test_build_treatment_dose_is_cumulative():
    months = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    pnl = _panel_skeleton(["a", "b"], months)
    dc = pd.DataFrame({
        "unit_id": ["a", "a"],
        "open_date": pd.to_datetime(["2023-02-01", "2023-03-01"]),
        "capacity_mw": [100.0, 50.0],
    })
    dose = panel.build_treatment_dose(pnl, dc, cap_transform="level")
    # a: 0 in Jan, 100 in Feb, 150 in Mar; b: always 0.
    assert dose.loc[("a", pd.Timestamp("2023-01-01"))] == pytest.approx(0.0)
    assert dose.loc[("a", pd.Timestamp("2023-02-01"))] == pytest.approx(100.0)
    assert dose.loc[("a", pd.Timestamp("2023-03-01"))] == pytest.approx(150.0)
    assert dose.loc[("b", pd.Timestamp("2023-03-01"))] == pytest.approx(0.0)


def test_treatment_dose_handles_closure():
    months = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    pnl = _panel_skeleton(["a"], months)
    dc = pd.DataFrame({
        "unit_id": ["a"],
        "open_date": pd.to_datetime(["2023-01-01"]),
        "close_date": pd.to_datetime(["2023-03-01"]),
        "capacity_mw": [100.0],
    })
    dose = panel.build_treatment_dose(pnl, dc, cap_transform="level")
    assert dose.loc[("a", pd.Timestamp("2023-02-01"))] == pytest.approx(100.0)
    assert dose.loc[("a", pd.Timestamp("2023-03-01"))] == pytest.approx(0.0)


def test_first_treatment_date_and_event_time():
    months = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    pnl = _panel_skeleton(["a", "b"], months)
    dc = pd.DataFrame({
        "unit_id": ["a"],
        "open_date": pd.to_datetime(["2023-02-01"]),
        "capacity_mw": [10.0],
    })
    cohorts = panel.first_treatment_date(dc)
    assert cohorts.loc["a"] == pd.Timestamp("2023-02-01")

    et = panel.build_event_time(pnl, cohorts)
    assert et.loc[("a", pd.Timestamp("2023-01-01"))] == pytest.approx(-1.0)
    assert et.loc[("a", pd.Timestamp("2023-02-01"))] == pytest.approx(0.0)
    assert et.loc[("a", pd.Timestamp("2023-03-01"))] == pytest.approx(1.0)
    # never-treated unit b has NaN event time
    assert np.isnan(et.loc[("b", pd.Timestamp("2023-02-01"))])
