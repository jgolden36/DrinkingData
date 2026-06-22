"""Spatial weights construction and spatial lags (§2.2)."""

import numpy as np
import pandas as pd
import pytest

from drinkingdata.spatial import weights as W


def test_row_standardize_rows_sum_to_one_or_zero():
    M = np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R = W.row_standardize(M)
    np.testing.assert_allclose(R.sum(axis=1), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(R[0], [0.0, 0.5, 0.5])


def test_from_edges_downstream_exposure_directed():
    # Flow a -> b -> c: downstream units are exposed to upstream ones.
    edges = pd.DataFrame({"src_unit_id": ["a", "b"], "dst_unit_id": ["b", "c"]})
    sw = W.from_edges(["a", "b", "c"], edges, standardize=True)
    pos = {u: i for i, u in enumerate(sw.order)}
    # b exposed to a; c exposed to b; a exposed to nobody.
    assert sw.W[pos["b"], pos["a"]] == pytest.approx(1.0)
    assert sw.W[pos["c"], pos["b"]] == pytest.approx(1.0)
    assert sw.W[pos["a"]].sum() == pytest.approx(0.0)


def test_from_edges_undirected_symmetric():
    edges = pd.DataFrame({"src_unit_id": ["a"], "dst_unit_id": ["b"]})
    sw = W.from_edges(["a", "b"], edges, directed=False, standardize=False)
    assert sw.W[0, 1] == sw.W[1, 0] == 1.0


def test_from_groups_complete_within_basin():
    units = pd.DataFrame({
        "unit_id": ["a", "b", "c", "d"],
        "basin_id": ["X", "X", "Y", "Y"],
    })
    sw = W.from_groups(units, standardize=True)
    pos = {u: i for i, u in enumerate(sw.order)}
    # a and b share basin X -> connected; a and c do not.
    assert sw.W[pos["a"], pos["b"]] == pytest.approx(1.0)
    assert sw.W[pos["a"], pos["c"]] == pytest.approx(0.0)


def test_from_knn_picks_nearest():
    units = pd.DataFrame({
        "unit_id": [0, 1, 2, 3],
        "x": [0.0, 1.0, 5.0, 6.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })
    sw = W.from_knn(units, k=1, standardize=False)
    # nearest neighbour of 0 is 1; of 2 is 3.
    assert sw.W[0, 1] == 1.0
    assert sw.W[2, 3] == 1.0


def test_lag_panel_spillover_dose():
    # Two units, two periods. b is downstream of a.
    edges = pd.DataFrame({"src_unit_id": ["a"], "dst_unit_id": ["b"]})
    sw = W.from_edges(["a", "b"], edges, standardize=True)
    idx = pd.MultiIndex.from_tuples(
        [("a", "2023-01"), ("b", "2023-01"), ("a", "2023-02"), ("b", "2023-02")],
        names=["unit_id", "date"],
    )
    dose = pd.Series([10.0, 0.0, 20.0, 0.0], index=idx)
    wd = sw.lag_panel(dose)
    # b's spillover exposure equals a's dose each period; a has none.
    assert wd.loc[("b", "2023-01")] == pytest.approx(10.0)
    assert wd.loc[("b", "2023-02")] == pytest.approx(20.0)
    assert wd.loc[("a", "2023-01")] == pytest.approx(0.0)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        W.SpatialWeights(np.zeros((2, 3)), ["a", "b"])
