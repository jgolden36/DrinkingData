"""Config resolution: paths, env overrides, defaults."""

from pathlib import Path

import pytest

from drinkingdata.config import Config


def test_from_dict_merges_defaults_and_run_settings():
    cfg = Config.from_dict({
        "data_dir": "/data",
        "run": {"spatial_unit": "county_supplier", "event_max": 48, "bogus": 1},
    })
    assert cfg.data_dir == Path("/data")
    assert cfg.run.spatial_unit == "county_supplier"
    assert cfg.run.event_max == 48
    # default paths still present
    assert "aterio_inventory" in cfg.paths


def test_path_resolves_relative_under_data_dir():
    cfg = Config.from_dict({"data_dir": "/data", "paths": {"epoch_models": "e.csv"}})
    assert cfg.path("epoch_models") == Path("/data/e.csv")


def test_path_absolute_passthrough():
    cfg = Config.from_dict({"data_dir": "/data", "paths": {"epoch_models": "/abs/e.csv"}})
    assert cfg.path("epoch_models") == Path("/abs/e.csv")


def test_unset_path_raises_helpful_error():
    cfg = Config.from_dict({"paths": {"huc8_shp": None}})
    with pytest.raises(ValueError):
        cfg.path("huc8_shp")
    with pytest.raises(KeyError):
        cfg.path("does_not_exist")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DRINKINGDATA_DATA_DIR", str(tmp_path))
    cfg = Config.load(config_file=tmp_path / "nonexistent.yaml")
    assert cfg.data_dir == tmp_path
