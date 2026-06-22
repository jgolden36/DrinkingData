"""Central configuration: data paths and run settings.

Replaces the hardcoded Windows ``BASE_DIR`` strings scattered across the legacy
scripts (e.g. ``TexasWaterAnalysis.py``) with a single resolvable config.

Resolution order for every setting:

    explicit argument  >  config.yaml  >  DRINKINGDATA_* env var  >  default

Example
-------
>>> from drinkingdata.config import Config
>>> cfg = Config.load()                 # reads ./config.yaml or env vars
>>> cfg.path("aterio_inventory")        # absolute Path under data_dir
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Defaults (mirror config.example.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_PATHS: Dict[str, Optional[str]] = {
    "aterio_inventory": "data_center_inventory_20251217.csv",
    "epoch_models": "Epoch Database - Notable Models.csv",
    "aqueduct_csv": "Aqueduct40_baseline_annual_y2023m07d05.csv",
    "twdb_water_glob": "SumFinal_CountyReportWithReuse*.csv",
    "tx_county_shp": None,
    "ca_swrcb_csv": None,
    "ca_sgma_basins": None,
    "huc8_shp": None,
    "huc12_shp": None,
    "nhdplus_flow_edges": None,
    "energy_marginal_generation": None,
    "jin2019_water_factors": None,
}


@dataclass
class RunSettings:
    """Estimation / pipeline knobs (see paper §2 and CLAUDE.md)."""

    spatial_unit: str = "huc8"          # {huc8, county_supplier, huc12}
    freq: str = "MS"                     # pandas offset alias (monthly start)
    states: List[str] = field(default_factory=lambda: ["TX", "CA"])
    hyperscalers: List[str] = field(
        default_factory=lambda: ["Google", "Microsoft", "Amazon", "Meta"]
    )
    event_min: int = -24
    event_max: int = 36
    event_omit: int = -1
    cap_transform: str = "log1p"         # {level, log1p, sqrt}
    random_seed: int = 20240101


@dataclass
class Config:
    """Top-level configuration object."""

    data_dir: Path = Path(".")
    outputs_dir: Path = Path("./outputs")
    paths: Dict[str, Optional[str]] = field(
        default_factory=lambda: dict(_DEFAULT_PATHS)
    )
    run: RunSettings = field(default_factory=RunSettings)

    # -- loading --------------------------------------------------------------

    @classmethod
    def load(cls, config_file: Optional[os.PathLike] = None) -> "Config":
        """Build a Config from (in order) a YAML file then environment vars.

        If ``config_file`` is omitted, ``./config.yaml`` is used when present.
        Any ``DRINKINGDATA_DATA_DIR`` / ``DRINKINGDATA_OUTPUTS_DIR`` environment
        variables override the corresponding file values.
        """
        data: Dict[str, Any] = {}

        path = Path(config_file) if config_file else Path("config.yaml")
        if path.exists():
            data = _read_yaml(path)

        cfg = cls.from_dict(data)

        # Environment overrides (highest precedence after explicit args).
        env_data_dir = os.environ.get("DRINKINGDATA_DATA_DIR")
        if env_data_dir:
            cfg.data_dir = Path(env_data_dir)
        env_out = os.environ.get("DRINKINGDATA_OUTPUTS_DIR")
        if env_out:
            cfg.outputs_dir = Path(env_out)

        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        paths = dict(_DEFAULT_PATHS)
        paths.update(data.get("paths") or {})

        run_kwargs = data.get("run") or {}
        known = {f.name for f in fields(RunSettings)}
        run = RunSettings(**{k: v for k, v in run_kwargs.items() if k in known})

        return cls(
            data_dir=Path(data.get("data_dir", ".")),
            outputs_dir=Path(data.get("outputs_dir", "./outputs")),
            paths=paths,
            run=run,
        )

    # -- resolution -----------------------------------------------------------

    def path(self, key: str) -> Path:
        """Resolve a named input path to an absolute ``Path``.

        Relative entries are resolved under ``data_dir``; absolute entries are
        returned as-is. Raises ``KeyError`` for unknown keys and ``ValueError``
        if the entry is unset (``None``) — these are the data files that still
        need to be wired up per the CLAUDE.md roadmap.
        """
        if key not in self.paths:
            raise KeyError(
                f"Unknown path key {key!r}. Known: {sorted(self.paths)}"
            )
        value = self.paths[key]
        if value is None:
            raise ValueError(
                f"Path {key!r} is not configured. Set it in config.yaml "
                f"(see config.example.yaml)."
            )
        p = Path(value)
        return p if p.is_absolute() else (self.data_dir / p)

    def output(self, *parts: str) -> Path:
        """Build a path under ``outputs_dir``, creating the directory."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir.joinpath(*parts)


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml is a core dependency

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
