# Drinking Data

*A Spatial Econometric Analysis of the Impacts of Data Centers and AI on Water
Resources in Texas and California* (Golden & Varshney).

This repository implements the estimation pipeline described in
`DrinkingData (2).tex`. See [`CLAUDE.md`](CLAUDE.md) for the full mapping of the
paper's methodology to code, the data-source catalog, and the project roadmap.

---

## What's here

The analysis is organized as an installable Python package, `drinkingdata`,
that consolidates and generalizes the original flat exploratory scripts into a
reusable three-stage pipeline (micro spatial-DiD → meso NWM → macro WEST).

```
drinkingdata/
├── config.py            # Paths & run settings (replaces hardcoded Windows paths)
├── data/
│   ├── providers.py     # Canonical operator → hyperscaler normalization
│   ├── loaders.py       # Aterio / Epoch / TWDB / CA / Aqueduct / EIA loaders
│   └── panel.py         # Unit×time panel, treatment dose, cohorts, event time
├── spatial/
│   ├── weights.py       # Hydrologic / institutional / contiguity / kNN / IDW W
│   └── units.py         # Point-in-polygon assignment of DCs to spatial units
├── estimation/
│   ├── did.py           # Non-spatial DiD (PanelOLS) — §2.3 baseline
│   ├── spatial_did.py   # SLX / SAR / SDM — §2.3
│   ├── impacts.py       # LeSage–Pace direct / indirect / total impacts
│   └── eventstudy.py    # Staggered cohort event study {β_k, θ_k} — §2.3
├── indirect/
│   └── water.py         # Indirect water = Σ_f ω_f · ΔÊ_f — §2.4
├── techno_economic.py   # TE benchmark + gap decomposition ΔΔ — §2.5
├── robustness.py        # Pre-trends, placebo, sensitivity — §2.6
└── viz/                 # Plotting helpers (maps, event-study, dose-response)
```

The legacy top-level scripts (`TexasWaterAnalysis.py`, `WaterAnalysis.py`, …)
are retained as references during the migration; their logic is being lifted
into the package modules above.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"      # full geo + spatial-econometrics stack
# or, for the lightweight pure-Python core (no geopandas/spreg):
pip install -e .
```

The heavy geospatial and spatial-econometrics dependencies
(`geopandas`, `linearmodels`, `libpysal`, `spreg`) are **lazily imported**, so
the package imports and the pure-numeric modules (weights math, panel
construction, indirect water, techno-economic benchmark) run without them.

## Configuration

Raw data is **not** committed (see `CLAUDE.md` §4). Point the pipeline at your
local data directory via an environment variable or a config file:

```bash
export DRINKINGDATA_DATA_DIR=/path/to/Data
```

or copy `config.example.yaml` to `config.yaml` and edit the paths. Resolution
order is: explicit argument → `config.yaml` → `DRINKINGDATA_*` env vars →
package defaults.

## Quick start

```python
from drinkingdata.config import Config
from drinkingdata.data import providers

cfg = Config.load()
primary, secondary = providers.map_to_hyperscaler("Anthropic")  # -> ("Google", None)
```

Run the test suite (pure-Python core, no data required):

```bash
pytest -q
```

## Status

This is **scaffolding plus initial code**. Pure-numeric cores (provider
normalization, spatial-weights construction, panel/event-time construction,
indirect-water accounting, techno-economic benchmark) are implemented and
unit-tested against synthetic data. Modules that require the geospatial /
spatial-econometrics stack or external data (`spatial_did`, `units`, loaders)
are functional skeletons with documented interfaces, ready to be wired to real
data per the `CLAUDE.md` roadmap.
