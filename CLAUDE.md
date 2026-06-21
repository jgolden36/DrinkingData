# CLAUDE.md

This file orients Claude Code (and human collaborators) to the **Drinking Data**
project: an econometric study of how AI data centers affect water resources in
Texas and California. It maps the existing Python code to the analysis described
in the paper (`DrinkingData (2).tex`) and records the conventions, data sources,
and open work needed to take the project from its current exploratory state to
the full estimation pipeline the paper specifies.

---

## 1. Project Overview

**Paper:** *Drinking Data: A Spatial Econometric Analysis of the Impacts of Data
Centers and AI on Water Resources in Texas and California* (Golden & Varshney).
Source: `DrinkingData (2).tex`.

**Research question.** Provide the first *causal, spatially explicit* estimates of
how AI data centers affect (i) direct water withdrawal/consumption, (ii)
wastewater discharge, and (iii) indirect water use through the electricity grid —
then scale those micro estimates to continental hydrology (NOAA NWM) and the wider
economy (WEST), and run policy counterfactuals.

**Companion paper.** This is the water-focused companion to an energy paper
(`Golden_Energy_companion`). It reuses that paper's treatment definition (staggered
frontier AI model releases) and data-center crosswalk, and consumes its estimated
marginal fossil-generation impacts as an input to the *indirect* water accounting.

**Three-stage architecture** (the spine of the whole project):

| Stage | Scale | Tool | Output |
|-------|-------|------|--------|
| **Micro** | HUC-8 / county×supplier | Spatial DiD (this repo) | Direct effect `β`, spillover effect `θ` on water outcomes |
| **Meso** | NHDPlus reaches (CONUS) | NOAA National Water Model | Streamflow, groundwater, soil moisture, ET anomalies |
| **Macro** | TX/CA regions | WEST nexus model | Agriculture, energy, GDP, price impacts |

Each stage is run **twice** — once with econometric estimates (`Δ^econ`) and once
with literature-derived techno-economic WUE coefficients (`Δ^te`) — and the
difference is itself a result (engineering vs. causal approaches).

---

## 2. Core Methodology (what the code must implement)

The paper (Section "Methodology") specifies the estimands precisely. When writing
or reviewing estimation code, conform to these:

### 2.1 Treatment
- **Unit `i`:** USGS HUC-8 sub-basin (primary), county×water-supplier cell
  (secondary), HUC-12 (robustness).
- **Treated:** unit contains ≥1 Aterio-classified **AI** data center owned by/
  contracted to a hyperscaler (Meta, Microsoft, Amazon, Google; Anthropic→Google).
- **Treatment events:** staggered frontier AI model releases (Epoch AI database),
  restricted to the four hyperscalers. Release date is a *noisy proxy* for the
  unobserved training window → use **dynamic event-study** coefficients, not a
  single discontinuity.

### 2.2 Spatial weights `W`
Built from **hydrologic/hydraulic connectivity**, not Euclidean distance:
shared HUC-8 boundary, NHDPlus downstream relationship, or shared managed
groundwater sub-basin (SGMA in CA / GCD in TX). Row-standardized baseline.
Robustness: k-NN (k∈{4,6,8}), inverse-distance (d̄∈{50,100,200} km),
flow-direction-only, groundwater-basin-only.

### 2.3 Estimators (in increasing generality)
- **SLX-DiD** (workhorse): `Y = α_i + δ_t + β·D + θ·(WD) + Xγ + ε`.
  `β` = direct ATT, `θ` = indirect/spillover ATT.
- **SAR-DiD:** adds `ρ·(WY)`; report LeSage–Pace direct/indirect/total impacts.
- **SDM-DiD** (preferred default): adds both `ρ·(WY)` and `(WX)η`; nests SLX/SAR.
  Run LR/Wald/common-factor tests to discriminate SLX/SAR/SEM/SDM.
- **Staggered spatial event study:** stacked, cohort-specific `{β_k, θ_k}` over
  event-time `k`, with unit×event and time×event fixed effects; SE clustered at
  the spatial-unit level (two-way unit×event as robustness).

### 2.4 Outcomes (estimate each separately)
1. Direct water withdrawal / consumption (MGal), at HUC-8×month.
2. Sector-disaggregated water use (municipal, manufacturing, mining,
   steam-electric, irrigation, livestock).
3. Wastewater discharge (MGD, NPDES eDMR).
4. **Indirect** water = `Σ_f ω_f^water · ΔÊ_f`, where `ΔÊ_f` is the energy
   companion's marginal generation by fuel and `ω_f^water` is from Jin et al. (2019).

### 2.5 Techno-economic benchmark & comparison
- `TE-Direct = Σ_j WUE_j · φ_j · Capacity_j · Util_j · Δt` (φ = 1/PUE).
- Central deliverable: decompose the gap `ΔΔ = Δ^econ − Δ^te` into
  **spillover + (marginal vs. average indirect) + direct-effect gap**, reported by
  sector and Aqueduct water-stress class.

### 2.6 Robustness battery (Section "Validating Assumptions")
Pre-trends/parallel-trends (+ Rambachan–Roth sensitivity), placebo tests
(randomized dates, randomized units, non-AI DC placebo), spatial-weights
sensitivity, treatment-definition sensitivity (baseline → confirmed locations →
verified dates), Callaway–Sant'Anna group-time ATTs, and anomaly detection
(STL+ESD) clustered around release dates.

---

## 3. Code Structure (current state)

> **Status legend:** ✅ functional · 🟡 skeleton/partial · 🗺️ visualization ·
> ⚠️ uses synthetic/placeholder geography that must be replaced for the paper.

All code is currently flat in the repo root and **Texas-centric / US-wide
exploratory**. The paper requires HUC-8 panels for *both* TX and CA with true
hydrologic weights; see §5 for the gap.

### 3.1 Data preparation & panel construction
| File | Status | Role |
|------|--------|------|
| `TexasWaterAnalysis.py` | ✅ | Builds the TX county panel: loads TWDB sectoral water CSVs, county shapefile, Aterio DC inventory; point-in-polygon assigns DCs to counties; creates treatment dummies (`has_ai_datacenter`, `ai_datacenter_capacity_mw`, `post_chatgpt`). Outputs `texas_water_datacenter_analysis.csv`. |
| `waterPreliminAnalysis.py` | 🟡 | EDA. Merges Epoch model list + Aterio inventory (`FLG_AI_FACILITY`) + WRI Aqueduct 4.0 stress scores; `map_organization_to_provider()` normalizes operator names to hyperscalers. |

### 3.2 Estimation
| File | Status | Role |
|------|--------|------|
| `TexasWaterPanelRegression.py` | ✅ | County-level DiD: `PanelOLS` with entity FE, time trend, AR(1) lag, clustered SE. Binary (hyperscaler/AI) and continuous (capacity) treatments × post-2021. Includes placebo (fake 2018) and pre-trends checks. Outputs regression tables/CSVs. **This is the non-spatial DiD.** |
| `WaterAnalysis.py` | 🟡 | The **spatial-DiD engine** and the most paper-aligned file. Generic over spatial units. Builds `W` three ways: contiguity (`libpysal` Queen/Rook), hydrologic network edges, within-group (utility/aquifer/basin). Computes spillover dose `WD`, Sun–Abraham cohort×event-time design, fits `PanelOLS` for direct + spillover effects. **Extend this toward SLX/SAR/SDM and the LeSage–Pace impacts.** |

### 3.3 Visualization & mapping 🗺️
| File | Status | Notes |
|------|--------|-------|
| `TexasWaterVisualization.py` | ✅ | Choropleths (per-capita water 2018 vs 2023, % change), indexed time series with ChatGPT line, dose-response scatter, sectoral stacked bars. |
| `ai_datacenter_water_stress_focused_map.py` | ✅🗺️ | AI DCs over high-stress zones, stage-colored; outputs at-risk CSV + summary. |
| `ai_datacenter_water_stress_analysis_charts.py` | ✅🗺️ | 9-panel dashboard + multi-sheet Excel of provider/state/zone exposure. |
| `water_stress_datacenter_analysis.py` | 🟡🗺️ | Geographic-only merge of Aqueduct stress × DCs; summary/temporal figures. |
| `water_stress_map_overlay.py`, `water_stress_professional_map.py`, `water_stress_analysis_gdb.py` | 🟡🗺️ | Progressively more polished stress maps; `_gdb` attempts to read the Aqueduct `.gdb` via fiona. |
| `create_proper_us_map.py`, `us_map_with_states.py` | 🟡⚠️ | Base-map loaders (Census/Natural Earth shapefiles), with synthetic fallbacks. |
| `create_realistic_water_map.py`, `create_water_overlay_map.py`, `water_depletion_map_with_boundaries.py` | 🟡⚠️ | **Synthetic** US outlines / hardcoded state polygons / rectangle stress zones. Presentation only — **not** a source of analytic geography. |

### 3.4 Artifacts
`water_stress_analysis.png` — committed example output.

---

## 4. Data Sources (per paper Table "Data sources")

| Source | Content | Use |
|--------|---------|-----|
| **Aterio** | DC location, owner, capacity (MW), AI classification, cooling type | Define treated set |
| **Epoch AI** | Model releases, FLOPs, params, dates | Treatment events |
| **TWDB Water Use Survey** | Sectoral water use by TX county | Direct-use outcome (TX) |
| **CA SWRCB / DWR / eWRIMS** | Sectoral use, urban supplier (SB X7-7 / 555/606), water rights, SGMA | Direct-use outcome (CA) |
| **EPA eDMR / ICIS-NPDES** | Facility-month wastewater | Wastewater outcome |
| **USGS NWIS / NWUIP** | Streamflow, groundwater; county×sector (5-yr) | Controls, weights, cross-check |
| **ERCOT / CAISO** | Hourly dispatch by fuel, prices | Marginal generator at incremental load |
| **EIA Forms 860 / 923 (Sch. 8)** | Generator + thermoelectric cooling/water | Map MWh → indirect water |
| **Jin et al. (2019)** | Water-consumption factors by tech/cooling | `ω_f^water` |
| **NWM (Cosgrove et al.)** | CONUS hydrofabric/forcing | Meso propagation |
| **WEST (Reimer et al.)** | Food–energy–water nexus sim | Macro propagation |
| **WRI Aqueduct, PRISM, Meteostat** | Stress, climate (wet-bulb, PDSI, precip) | Controls, scenario design |

> **Data is not committed to the repo.** Scripts reference local paths (e.g.
> Aterio `data_center_inventory_*.csv`, Epoch `Notable Models.csv`, Aqueduct CSV/
> GDB). Keep raw data out of git; document expected paths near each loader.

---

## 5. From Current Code to the Paper (gap analysis / roadmap)

The existing code is a strong **Texas county** prototype. To realize the paper:

1. **Add California.** Mirror `TexasWaterAnalysis.py` for CA (eWRIMS / SWRCB urban
   supplier / SGMA). Stack TX+CA into one panel.
2. **Switch the spatial unit to HUC-8 (and HUC-12 robustness).** Replace county
   polygons with the NWM/NHDPlus hydrofabric; re-run the DC point-in-polygon join
   against HUC-8.
3. **Build the real hydrologic `W`.** Implement the connectivity rules in §2.2
   inside `WaterAnalysis.py` (`build_W_network_edges` is the hook). Retire the
   synthetic-geography map scripts from the analytic path.
4. **Promote the estimator from county-DiD to spatial DiD.** Extend
   `WaterAnalysis.py` to SLX → SAR → SDM, and implement LeSage–Pace
   direct/indirect/total impact decomposition with simulated CIs.
5. **Tie treatment to staggered model releases** (Epoch), not a single
   post-ChatGPT cutoff. Implement stacked cohort event-study `{β_k, θ_k}`.
6. **Wastewater outcome.** Add eDMR/ICIS-NPDES ingestion and discharge equations.
7. **Indirect water module.** Consume the energy companion's `ΔÊ_f` and apply
   `ω_f^water`; bootstrap joint uncertainty.
8. **Techno-economic benchmark + gap decomposition** (§2.5).
9. **Robustness battery** (§2.6).
10. **Meso/Macro linkage** to NWM (Eq. NWM-forcing) and WEST, with parallel
    `Δ^econ` / `Δ^te` runs and the five counterfactuals (efficiency, siting/
    moratoria, geographic reallocation, scarcity pricing, water–energy nexus).

---

## 6. Conventions & Environment

- **Language:** Python 3. Key libs: `pandas`, `geopandas`, `shapely`,
  `statsmodels`, `linearmodels` (`PanelOLS`), `libpysal`/`spreg` (spatial),
  `matplotlib`/`seaborn`, `openpyxl`. (`spreg` will likely be needed for SAR/SDM
  MLE — not yet imported anywhere.)
- **Treatment timing:** prefer event-time relative to each model release; the
  legacy `post_chatgpt`/`post_2021` cutoff is a prototype convenience to be
  superseded by the staggered design.
- **Outputs:** scripts write PNG/CSV/XLSX to the repo root. Consider a future
  `outputs/` directory; not enforced yet.
- **Provider normalization:** always route operator strings through
  `map_organization_to_provider()` (in `waterPreliminAnalysis.py`) so hyperscaler
  attribution is consistent across files.
- **Synthetic geography is presentation-only.** Never use the hardcoded
  state/zone polygons (`create_realistic_water_map.py`,
  `water_depletion_map_with_boundaries.py`, etc.) as analytic spatial units.

---

## 7. Git / Workflow

- Active development branch: `claude/hopeful-tesla-1ijxd5`.
- Commit with clear messages; do not open PRs unless explicitly requested.
- Raw data and large binaries stay out of version control.
