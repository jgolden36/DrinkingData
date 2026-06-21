"""
Skeleton: Spatial DiD / event-study with spillovers (hydrologic + institutional),
dose treatment (capacity), and spatially-robust inference.

Assumptions (edit as needed):
- GIS panel is at unit_id × time (e.g., county-month, HUC8-month, utility-month).
- You have a GeoDataFrame with polygons for units: units_gdf (unit_id, geometry, ...).
- You have a panel DataFrame: water_panel_df with columns:
    unit_id, date (datetime64), usage, price, availability, controls...
- You have a data center dataset: dc_df with columns:
    dc_id, open_date (datetime64), close_date (optional), capacity_mw (or IT_MW),
    geometry (Point), maybe training_window_start/end (optional).
- You can optionally provide:
    (a) surface-water network mapping: upstream/downstream adjacency between units, or
    (b) aquifer_id per unit, or
    (c) utility_id per unit.
This code builds multiple exposure matrices and estimates:
    y_it = FE_i + FE_t + β * D_it + λ * (W D)_it + θX_it + ε_it
and event-study (Sun-Abraham-style) on direct D_it (and optionally spillovers).

Requires:
    pandas, numpy, geopandas, shapely, linearmodels, statsmodels, libpysal (optional),
    scipy (optional for sparse ops)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from shapely.ops import unary_union

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# Optional: for contiguity weights (queen/rook) if you want spatial adjacency
try:
    from libpysal.weights import Queen, Rook
    HAS_LIBPYSAL = True
except Exception:
    HAS_LIBPYSAL = False

# Optional: sparse matrices for big panels
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ----------------------------
# Config
# ----------------------------

@dataclass
class AnalysisConfig:
    # panel frequency; used for event time bins
    time_freq: str = "MS"  # Month start; set "D", "W", "Q", etc.
    # event study window in periods (relative to open_date)
    event_min: int = -24
    event_max: int = 36
    # omit period for normalization
    event_omit: int = -1
    # treatment dose: capacity scaling
    cap_transform: str = "log1p"  # {"level","log1p","sqrt"}
    # distance threshold for "nearby" spillover as fallback (meters)
    nearby_buffer_m: float = 50_000.0
    # whether to include unit-specific linear trends
    unit_trends: bool = False
    # cluster dimension(s)
    cluster_on: List[str] = None  # e.g., ["basin_id"] or ["state"]


# ----------------------------
# Utilities
# ----------------------------

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def set_panel_index(df: pd.DataFrame, unit_col: str, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col])
    df = df.set_index([unit_col, time_col])
    return df

def transform_capacity(x: pd.Series, how: str) -> pd.Series:
    if how == "level":
        return x
    if how == "log1p":
        return np.log1p(np.clip(x, a_min=0, a_max=None))
    if how == "sqrt":
        return np.sqrt(np.clip(x, a_min=0, a_max=None))
    raise ValueError(f"Unknown cap_transform: {how}")

def normalize_rows(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return W / row_sums

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how="left") -> pd.DataFrame:
    out = left.merge(right, on=on, how=how, validate="m:1")
    return out

def assert_unique(df: pd.DataFrame, cols: List[str], name: str):
    dup = df.duplicated(cols).sum()
    if dup > 0:
        raise ValueError(f"{name} has {dup} duplicates on keys {cols}.")


# ----------------------------
# Step 1: Spatial join data centers -> units
# ----------------------------

def spatial_assign_datacenters_to_units(
    units_gdf: gpd.GeoDataFrame,
    dc_gdf: gpd.GeoDataFrame,
    unit_id_col: str = "unit_id",
) -> gpd.GeoDataFrame:
    """
    Returns dc_gdf with a unit_id column assigned by point-in-polygon.
    """
    if units_gdf.crs != dc_gdf.crs:
        dc_gdf = dc_gdf.to_crs(units_gdf.crs)

    joined = gpd.sjoin(
        dc_gdf,
        units_gdf[[unit_id_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    if joined[unit_id_col].isna().any():
        n = joined[unit_id_col].isna().sum()
        print(f"[WARN] {n} data centers were not matched to a unit polygon (outside bounds?).")
    return joined


# ----------------------------
# Step 2: Build unit-time treatment dose D_it
# ----------------------------

def build_unit_time_treatment(
    panel_index: pd.MultiIndex,
    dc_assigned_df: pd.DataFrame,
    unit_id_col: str = "unit_id",
    open_col: str = "open_date",
    capacity_col: str = "capacity_mw",
    date_level_name: str = "date",
    cap_transform: str = "log1p",
) -> pd.Series:
    """
    Construct D_it = sum(capacity) for all centers that are open at time t, in unit i.
    If you have close dates, add them (see placeholder).
    """
    dc = dc_assigned_df.copy()
    dc = ensure_datetime(dc, open_col)
    # Optional:
    if "close_date" in dc.columns:
        dc = ensure_datetime(dc, "close_date")

    # Expand data centers into unit-time series by merging on time >= open_date (and <= close if exists).
    # Efficient approach: for each unit, compute cumulative sum by open_date.
    idx_df = pd.DataFrame({unit_id_col: panel_index.get_level_values(0),
                           date_level_name: panel_index.get_level_values(1)})
    idx_df = idx_df.drop_duplicates()

    # Aggregate openings by (unit, open_date)
    opens = (
        dc.dropna(subset=[unit_id_col, open_col, capacity_col])
          .groupby([unit_id_col, open_col], as_index=False)[capacity_col].sum()
          .rename(columns={open_col: date_level_name, capacity_col: "cap_open"})
    )

    # Align opening dates to panel frequency (e.g., month start) to avoid partial-period confusion
    opens[date_level_name] = opens[date_level_name].dt.to_period(idx_df[date_level_name].dt.freqstr if hasattr(idx_df[date_level_name].dt, "freqstr") else "M").dt.to_timestamp()

    # Merge openings and compute cumulative capacity by unit over time
    tmp = idx_df.merge(opens, on=[unit_id_col, date_level_name], how="left")
    tmp["cap_open"] = tmp["cap_open"].fillna(0.0)
    tmp = tmp.sort_values([unit_id_col, date_level_name])
    tmp["cap_cum"] = tmp.groupby(unit_id_col)["cap_open"].cumsum()

    # Optional: handle closures by subtracting cumulative close events.
    # if "close_date" in dc.columns:
    #     closes = (dc.dropna(subset=[unit_id_col, "close_date", capacity_col])
    #                .groupby([unit_id_col, "close_date"], as_index=False)[capacity_col].sum()
    #                .rename(columns={"close_date": date_level_name, capacity_col: "cap_close"}))
    #     closes[date_level_name] = closes[date_level_name].dt.to_period("M").dt.to_timestamp()
    #     tmp = tmp.merge(closes, on=[unit_id_col, date_level_name], how="left")
    #     tmp["cap_close"] = tmp["cap_close"].fillna(0.0)
    #     tmp["cap_close_cum"] = tmp.groupby(unit_id_col)["cap_close"].cumsum()
    #     tmp["cap_cum"] = tmp["cap_cum"] - tmp["cap_close_cum"]

    D = transform_capacity(tmp["cap_cum"], cap_transform)
    D.index = pd.MultiIndex.from_frame(tmp[[unit_id_col, date_level_name]])
    D.name = "D_it"
    # Reindex exactly to panel_index
    D = D.reindex(panel_index).fillna(0.0)
    return D


# ----------------------------
# Step 3: Build spillover exposure matrices W
# ----------------------------

def build_W_contiguity(units_gdf: gpd.GeoDataFrame, unit_id_col="unit_id", kind="queen") -> Tuple[np.ndarray, List]:
    """
    Queen/Rook contiguity adjacency. Use ONLY as fallback if you don't have hydrologic connectivity.
    Returns W row-normalized, and the unit_id order.
    """
    if not HAS_LIBPYSAL:
        raise ImportError("libpysal not installed; cannot build contiguity weights.")

    g = units_gdf[[unit_id_col, "geometry"]].dropna().reset_index(drop=True).copy()
    order = g[unit_id_col].tolist()
    if kind == "queen":
        w = Queen.from_dataframe(g)
    elif kind == "rook":
        w = Rook.from_dataframe(g)
    else:
        raise ValueError("kind must be queen or rook")

    # Build dense W (fine for small N; for large N consider sparse)
    n = len(order)
    W = np.zeros((n, n), dtype=float)
    for i, neigh in w.neighbors.items():
        if len(neigh) == 0:
            continue
        W[i, neigh] = 1.0
    W = normalize_rows(W)
    return W, order

def build_W_network_edges(
    units: pd.DataFrame,
    edges: pd.DataFrame,
    unit_id_col="unit_id",
    src_col="src_unit_id",
    dst_col="dst_unit_id",
    weight_col: Optional[str] = None,
) -> Tuple[np.ndarray, List]:
    """
    Build W from a directed or undirected edge list (e.g., hydrologic upstream->downstream,
    or same-utility links, etc.). If weight_col is None, uses 1s.
    """
    order = units[unit_id_col].dropna().unique().tolist()
    pos = {u: i for i, u in enumerate(order)}
    n = len(order)
    W = np.zeros((n, n), dtype=float)

    e = edges.dropna(subset=[src_col, dst_col]).copy()
    if weight_col is None:
        e["_w"] = 1.0
        weight_col = "_w"

    for _, r in e.iterrows():
        s, d = r[src_col], r[dst_col]
        if s in pos and d in pos and s != d:
            W[pos[d], pos[s]] += float(r[weight_col])  # interpret as exposure of d to s
    W = normalize_rows(W)
    return W, order

def build_W_within_group(units: pd.DataFrame, group_col: str, unit_id_col="unit_id") -> Tuple[np.ndarray, List]:
    """
    Institutional spillovers: same utility, same groundwater district, same aquifer, etc.
    Complete graph within group (excluding self).
    """
    u = units[[unit_id_col, group_col]].dropna().copy()
    order = u[unit_id_col].unique().tolist()
    pos = {uid: i for i, uid in enumerate(order)}
    n = len(order)
    W = np.zeros((n, n), dtype=float)

    for gval, grp in u.groupby(group_col):
        ids = grp[unit_id_col].unique().tolist()
        if len(ids) <= 1:
            continue
        idxs = [pos[x] for x in ids if x in pos]
        for i in idxs:
            for j in idxs:
                if i != j:
                    W[i, j] = 1.0
    W = normalize_rows(W)
    return W, order


# ----------------------------
# Step 4: Create event time and Sun–Abraham-style cohort dummies
# ----------------------------

def first_treatment_date(dc_assigned_df: pd.DataFrame, unit_id_col="unit_id", open_col="open_date") -> pd.Series:
    """
    T_i = first opening date in unit i.
    """
    dc = dc_assigned_df.dropna(subset=[unit_id_col, open_col]).copy()
    dc = ensure_datetime(dc, open_col)
    Ti = dc.groupby(unit_id_col)[open_col].min()
    Ti.name = "T_i"
    return Ti

def build_event_time(panel: pd.DataFrame, Ti: pd.Series, unit_col="unit_id", time_col="date") -> pd.Series:
    """
    event_time = periods since first opening.
    For monthly data, event_time in months; for quarterly adjust accordingly.
    """
    df = panel[[unit_col, time_col]].drop_duplicates().copy()
    df = df.merge(Ti.reset_index(), on=unit_col, how="left")
    # periods difference (monthly default)
    # Align to month start
    df[time_col] = pd.to_datetime(df[time_col]).dt.to_period("M").dt.to_timestamp()
    df["T_i"] = pd.to_datetime(df["T_i"]).dt.to_period("M").dt.to_timestamp()
    df["event_time"] = np.where(
        df["T_i"].notna(),
        (df[time_col].dt.to_period("M") - df["T_i"].dt.to_period("M")).astype(int),
        np.nan,
    )
    et = df.set_index([unit_col, time_col])["event_time"]
    return et

def sun_abraham_design_matrix(
    panel_index: pd.MultiIndex,
    event_time: pd.Series,
    cohorts: pd.Series,
    event_min: int,
    event_max: int,
    omit: int = -1,
) -> pd.DataFrame:
    """
    Build cohort-specific event-time dummies:
        1{cohort=c} * 1{event_time=k}
    Omits (k=omit) within each cohort to normalize.
    Then you can aggregate across cohorts after estimation, or
    estimate with all interactions and post-process.

    This is a skeleton; in practice you might want to bin tails.
    """
    # Align series to index
    et = event_time.reindex(panel_index)
    ch = cohorts.reindex(panel_index.get_level_values(0)).reindex(panel_index.get_level_values(0)).values
    # cohorts per row
    # easiest: create dataframe from index
    idx_df = pd.DataFrame({
        "unit_id": panel_index.get_level_values(0),
        "date": panel_index.get_level_values(1),
        "event_time": et.values,
    })
    idx_df["cohort"] = idx_df["unit_id"].map(cohorts.to_dict())
    idx_df = idx_df.dropna(subset=["cohort", "event_time"])

    cols = []
    X = pd.DataFrame(index=panel_index)

    for k in range(event_min, event_max + 1):
        if k == omit:
            continue
        name = f"ET_{k}"
        # pooled event-time dummy (not cohort-specific) — useful baseline
        X[name] = (et == k).astype(float)

    # cohort-specific interactions (often huge; consider limiting cohorts)
    # Here: cohort defined as first-treatment year-month integer label
    idx_df["cohort_label"] = pd.to_datetime(idx_df["cohort"]).dt.to_period("M").astype(str)

    for k in range(event_min, event_max + 1):
        if k == omit:
            continue
        for c in idx_df["cohort_label"].unique():
            cname = f"C_{c}__ET_{k}"
            mask = (idx_df["cohort_label"] == c) & (idx_df["event_time"] == k)
            # assign into X using index alignment
            X.loc[pd.MultiIndex.from_frame(idx_df.loc[mask, ["unit_id", "date"]]), cname] = 1.0
            cols.append(cname)

    X = X.fillna(0.0)
    return X


# ----------------------------
# Step 5: Estimation helpers (PanelOLS)
# ----------------------------

def fit_panel_ols(
    y: pd.Series,
    X: pd.DataFrame,
    add_constant: bool = False,
    entity_effects: bool = True,
    time_effects: bool = True,
    clusters: Optional[pd.DataFrame] = None,
) -> object:
    """
    Uses linearmodels.PanelOLS with FE and clustered SE (if provided).
    clusters should be a DataFrame aligned to y.index with columns specifying cluster ids.
    """
    y = y.dropna()
    X = X.reindex(y.index).copy()
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
    if clusters is not None:
        clusters = clusters.reindex(y.index)
        res = mod.fit(cov_type="clustered", clusters=clusters, debiased=True)
    else:
        res = mod.fit(cov_type="robust", debiased=True)
    return res

def build_WD_panel(
    D: pd.Series,
    W: np.ndarray,
    unit_order: List,
    panel_index: pd.MultiIndex,
    unit_col_name: str = "unit_id",
) -> pd.Series:
    """
    Compute (W D)_it where W operates on units and is applied period-by-period.
    Assumes panel_index is (unit_id, date).
    """
    # map unit -> position
    pos = {u: i for i, u in enumerate(unit_order)}
    units = panel_index.get_level_values(0)
    dates = panel_index.get_level_values(1)
    # unique dates
    uniq_dates = pd.Index(dates.unique()).sort_values()

    WD = pd.Series(index=panel_index, dtype=float, name="WD_it")

    for t in uniq_dates:
        # D vector for this date in unit order
        d_t = np.zeros(len(unit_order), dtype=float)
        mask = (dates == t)
        u_t = units[mask]
        dvals = D[mask].values
        for uu, vv in zip(u_t, dvals):
            if uu in pos:
                d_t[pos[uu]] = vv
        wd_t = W @ d_t
        # fill back
        out = np.array([wd_t[pos[uu]] if uu in pos else np.nan for uu in u_t], dtype=float)
        WD.loc[pd.MultiIndex.from_arrays([u_t, np.repeat(t, len(u_t))])] = out

    WD = WD.fillna(0.0)
    return WD


# ----------------------------
# Main pipeline
# ----------------------------

def run_analysis(
    units_gdf: gpd.GeoDataFrame,
    water_panel_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    config: AnalysisConfig,
    unit_id_col: str = "unit_id",
    time_col: str = "date",
    outcome_cols: List[str] = None,  # e.g., ["availability", "usage", "price"]
    control_cols: List[str] = None,  # e.g., drought, temp, precip, population...
    # Optional extra columns in units_gdf for clustering or spillover mapping
    basin_col: Optional[str] = "basin_id",
    aquifer_col: Optional[str] = "aquifer_id",
    utility_col: Optional[str] = "utility_id",
    # Optional hydrologic edges: DataFrame with src_unit_id, dst_unit_id (upstream->downstream)
    hydro_edges: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Returns dict of fitted model results by outcome and spec.
    """
    if config.cluster_on is None:
        config.cluster_on = [c for c in [basin_col] if c in units_gdf.columns]

    if outcome_cols is None:
        outcome_cols = ["availability"]
    if control_cols is None:
        control_cols = []

    # --- Harmonize panel dates
    panel = water_panel_df.copy()
    panel = ensure_datetime(panel, time_col)
    panel[time_col] = panel[time_col].dt.to_period("M").dt.to_timestamp()  # monthly alignment
    assert_unique(panel, [unit_id_col, time_col], "water_panel_df")

    # --- Geo prep
    units = units_gdf[[unit_id_col, "geometry"] + [c for c in [basin_col, aquifer_col, utility_col] if c in units_gdf.columns]].copy()
    units = units.dropna(subset=[unit_id_col, "geometry"])
    units = units.set_index(unit_id_col, drop=False)

    # --- Data centers to GeoDataFrame
    dc = dc_df.copy()
    dc = ensure_datetime(dc, "open_date")
    if "geometry" not in dc.columns:
        # If you have lon/lat columns, create geometry here:
        # dc["geometry"] = gpd.points_from_xy(dc["lon"], dc["lat"])
        raise ValueError("dc_df must include a 'geometry' column with shapely Points.")
    dc_gdf = gpd.GeoDataFrame(dc, geometry="geometry", crs=units_gdf.crs)

    # --- Assign DCs to units
    dc_assigned = spatial_assign_datacenters_to_units(units.reset_index(drop=True), dc_gdf, unit_id_col=unit_id_col)

    # --- Panel index
    panel_idx = pd.MultiIndex.from_frame(panel[[unit_id_col, time_col]].copy())
    panel_idx.names = [unit_id_col, time_col]

    # --- Treatment dose D_it (capacity cumulative)
    D_it = build_unit_time_treatment(
        panel_index=panel_idx,
        dc_assigned_df=dc_assigned,
        unit_id_col=unit_id_col,
        open_col="open_date",
        capacity_col="capacity_mw",
        date_level_name=time_col,
        cap_transform=config.cap_transform,
    )

    # --- Spillover matrices
    # Priority: hydrologic edges (best), else institutional (aquifer/utility), else contiguity.
    W_specs: Dict[str, Tuple[np.ndarray, List]] = {}

    if hydro_edges is not None:
        W_hydro, order_hydro = build_W_network_edges(
            units.reset_index(),
            edges=hydro_edges,
            unit_id_col=unit_id_col,
            src_col="src_unit_id",
            dst_col="dst_unit_id",
            weight_col=None,
        )
        W_specs["W_hydro"] = (W_hydro, order_hydro)

    if aquifer_col in units.columns and units[aquifer_col].notna().any():
        W_aq, order_aq = build_W_within_group(units.reset_index(), group_col=aquifer_col, unit_id_col=unit_id_col)
        W_specs["W_aquifer"] = (W_aq, order_aq)

    if utility_col in units.columns and units[utility_col].notna().any():
        W_ut, order_ut = build_W_within_group(units.reset_index(), group_col=utility_col, unit_id_col=unit_id_col)
        W_specs["W_utility"] = (W_ut, order_ut)

    if len(W_specs) == 0:
        if HAS_LIBPYSAL:
            Wq, order_q = build_W_contiguity(units.reset_index(), unit_id_col=unit_id_col, kind="queen")
            W_specs["W_queen"] = (Wq, order_q)
        else:
            raise RuntimeError("No spillover mapping provided and libpysal not available.")

    # --- Build analysis DataFrame with D_it
    panel2 = panel.copy()
    panel2["D_it"] = D_it.values

    # Clusters DataFrame aligned to panel index
    clusters_df = None
    if config.cluster_on:
        cl = units.reset_index(drop=True)[[unit_id_col] + config.cluster_on].copy()
        panel2 = panel2.merge(cl, on=unit_id_col, how="left")
        clusters_df = panel2.set_index([unit_id_col, time_col])[config.cluster_on]

    # --- Cohort & event-time for event-study
    Ti = first_treatment_date(dc_assigned, unit_id_col=unit_id_col, open_col="open_date")
    et = build_event_time(panel2, Ti, unit_col=unit_id_col, time_col=time_col)

    # Define cohorts as first treatment date (monthly)
    cohorts = Ti.copy()
    cohorts = pd.to_datetime(cohorts).dt.to_period("M").dt.to_timestamp()

    # Convert to panel-indexed objects for estimation
    panel_ix = set_panel_index(panel2, unit_id_col, time_col)

    results: Dict[str, object] = {}

    # ------------------------
    # SPEC A: Baseline DiD with spillover term(s)
    # ------------------------
    for yname in outcome_cols:
        y = panel_ix[yname].astype(float)

        # Base X: direct dose + controls
        X_base = pd.DataFrame(index=panel_ix.index)
        X_base["D_it"] = panel_ix["D_it"].astype(float)

        for c in control_cols:
            X_base[c] = panel_ix[c].astype(float)

        # Optional unit trends
        if config.unit_trends:
            # create linear time trend interacted with unit FE implicitly (approx via demeaning)
            # Better: create unit-specific trend by including (t) and absorbing FE; skeleton only.
            tnum = panel_ix.index.get_level_values(1).map(pd.Timestamp.toordinal).astype(float)
            X_base["t"] = tnum

        # Fit without spillover
        res0 = fit_panel_ols(y, X_base, entity_effects=True, time_effects=True, clusters=clusters_df)
        results[f"{yname}__did_no_spill"] = res0

        # Fit with each spillover mapping
        for wname, (W, order) in W_specs.items():
            WD = build_WD_panel(D=panel_ix["D_it"], W=W, unit_order=order, panel_index=panel_ix.index)
            X = X_base.copy()
            X[f"{wname}_D"] = WD.astype(float)

            res = fit_panel_ols(y, X, entity_effects=True, time_effects=True, clusters=clusters_df)
            results[f"{yname}__did_{wname}"] = res

    # ------------------------
    # SPEC B: Event-study (pooled ET dummies) + spillover
    # Note: This is not fully Sun–Abraham unless you use the cohort interactions matrix;
    # but it’s a clean starting point. Cohort interactions matrix can be huge.
    # ------------------------
    # Pooled event time dummies
    event_X = pd.DataFrame(index=panel_ix.index)
    et_aligned = et.reindex(panel_ix.index)
    for k in range(config.event_min, config.event_max + 1):
        if k == config.event_omit:
            continue
        event_X[f"ET_{k}"] = (et_aligned == k).astype(float)

    # You may also include pre-period "never treated" handling:
    # - units with no Ti have et = NaN => all ET dummies 0, which is fine.

    for yname in outcome_cols:
        y = panel_ix[yname].astype(float)

        X_base = event_X.copy()
        for c in control_cols:
            X_base[c] = panel_ix[c].astype(float)

        res_es = fit_panel_ols(y, X_base, entity_effects=True, time_effects=True, clusters=clusters_df)
        results[f"{yname}__eventstudy_pooled"] = res_es

        for wname, (W, order) in W_specs.items():
            # spillover exposure of dose in each period
            WD = build_WD_panel(D=panel_ix["D_it"], W=W, unit_order=order, panel_index=panel_ix.index)
            X = X_base.copy()
            X[f"{wname}_D"] = WD.astype(float)

            res_es_w = fit_panel_ols(y, X, entity_effects=True, time_effects=True, clusters=clusters_df)
            results[f"{yname}__eventstudy_pooled_{wname}"] = res_es_w

    # ------------------------
    # SPEC C (optional, heavy): Sun–Abraham-style cohort interactions
    # Warning: can explode in dimensionality. Consider coarsening cohorts to year.
    # ------------------------
    # Uncomment if you really want it.
    # sa_X = sun_abraham_design_matrix(
    #     panel_index=panel_ix.index,
    #     event_time=et,
    #     cohorts=cohorts,
    #     event_min=config.event_min,
    #     event_max=config.event_max,
    #     omit=config.event_omit,
    # )
    # for yname in outcome_cols:
    #     y = panel_ix[yname].astype(float)
    #     X = sa_X.copy()
    #     for c in control_cols:
    #         X[c] = panel_ix[c].astype(float)
    #     res_sa = fit_panel_ols(y, X, entity_effects=True, time_effects=True, clusters=clusters_df)
    #     results[f"{yname}__sunabraham"] = res_sa

    return results


# ----------------------------
# Example usage (edit to your data)
# ----------------------------
if __name__ == "__main__":
    cfg = AnalysisConfig(
        event_min=-24,
        event_max=36,
        event_omit=-1,
        cap_transform="log1p",
        unit_trends=False,
        cluster_on=["basin_id"],  # change to ["aquifer_id"] or ["state"] etc.
    )

    # Load your data here:
    # units_gdf = gpd.read_file("units_polygons.gpkg")  # must contain unit_id + geometry
    # water_panel_df = pd.read_parquet("water_panel.parquet")  # unit_id, date, outcomes, controls
    # dc_df = gpd.read_file("datacenters.gpkg")  # dc_id, open_date, capacity_mw, geometry (Point)

    # Optional hydrologic edges:
    # hydro_edges = pd.read_parquet("hydro_edges.parquet")  # src_unit_id, dst_unit_id

    # results = run_analysis(
    #     units_gdf=units_gdf,
    #     water_panel_df=water_panel_df,
    #     dc_df=dc_df,
    #     config=cfg,
    #     unit_id_col="unit_id",
    #     time_col="date",
    #     outcome_cols=["availability", "usage", "price"],
    #     control_cols=["precip", "temp", "drought_index", "population", "ag_withdrawals"],
    #     basin_col="basin_id",
    #     aquifer_col="aquifer_id",
    #     utility_col="utility_id",
    #     hydro_edges=hydro_edges,
    # )

    # Print a couple summaries
    # for k, v in results.items():
    #     print("\n" + "=" * 80)
    #     print(k)
    #     print(v.summary)

"""
Notes / next steps you’ll likely add:
1) Utilization uncertainty:
   - Treat D_it as "installed capacity" (intent-to-treat).
   - Add proxy for load (grid substation load changes, transformer capacity upgrades, etc.) if available.
2) Training windows:
   - Add a second dose: D_train_it = capacity * 1{training window active}
   - Then include both D_it and D_train_it (and their spillovers) to separate baseline vs spikes.
3) Better water-physics exposure:
   - Replace contiguity W with downstream W from flow networks (NHDPlus/HydroBASINS),
     and aquifer W from hydrogeologic units (USGS aquifers / local GMD).
4) Inference:
   - Conley SEs (spatial HAC) are not built into PanelOLS; if you need them,
     you can export residuals and implement Conley manually (or use `pyfixest` if installed).
5) Multiple outcomes:
   - Correct for multiple testing or focus on primary endpoint (availability) + pre-specified secondary endpoints.
"""