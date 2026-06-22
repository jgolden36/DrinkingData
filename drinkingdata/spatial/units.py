"""Assign data centers to spatial units via point-in-polygon (CLAUDE.md §5.2).

Generalizes the county point-in-polygon join in ``TexasWaterAnalysis.py`` /
``WaterAnalysis.py`` to any unit polygons — the target for the paper is the
NWM/NHDPlus HUC-8 hydrofabric (with HUC-12 for robustness), but the same code
serves county×supplier cells. Imports ``geopandas`` lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd


def assign_datacenters_to_units(
    units_gdf: "gpd.GeoDataFrame",
    dc_gdf: "gpd.GeoDataFrame",
    *,
    unit_col: str = "unit_id",
    predicate: str = "within",
) -> "gpd.GeoDataFrame":
    """Return ``dc_gdf`` with a ``unit_col`` assigned by point-in-polygon.

    Reprojects the data centers to the units' CRS first. Warns (does not fail) on
    data centers that fall outside every polygon, since those are typically
    offshore/erroneous coordinates worth inspecting rather than silently dropping.
    """
    import geopandas as gpd

    if dc_gdf.crs != units_gdf.crs:
        dc_gdf = dc_gdf.to_crs(units_gdf.crs)

    joined = gpd.sjoin(
        dc_gdf,
        units_gdf[[unit_col, "geometry"]],
        how="left",
        predicate=predicate,
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])

    n_unmatched = int(joined[unit_col].isna().sum())
    if n_unmatched:
        import warnings

        warnings.warn(
            f"{n_unmatched} data center(s) did not fall within any unit polygon "
            f"(check coordinates / CRS).",
            stacklevel=2,
        )
    return joined
