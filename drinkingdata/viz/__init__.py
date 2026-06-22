"""Visualization helpers (event-study paths, dose-response, choropleths).

Consolidates the figure-making logic scattered across the legacy ``*Visualization``
and ``*_map`` scripts. ``matplotlib`` / ``geopandas`` are imported lazily inside
each function so importing the package never pulls in the plotting stack.

NOTE (CLAUDE.md §6): synthetic-geography scripts are presentation-only and must
never be used as analytic spatial units. This package plots *real* units and
estimation outputs only.
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = ["plot_event_study"]


def plot_event_study(
    coefs: "Sequence[float]",
    event_times: "Sequence[int]",
    *,
    lower: "Optional[Sequence[float]]" = None,
    upper: "Optional[Sequence[float]]" = None,
    title: str = "Event-study estimates",
    ylabel: str = "Effect on water outcome",
    ax=None,
):
    """Plot dynamic {β_k} (or {θ_k}) with a confidence band and reference lines.

    Returns the matplotlib Axes. Draws a horizontal zero line and a vertical line
    at event time 0 (the treatment/release period).
    """
    import matplotlib.pyplot as plt  # lazy

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    ax.plot(event_times, coefs, marker="o", color="#1f4e79", zorder=3)
    if lower is not None and upper is not None:
        ax.fill_between(event_times, lower, upper, alpha=0.2, color="#1f4e79", zorder=1)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Event time (periods since model release)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
