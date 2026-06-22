"""Drinking Data — spatial econometric analysis of AI data centers' impacts on
water resources in Texas and California (Golden & Varshney).

This package implements the micro (spatial-DiD) stage of the three-stage
architecture described in ``CLAUDE.md`` and the paper ``DrinkingData (2).tex``,
plus the indirect-water accounting and techno-economic benchmark that feed the
central gap-decomposition deliverable.

Heavy geospatial / spatial-econometrics dependencies (geopandas, linearmodels,
libpysal, spreg) are imported lazily inside the functions that need them, so the
package and its pure-numeric modules import cleanly without the full stack.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "Config"]


def __getattr__(name: str):
    # Lazy re-export so ``from drinkingdata import Config`` works without
    # importing submodules at package import time.
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
