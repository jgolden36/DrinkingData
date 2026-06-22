"""Command-line entry point for the Drinking Data pipeline.

Currently exposes scaffolding-level commands:

* ``drinkingdata config``  — show the resolved configuration & data paths.
* ``drinkingdata check``   — import the package and report which optional
  (heavy) dependencies are available.

As the pipeline modules are wired to real data, subcommands for ``build-panel``,
``estimate``, and ``robustness`` will be added here.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import List, Optional

from .config import Config

_OPTIONAL_DEPS = [
    "geopandas", "shapely", "statsmodels", "linearmodels",
    "libpysal", "spreg", "matplotlib", "seaborn", "openpyxl",
]


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    print(f"data_dir   : {cfg.data_dir}")
    print(f"outputs_dir: {cfg.outputs_dir}")
    print(f"run        : {cfg.run}")
    print("paths:")
    for key, value in cfg.paths.items():
        marker = " " if value else "·"  # · = unconfigured (per roadmap)
        print(f"  [{marker}] {key}: {value}")
    return 0


def _cmd_check(_: argparse.Namespace) -> int:
    import drinkingdata

    print(f"drinkingdata {drinkingdata.__version__} imported OK")
    print("optional dependencies:")
    for dep in _OPTIONAL_DEPS:
        ok = importlib.util.find_spec(dep) is not None
        print(f"  [{'x' if ok else ' '}] {dep}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drinkingdata", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="Show resolved configuration")
    p_config.set_defaults(func=_cmd_config)

    p_check = sub.add_parser("check", help="Report import status of dependencies")
    p_check.set_defaults(func=_cmd_check)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
