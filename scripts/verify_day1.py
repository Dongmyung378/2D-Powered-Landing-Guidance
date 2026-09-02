"""Verify the Day 1 environment, package import, and default configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy
import scipy
import yaml

from powered_landing_guidance import ACTION_NAMES, STATE_NAMES, load_config


def main() -> int:
    expected_python = (3, 12, 7)
    actual_python = sys.version_info[:3]
    if actual_python != expected_python:
        print(f"FAIL: expected Python {expected_python}, found {actual_python}")
        return 1

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {numpy.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"State ({len(STATE_NAMES)}): {', '.join(STATE_NAMES)}")
    print(f"Control ({len(ACTION_NAMES)}): {', '.join(ACTION_NAMES)}")
    print(f"Config schema: {config['schema_version']}")
    print("Day 1 verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
