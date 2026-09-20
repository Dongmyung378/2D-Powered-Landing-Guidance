"""Day 19 objective-weight, mesh, replay, and artifact checks."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("casadi")

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.optimal_control_study import StudySettings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def test_study_settings_convert_angular_tolerances_to_radians() -> None:
    settings = StudySettings.from_config(CONFIG)

    assert settings.mesh_intervals == (25, 50, 100, 200)
    assert settings.replay_dt_s == pytest.approx(0.005)
    assert settings.tolerance_by_state["x"] == pytest.approx(0.001)
    assert settings.tolerance_by_state["theta"] == pytest.approx(np.deg2rad(0.01))
    assert [profile.name for profile in settings.objective_profiles] == [
        "fuel_priority",
        "baseline",
        "smooth_control",
    ]


def test_cli_runs_all_sweeps_and_refuses_to_overwrite(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 25
    config["optimal_control"]["study"]["mesh_intervals"] = [20, 25, 30]
    config["optimal_control"]["study"]["replay_dt_s"] = 0.01
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    destination = tmp_path / "study"
    command = [
        sys.executable,
        str(ROOT / "scripts/analyze_optimal_control.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(destination),
    ]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report = json.loads((destination / "day19-study.json").read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["completion_gate"]["passed"] is True
    assert report["objective_scales"]["throttle_rate_per_s"] == pytest.approx(2.0)
    assert np.rad2deg(report["objective_scales"]["gimbal_rate_rad_s"]) == pytest.approx(60.0)
    assert len(report["tradeoff_cases"]) == 3
    assert [case["intervals"] for case in report["mesh_cases"]] == [20, 25, 30]
    assert all(case["replay_passed"] for case in report["tradeoff_cases"])
    assert all(case["replay_passed"] for case in report["mesh_cases"])
    assert (destination / "day19-study.png").stat().st_size > 10_000

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
