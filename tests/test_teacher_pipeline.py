"""Day 20 batch generation, process policy, metadata, and dataset checks."""

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
from powered_landing_guidance.teacher_pipeline import (  # noqa: E402
    TeacherPipelineSettings,
    initial_guess_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def _run_cli(config: dict, tmp_path: Path, name: str) -> tuple[subprocess.CompletedProcess, Path]:
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    destination = tmp_path / name
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_teacher_pipeline.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(destination),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return completed, destination


def test_default_policy_is_reproducible_and_warm_then_pid() -> None:
    settings = TeacherPipelineSettings.from_config(CONFIG)

    assert settings.episodes == settings.minimum_cases == 100
    assert settings.seed == 20260920
    assert settings.runner == "sequential_subprocess"
    assert initial_guess_plan(False, 1) == ("pid", "pid")
    assert initial_guess_plan(True, 1) == ("previous_success", "pid")
    with pytest.raises(ValueError, match="max_retries"):
        initial_guess_plan(True, -1)
    with pytest.raises(ValueError, match="retry_initial_guess"):
        initial_guess_plan(True, 1, "none")


def test_cli_generates_compact_dataset_and_refuses_to_overwrite(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 30
    config["teacher_pipeline"].update(
        {
            "episodes": 2,
            "minimum_cases": 2,
            "seed": 20260920,
            "attempt_timeout_s": 20.0,
            "max_retries": 1,
            "worker_restart_after_attempts": 2,
        }
    )
    first, destination = _run_cli(config, tmp_path, "small-batch")

    assert first.returncode == 0, first.stderr
    report = json.loads((destination / "teacher-pipeline.json").read_text(encoding="utf-8"))
    assert report["status"] in {"complete", "completed_with_failures"}
    assert report["summary"]["executed_cases"] == 2
    assert report["completion_gate"]["passed"] is True
    assert len(report["cases"]) == 2
    assert len(report["failures"]) == report["summary"]["failed_cases"]
    with np.load(destination / "teacher-trajectories.npz", allow_pickle=False) as dataset:
        assert set(dataset.files) == {
            "case_indices",
            "initial_states",
            "times_s",
            "states_x_z_vx_vz_theta_omega_mass",
            "controls_throttle_gimbal_rad",
            "durations_s",
        }
        successes = report["summary"]["successful_cases"]
        assert dataset["case_indices"].shape == (successes,)
        assert dataset["initial_states"].shape == (successes, 7)
        assert dataset["controls_throttle_gimbal_rad"].shape == (successes, 30, 2)

    command = [
        sys.executable,
        str(ROOT / "scripts/generate_teacher_pipeline.py"),
        "--config",
        str(tmp_path / "small-batch.yaml"),
        "--output-dir",
        str(destination),
    ]
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_timeout_is_enforced_retried_and_recorded(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 20
    config["teacher_pipeline"].update(
        {
            "episodes": 1,
            "minimum_cases": 1,
            "attempt_timeout_s": 0.001,
            "max_retries": 1,
            "worker_restart_after_attempts": 5,
        }
    )
    completed, destination = _run_cli(config, tmp_path, "timeout")

    assert completed.returncode == 0, completed.stderr
    report = json.loads((destination / "teacher-pipeline.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_with_failures"
    assert report["summary"]["executed_cases"] == 1
    assert report["summary"]["failed_cases"] == 1
    assert report["summary"]["retried_cases"] == 1
    assert report["summary"]["timeout_attempts"] == 2
    assert len(report["failures"]) == 1
    assert [attempt["status"] for attempt in report["failures"][0]["attempts"]] == [
        "timeout",
        "timeout",
    ]


def test_solver_failure_preserves_stage_status_and_constraint_diagnostics(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 20
    config["optimal_control"]["duration_s"] = [0.1, 0.2]
    config["teacher_pipeline"].update(
        {
            "episodes": 1,
            "minimum_cases": 1,
            "attempt_timeout_s": 20.0,
            "max_retries": 0,
        }
    )
    completed, destination = _run_cli(config, tmp_path, "solver-failure")

    assert completed.returncode == 0, completed.stderr
    report = json.loads((destination / "teacher-pipeline.json").read_text(encoding="utf-8"))
    attempt = report["failures"][0]["attempts"][0]
    assert attempt["status"] == "solver_failure"
    assert attempt["stage"] == "A"
    assert attempt["solver_status"]
    assert attempt["diagnostics"]["worst_constraint"]
    assert attempt["diagnostics"]["worst_violation"] > 0.0
