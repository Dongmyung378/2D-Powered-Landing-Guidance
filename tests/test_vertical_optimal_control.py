"""Day 16 vertical NLP convergence, constraints, rollout, and saved outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem  # noqa: E402
from powered_landing_guidance.vertical_optimal_control import (  # noqa: E402
    VerticalSolveError,
    initial_guess,
    solve_vertical_landing,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/default.yaml")
INITIAL = (0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)


def problem(intervals: int = 40) -> LandingOptimalControlProblem:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = intervals
    return LandingOptimalControlProblem.from_config(config, INITIAL)


def test_initial_guess_has_consistent_mesh_and_bounded_controls() -> None:
    model = problem()
    states, throttle, duration = initial_guess(model)
    assert states.shape == (model.intervals + 1, 3)
    assert throttle.shape == (model.intervals,)
    np.testing.assert_array_equal(states[0], (100.0, -20.0, 1000.0))
    assert model.duration_bounds_s[0] <= duration <= model.duration_bounds_s[1]
    assert np.all((throttle >= 0.0) & (throttle <= 1.0))
    assert np.all(np.isfinite(states))


def test_two_stage_solver_converges_and_matches_independent_simulator() -> None:
    model = problem()
    result = solve_vertical_landing(model)
    assert result.stage_a.status == "Solve_Succeeded"
    assert result.stage_b.status == "Solve_Succeeded"
    assert result.stage_a.max_terminal_violation <= model.feasibility_tolerance
    assert result.stage_b.max_hard_violation <= model.feasibility_tolerance
    assert result.states.shape == (model.intervals + 1, 3)
    assert result.throttle.shape == (model.intervals,)
    assert abs(result.states[-1, 0] - model.ground_z_m) < 1e-6
    assert -model.terminal_limits[2] <= result.states[-1, 1] <= 0.0
    assert np.min(result.states[:, 0]) >= model.ground_z_m - 1e-6
    assert np.min(result.states[:, 2]) >= model.parameters.dry_mass_kg + 1.0 - 1e-6
    assert np.max(np.abs(result.rollout_states[-1] - result.states[-1])) < 1e-3
    assert np.min(result.rollout_states[:, 0]) >= model.ground_z_m - 1e-3


def test_vertical_solver_rejects_nonvertical_state_and_bad_rollout_step() -> None:
    model = problem()
    with pytest.raises(ValueError, match="integration_step_s"):
        solve_vertical_landing(model, integration_step_s=0.0)
    nonvertical = (*INITIAL[:2], 1.0, *INITIAL[3:])
    with pytest.raises(ValueError, match="zero horizontal"):
        solve_vertical_landing(LandingOptimalControlProblem.from_config(CONFIG, nonvertical))


def test_infeasible_horizon_preserves_stage_and_violation_diagnostics() -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 20
    config["optimal_control"]["duration_s"] = [0.1, 0.2]
    model = LandingOptimalControlProblem.from_config(config, INITIAL)
    with pytest.raises(VerticalSolveError, match="feasibility_tolerance_not_met") as caught:
        solve_vertical_landing(model)
    assert caught.value.stage == "A"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.max_terminal_violation > 1.0


def test_command_writes_report_and_plot_without_overwriting(tmp_path: Path) -> None:
    output_dir = tmp_path / "vertical"
    command = [
        sys.executable,
        str(ROOT / "scripts/solve_vertical_landing.py"),
        "--output-dir",
        str(output_dir),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report = json.loads((output_dir / "vertical-teacher.json").read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["simulator_validation"]["passed"] is True
    assert report["stage_b"]["max_hard_violation"] < 1e-3
    assert report["simulator_validation"]["max_node_disagreement"] < 1e-3
    assert (output_dir / "vertical-teacher.png").stat().st_size > 10_000
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
