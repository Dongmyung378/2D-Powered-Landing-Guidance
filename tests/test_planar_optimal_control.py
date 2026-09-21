"""Day 18 full planar NLP, physical limits, replay, and diagnostics tests."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.dynamics import state_derivative  # noqa: E402
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem  # noqa: E402
from powered_landing_guidance.planar_optimal_control import (  # noqa: E402
    PlanarSolveError,
    _rhs_symbolic,
    control_sequence_initial_guess,
    pid_initial_guess,
    solve_planar_landing,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")
INITIAL = (
    10.0,
    100.0,
    0.0,
    -20.0,
    np.deg2rad(3.0),
    np.deg2rad(-1.0),
    1000.0,
)


def problem(
    *,
    intervals: int = 40,
    initial: tuple[float, ...] = INITIAL,
) -> LandingOptimalControlProblem:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = intervals
    return LandingOptimalControlProblem.from_config(config, initial)


def test_full_symbolic_rhs_matches_planar_simulator() -> None:
    model = problem()
    state = np.asarray((8.0, 70.0, -1.0, -12.0, 0.12, -0.03, 965.0))
    action = np.asarray((0.72, np.deg2rad(-4.0)))
    symbolic_state = ca.MX.sym("state", 7)
    symbolic_action = ca.MX.sym("action", 2)
    rhs = ca.Function(
        "full_planar_rhs",
        [symbolic_state, symbolic_action],
        [_rhs_symbolic(symbolic_state, symbolic_action, model)],
    )

    actual = np.asarray(rhs(state, action)).reshape(-1)
    expected = state_derivative(0.0, state, action, model.parameters)

    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual[5] > 0.0


def test_pid_guess_contains_full_dynamics_and_bounded_controls() -> None:
    model = problem()
    guess = pid_initial_guess(model, CONFIG)

    assert guess.source == "pid"
    assert guess.source_outcome == "success"
    assert guess.states.shape == (model.intervals + 1, 7)
    assert guess.controls.shape == (model.intervals, 2)
    assert np.all(np.isfinite(guess.states))
    assert np.all((guess.controls[:, 0] >= 0.0) & (guess.controls[:, 0] <= 1.0))
    assert np.max(np.abs(guess.controls[:, 1])) <= model.parameters.gimbal_limit_rad
    np.testing.assert_array_equal(guess.states[0], INITIAL)
    defects = model.rk4_defects(guess.states, guess.controls, guess.duration_s)
    np.testing.assert_allclose(defects, 0.0, atol=1e-12)


def test_control_sequence_warm_start_reintegrates_from_new_initial_state() -> None:
    original = problem()
    pid_guess = pid_initial_guess(original, CONFIG)
    shifted_initial = (
        -8.0,
        105.0,
        1.0,
        -19.0,
        np.deg2rad(-2.0),
        np.deg2rad(1.0),
        980.0,
    )
    shifted = problem(initial=shifted_initial)

    warm = control_sequence_initial_guess(
        shifted,
        pid_guess.controls,
        pid_guess.duration_s,
        source="previous_success",
        source_outcome="case_0000",
    )

    assert warm.source == "previous_success"
    assert warm.source_outcome == "case_0000"
    np.testing.assert_allclose(warm.states[0], shifted_initial)
    np.testing.assert_allclose(
        shifted.rk4_defects(warm.states, warm.controls, warm.duration_s),
        0.0,
        atol=1e-12,
    )

    invalid = pid_guess.controls.copy()
    invalid[0, 0] = 1.1
    with pytest.raises(ValueError, match="throttle"):
        control_sequence_initial_guess(
            shifted,
            invalid,
            pid_guess.duration_s,
            source="previous_success",
        )


def test_full_planar_solver_lands_with_attitude_and_rate_limits() -> None:
    model = problem()
    result = solve_planar_landing(model, CONFIG)
    final = result.rollout_states[-1]

    assert result.stage_a.status == "Solve_Succeeded"
    assert result.stage_b.status == "Solve_Succeeded"
    assert result.stage_a.max_terminal_violation <= model.feasibility_tolerance
    assert result.stage_b.max_hard_violation <= model.feasibility_tolerance
    objective = model.objective(result.states[-1], result.controls, result.duration_s)
    assert result.stage_b.objective == pytest.approx(objective.weighted_total, abs=1e-8)
    assert np.max(model.terminal_violations(final)) <= model.feasibility_tolerance
    assert np.min(result.rollout_states[:, 1]) >= model.ground_z_m - 1e-3
    assert np.max(np.abs(result.rollout_states[:, 4])) <= model.max_abs_tilt_rad + 1e-6
    assert np.max(np.abs(result.rollout_states[:, 5])) <= model.max_abs_angular_rate_rad_s + 1e-6
    assert np.max(np.abs(result.controls[:, 1])) <= model.parameters.gimbal_limit_rad + 1e-6
    interpolated = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(7)
        ]
    )
    assert np.max(np.abs(interpolated - result.states)) < 1e-3


def test_initial_attitude_and_rate_must_satisfy_path_limits() -> None:
    tilted = (*INITIAL[:4], np.deg2rad(21.0), INITIAL[5], INITIAL[6])
    with pytest.raises(ValueError, match="initial attitude"):
        solve_planar_landing(problem(initial=tilted), CONFIG)

    spinning = (*INITIAL[:5], np.deg2rad(31.0), INITIAL[6])
    with pytest.raises(ValueError, match="initial angular rate"):
        solve_planar_landing(problem(initial=spinning), CONFIG)


def test_impossible_horizon_reports_named_constraint_violation() -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 20
    config["optimal_control"]["duration_s"] = [0.1, 0.2]
    model = LandingOptimalControlProblem.from_config(config, INITIAL)

    with pytest.raises(PlanarSolveError) as caught:
        solve_planar_landing(model, config)

    error = caught.value
    assert error.stage == "A"
    assert error.diagnostics is not None
    assert error.diagnostics.worst_constraint
    assert error.diagnostics.worst_violation > 1.0
    assert set(error.diagnostics.terminal_violations) == {
        "x",
        "z",
        "vx",
        "vz",
        "theta",
        "omega",
    }


def test_cli_writes_validated_report_and_plot_without_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "planar"
    command = [
        sys.executable,
        str(ROOT / "scripts/solve_planar_landing.py"),
        "--output-dir",
        str(destination),
    ]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report = json.loads((destination / "planar-3dof-teacher.json").read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["simulator_validation"]["passed"] is True
    assert report["stage_a"]["status"] == "Solve_Succeeded"
    assert report["stage_b"]["status"] == "Solve_Succeeded"
    assert (destination / "planar-3dof-teacher.png").stat().st_size > 10_000

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
