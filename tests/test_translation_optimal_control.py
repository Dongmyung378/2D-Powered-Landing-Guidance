"""Day 17 translational NLP, PID seed, limits, and saved trajectory tests."""

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
from powered_landing_guidance.translation_optimal_control import (  # noqa: E402
    TranslationSolveError,
    _rhs_symbolic,
    analytic_initial_guess,
    pid_initial_guess,
    solve_translation_landing,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")
INITIAL = (10.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)


def problem(
    *, intervals: int = 40, initial: tuple[float, ...] = INITIAL
) -> LandingOptimalControlProblem:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = intervals
    return LandingOptimalControlProblem.from_config(config, initial)


def test_ideal_vector_rhs_matches_planar_simulator_in_translation() -> None:
    model = problem()
    state = np.asarray((10.0, 80.0, -1.0, -15.0, 975.0))
    action = np.asarray((0.7, np.deg2rad(-10.0)))
    symbolic_state = ca.MX.sym("state", 5)
    symbolic_action = ca.MX.sym("action", 2)
    rhs = ca.Function(
        "rhs",
        [symbolic_state, symbolic_action],
        [_rhs_symbolic(symbolic_state, symbolic_action, model)],
    )
    actual = np.asarray(rhs(state, action)).reshape(-1)
    full_state = np.asarray((state[0], state[1], state[2], state[3], action[1], 0.0, state[4]))
    full = state_derivative(0.0, full_state, (action[0], 0.0), model.parameters)
    np.testing.assert_allclose(actual, full[[0, 1, 2, 3, 6]], atol=1e-14)
    assert actual[2] < 0.0


def test_analytic_and_pid_initial_guesses_have_bounded_controls() -> None:
    model = problem()
    for guess in (analytic_initial_guess(model), pid_initial_guess(model, CONFIG)):
        assert guess.states.shape == (model.intervals + 1, 5)
        assert guess.controls.shape == (model.intervals, 2)
        assert np.all(np.isfinite(guess.states))
        assert np.all((guess.controls[:, 0] >= 0) & (guess.controls[:, 0] <= 1))
        assert np.max(np.abs(guess.controls[:, 1])) <= model.max_abs_thrust_angle_rad
        np.testing.assert_array_equal(guess.states[0], (10.0, 100.0, 0.0, -20.0, 1000.0))
    assert pid_initial_guess(model, CONFIG).pid_outcome == "success"


@pytest.mark.parametrize("guess_source", ["analytic", "pid"])
def test_planar_translation_corrects_horizontal_error(guess_source: str) -> None:
    model = problem()
    result = solve_translation_landing(
        model,
        guess_source=guess_source,
        pid_config=CONFIG if guess_source == "pid" else None,
    )
    final = result.rollout_states[-1]
    assert result.stage_a.status == "Solve_Succeeded"
    assert result.stage_b.status == "Solve_Succeeded"
    assert result.stage_a.max_terminal_violation <= model.feasibility_tolerance
    assert result.stage_b.max_hard_violation <= model.feasibility_tolerance
    assert abs(final[0] - model.target_x_m) < 0.1
    assert abs(final[2]) < 0.1
    assert abs(final[1] - model.ground_z_m) < 1e-3
    assert -model.terminal_limits[2] <= final[3] <= 0.0
    assert np.min(result.rollout_states[:, 1]) >= model.ground_z_m - 1e-3
    assert np.max(np.abs(result.rollout_states[-1] - result.states[-1])) < 1e-3
    assert np.max(np.abs(result.controls[:, 1])) <= model.max_abs_thrust_angle_rad + 1e-6
    assert result.guess_source == guess_source
    if guess_source == "pid":
        assert result.pid_outcome == "success"


def test_solver_rejects_missing_pid_config_and_nonzero_attitude() -> None:
    model = problem()
    with pytest.raises(ValueError, match="pid_config"):
        solve_translation_landing(model, guess_source="pid")
    tilted = (*INITIAL[:4], 0.1, 0.0, INITIAL[6])
    with pytest.raises(ValueError, match="zero initial attitude"):
        solve_translation_landing(problem(initial=tilted))


def test_impossible_time_horizon_keeps_violation_diagnostics() -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 20
    config["optimal_control"]["duration_s"] = [0.1, 0.2]
    model = LandingOptimalControlProblem.from_config(config, INITIAL)
    with pytest.raises(TranslationSolveError) as caught:
        solve_translation_landing(model)
    assert caught.value.stage == "A"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.max_terminal_violation > 1.0


def test_cli_writes_report_plot_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "translation"
    command = [
        sys.executable,
        str(ROOT / "scripts/solve_translation_landing.py"),
        "--output-dir",
        str(destination),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report = json.loads((destination / "translation-teacher.json").read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["simulator_validation"]["passed"] is True
    assert report["simulator_validation"]["horizontal_error_reduction_m"] > 9.0
    assert report["simulator_validation"]["max_node_disagreement"] < 1e-3
    assert (destination / "translation-teacher.png").stat().st_size > 10_000
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
