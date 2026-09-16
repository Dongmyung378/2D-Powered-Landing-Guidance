"""Day 15 formulation, simulator parity, constraints, and objective checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import frozen_baseline_initial_states, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    simulate_planar,
    state_derivative,
)
from powered_landing_guidance.model import State
from powered_landing_guidance.optimal_control import (
    PATH_MARGIN_NAMES,
    TERMINAL_RESIDUAL_NAMES,
    LandingOptimalControlProblem,
)

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs/pid-baseline-v1.yaml")
INITIAL = np.asarray((10.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0))


def problem() -> LandingOptimalControlProblem:
    return LandingOptimalControlProblem.from_config(CONFIG, INITIAL)


def test_nominal_ocp_dynamics_match_simulator_in_smooth_region() -> None:
    model = problem()
    parameters = PlanarDynamicsParameters.from_config(CONFIG)
    state = np.asarray((12.0, 80.0, -1.0, -10.0, 0.12, -0.02, 970.0))
    action = np.asarray((0.65, np.deg2rad(-4.0)))

    np.testing.assert_allclose(
        model.dynamics(state, action), state_derivative(0.0, state, action, parameters), atol=1e-14
    )
    assert model.dynamics(state, action)[6] < 0.0
    assert model.dynamics(state, np.asarray((0.0, 0.0)))[3] == pytest.approx(
        -parameters.gravity_m_s2
    )


def test_problem_uses_the_frozen_comparison_initial_state() -> None:
    initial_states = frozen_baseline_initial_states(CONFIG)
    model = LandingOptimalControlProblem.from_config(CONFIG, initial_states[0])

    np.testing.assert_array_equal(model.initial_state.as_array(), initial_states[0])
    assert model.intervals == 100
    assert model.duration_bounds_s == (5.0, 25.0)


def test_path_and_terminal_constraints_follow_simulator_limits() -> None:
    model = problem()
    p = model.parameters
    margins = model.path_margins(INITIAL, (0.5, 0.0))
    assert len(margins) == len(PATH_MARGIN_NAMES) == 6
    assert np.all(margins >= 0.0)

    depleted = INITIAL.copy()
    depleted[6] = p.dry_mass_kg
    assert model.path_margins(depleted, (0.5, 0.0))[1] == pytest.approx(-1.0)
    assert model.path_margins(INITIAL, (1.1, 0.0))[3] < 0.0
    assert model.path_margins(INITIAL, (0.5, p.gimbal_limit_rad + 0.01))[5] < 0.0

    terminal = np.asarray((0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 900.0))
    assert len(model.terminal_violations(terminal)) == len(TERMINAL_RESIDUAL_NAMES) == 6
    np.testing.assert_array_equal(model.terminal_violations(terminal), 0.0)
    assert model.feasibility_score(terminal) == 0.0
    assert model.terminal_is_feasible(terminal)

    missed = terminal.copy()
    missed[0] = 2.0
    missed[3] = 0.5
    np.testing.assert_allclose(model.terminal_violations(missed), (1.0, 0.0, 0.0, 0.25, 0.0, 0.0))
    assert not model.terminal_is_feasible(missed)


def test_objective_terms_are_dimensionless_and_separate_from_feasibility() -> None:
    model = problem()
    terminal = np.asarray((0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 950.0))
    controls = np.tile((0.5, 0.0), (model.intervals, 1))
    terms = model.objective(terminal, controls)

    assert terms.fuel == pytest.approx(0.2)
    assert terms.touchdown == 0.0
    assert terms.smoothness == 0.0
    assert terms.weighted_total == pytest.approx(model.fuel_weight * 0.2)

    controls[1, 0] = 0.8
    assert model.objective(terminal, controls).smoothness > 0.0
    terminal[0] = 0.5
    assert model.objective(terminal, controls).touchdown > 0.0


def test_free_horizon_rk4_defects_match_independent_simulator_rollout() -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 4
    model = LandingOptimalControlProblem.from_config(config, INITIAL)
    duration = 5.0
    h = model.mesh_step_s(duration)
    controls = np.tile((0.5, 0.0), (model.intervals, 1))
    states = [INITIAL.copy()]
    for k in range(model.intervals):
        action = controls[k]
        _, simulated = simulate_planar(
            State.from_array(states[-1]),
            action,
            model.parameters,
            duration_s=h,
            dt_s=h,
        )
        states.append(simulated[-1])

    defects = model.rk4_defects(np.asarray(states), controls, duration)
    assert defects.shape == (4, 7)
    np.testing.assert_allclose(defects, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="duration_s"):
        model.mesh_step_s(30.0)


def test_problem_rejects_exhausted_initial_state_and_bad_arrays() -> None:
    depleted = INITIAL.copy()
    depleted[6] = 751.0
    with pytest.raises(ValueError, match="reserve"):
        LandingOptimalControlProblem.from_config(CONFIG, depleted)

    model = problem()
    with pytest.raises(ValueError, match="action"):
        model.dynamics(INITIAL, (np.nan, 0.0))
    with pytest.raises(ValueError, match="controls"):
        model.objective(INITIAL, np.zeros((model.intervals - 1, 2)))
