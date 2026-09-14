"""Suicide-burn estimate, timing classification and vertical rollout checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import load_config
from powered_landing_guidance.controllers import (
    HorizontalAttitudeController,
    SuicideBurnController,
    VerticalVelocityPIDController,
    VerticalVelocityProfile,
    classify_ignition_timing,
    estimate_suicide_burn,
    estimate_variable_mass_suicide_burn,
)
from powered_landing_guidance.dynamics import PlanarDynamicsParameters, simulate_planar
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.model import State
from scripts.evaluate_horizontal_control import (
    evaluate_horizontal_control,
    sample_horizontal_initial_states,
)
from scripts.sweep_velocity_gains import evaluate_gains, sample_vertical_initial_states

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
PARAMETERS = PlanarDynamicsParameters.from_config(CONFIG)


def test_horizontal_controller_commands_tilt_toward_pad() -> None:
    controller = HorizontalAttitudeController.from_config(CONFIG)
    state = State(10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    hover_throttle = state.mass * PARAMETERS.gravity_m_s2 / PARAMETERS.max_thrust_n

    action = controller.command(state, base_throttle=hover_throttle)

    assert controller.horizontal_acceleration_m_s2 < 0.0
    assert controller.target_attitude_rad < 0.0
    assert action[1] > 0.0


def test_horizontal_controller_limits_tilt_and_gimbal() -> None:
    controller = HorizontalAttitudeController.from_config(CONFIG)
    controller.position_kp_s2 = 100.0
    controller.max_horizontal_acceleration_m_s2 = 100.0
    state = State(100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    hover_throttle = state.mass * PARAMETERS.gravity_m_s2 / PARAMETERS.max_thrust_n

    action = controller.command(state, base_throttle=hover_throttle)

    assert controller.target_attitude_rad == pytest.approx(-controller.max_tilt_rad)
    assert controller.tilt_limited
    assert abs(action[1]) <= PARAMETERS.gimbal_limit_rad


def test_horizontal_controller_compensates_vertical_thrust_loss() -> None:
    controller = HorizontalAttitudeController.from_config(CONFIG)
    state = State(8.0, 100.0, 0.0, 0.0, np.deg2rad(7.0), 0.0, 1000.0)
    base_throttle = state.mass * PARAMETERS.gravity_m_s2 / PARAMETERS.max_thrust_n

    action = controller.command(state, base_throttle=base_throttle)
    vertical_throttle = action[0] * np.cos(state.theta + action[1])

    assert vertical_throttle == pytest.approx(base_throttle, abs=1e-8)
    assert action[0] >= base_throttle


def test_horizontal_controller_handles_zero_throttle_without_invalid_gimbal() -> None:
    controller = HorizontalAttitudeController.from_config(CONFIG)
    state = State(10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)

    action = controller.command(state, base_throttle=0.0)

    np.testing.assert_array_equal(action, [0.0, 0.0])
    assert controller.gimbal_limited


def test_horizontal_evaluation_is_reproducible_and_converges_toward_pad() -> None:
    first = sample_horizontal_initial_states(CONFIG, episodes=20, seed=20260914)
    second = sample_horizontal_initial_states(CONFIG, episodes=20, seed=20260914)

    np.testing.assert_array_equal(first, second)
    result = evaluate_horizontal_control(CONFIG, first)

    assert result.completed_episodes == result.episodes
    assert result.converged_episodes == result.episodes
    assert result.mean_final_abs_error_m < result.mean_initial_abs_error_m
    assert (
        result.max_target_tilt_deg
        <= CONFIG["horizontal_attitude_controller"]["outer_loop"]["max_tilt_deg"]
    )
    assert result.max_gimbal_deg <= CONFIG["vehicle"]["gimbal_limit_deg"]


def test_vertical_velocity_profile_slows_descent_near_ground() -> None:
    profile = VerticalVelocityProfile(
        touchdown_speed_m_s=1.0,
        max_descent_speed_m_s=30.0,
        deceleration_m_s2=3.0,
    )

    assert profile.target_velocity_m_s(0.0) == pytest.approx(-1.0)
    assert profile.target_velocity_m_s(50.0) < profile.target_velocity_m_s(10.0)
    assert profile.target_velocity_m_s(1000.0) == pytest.approx(-30.0)
    assert profile.target_acceleration_m_s2(10.0) == pytest.approx(3.0)
    assert profile.target_acceleration_m_s2(1000.0) == pytest.approx(0.0)


def test_velocity_pid_freezes_integrator_when_throttle_saturates() -> None:
    controller = VerticalVelocityPIDController.from_config(CONFIG, kp=1.0, ki=1.0, kd=0.0)
    too_fast = State(0.0, 1.0, 0.0, -50.0, 0.0, 0.0, 1000.0)
    too_slow = State(0.0, 100.0, 0.0, 10.0, 0.0, 0.0, 1000.0)

    np.testing.assert_array_equal(controller.command(too_fast), [1.0, 0.0])
    assert controller.saturated
    assert controller.integral_error_m == 0.0

    controller.reset()
    np.testing.assert_array_equal(controller.command(too_slow), [0.0, 0.0])
    assert controller.saturated
    assert controller.integral_error_m == 0.0


def test_velocity_pid_reset_clears_accumulated_state() -> None:
    controller = VerticalVelocityPIDController.from_config(CONFIG)
    state = State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)

    action = controller.command(state)
    assert 0.0 <= action[0] <= 1.0
    assert controller.previous_error_m_s is not None
    assert controller.target_vertical_speed_m_s is not None

    controller.reset()
    assert controller.integral_error_m == 0.0
    assert controller.previous_error_m_s is None
    assert controller.target_vertical_speed_m_s is None


def test_constant_mass_estimate_matches_closed_form_solution() -> None:
    state = State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)
    estimate = estimate_suicide_burn(
        state,
        PARAMETERS,
        target_touchdown_speed_m_s=2.0,
        reaction_time_s=0.02,
    )
    net_deceleration = PARAMETERS.max_thrust_n / state.mass - PARAMETERS.gravity_m_s2
    stopping_distance = (20.0**2 - 2.0**2) / (2.0 * net_deceleration)
    reaction_distance = 20.0 * 0.02 + 0.5 * PARAMETERS.gravity_m_s2 * 0.02**2
    burn_time = (20.0 - 2.0) / net_deceleration
    mass_flow_rate = PARAMETERS.max_thrust_n / (
        PARAMETERS.specific_impulse_s * PARAMETERS.standard_gravity_m_s2
    )

    assert estimate.net_deceleration_m_s2 == pytest.approx(net_deceleration)
    assert estimate.stopping_distance_m == pytest.approx(stopping_distance)
    assert estimate.reaction_distance_m == pytest.approx(reaction_distance)
    assert estimate.ignition_height_m == pytest.approx(stopping_distance + reaction_distance)
    assert estimate.burn_time_s == pytest.approx(burn_time)
    assert estimate.propellant_required_kg == pytest.approx(mass_flow_rate * burn_time)
    assert estimate.feasible


def test_faster_or_heavier_descent_requires_higher_ignition() -> None:
    slow = estimate_suicide_burn(
        State(0.0, 100.0, 0.0, -10.0, 0.0, 0.0, 900.0),
        PARAMETERS,
    )
    fast = estimate_suicide_burn(
        State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 900.0),
        PARAMETERS,
    )
    light = estimate_suicide_burn(
        State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 900.0),
        PARAMETERS,
    )
    heavy = estimate_suicide_burn(
        State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0),
        PARAMETERS,
    )

    assert fast.ignition_height_m > slow.ignition_height_m
    assert heavy.ignition_height_m > light.ignition_height_m


def test_variable_mass_estimate_needs_less_distance_than_constant_mass() -> None:
    state = State(0.0, 100.0, 0.0, -25.0, 0.0, 0.0, 1000.0)

    constant = estimate_suicide_burn(state, PARAMETERS)
    variable = estimate_variable_mass_suicide_burn(state, PARAMETERS)

    assert constant.mass_model == "constant"
    assert variable.mass_model == "variable"
    assert variable.ignition_height_m < constant.ignition_height_m
    assert variable.propellant_required_kg < constant.propellant_required_kg


def test_variable_mass_estimate_matches_simulator_rollout() -> None:
    state = State(0.0, 100.0, 0.0, -25.0, 0.0, 0.0, 1000.0)
    estimate = estimate_variable_mass_suicide_burn(state, PARAMETERS)

    _, states = simulate_planar(
        state,
        [1.0, 0.0],
        PARAMETERS,
        duration_s=estimate.burn_time_s,
        dt_s=0.01,
    )

    assert states[-1, 3] == pytest.approx(-2.0, abs=1e-9)
    assert state.z - states[-1, 1] == pytest.approx(estimate.stopping_distance_m, abs=1e-9)
    assert state.mass - states[-1, 6] == pytest.approx(estimate.propellant_required_kg, abs=1e-9)


@pytest.mark.parametrize(
    ("actual", "required", "expected"),
    [
        (25.0, 20.0, "early_burn"),
        (20.2, 20.0, "on_time"),
        (15.0, 20.0, "late_burn"),
    ],
)
def test_ignition_timing_classification(actual, required, expected) -> None:
    assert classify_ignition_timing(actual, required, tolerance_m=0.5) == expected


def test_infeasible_vehicle_uses_immediate_best_effort_thrust() -> None:
    weak_parameters = replace(PARAMETERS, max_thrust_n=9000.0)
    controller = SuicideBurnController(
        parameters=weak_parameters,
        ground_z_m=0.0,
        target_touchdown_speed_m_s=2.0,
        control_interval_s=0.02,
        reaction_time_s=0.02,
    )
    state = State(0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)

    estimate = controller.estimate(state)
    action = controller.command(state)

    assert not estimate.feasible
    assert np.isinf(estimate.ignition_height_m)
    np.testing.assert_array_equal(action, [1.0, 0.0])
    assert controller.ignition_timing == "infeasible"


def test_controller_keeps_engine_on_after_ignition() -> None:
    controller = SuicideBurnController.from_config(CONFIG)
    falling = State(0.0, 10.0, 0.0, -20.0, 0.0, 0.0, 1000.0)
    rising = State(0.0, 50.0, 0.0, 5.0, 0.0, 0.0, 990.0)

    np.testing.assert_array_equal(controller.command(falling), [1.0, 0.0])
    np.testing.assert_array_equal(controller.command(rising), [1.0, 0.0])
    assert controller.ignited

    controller.reset()
    assert not controller.ignited
    assert controller.ignition_timing is None


def test_vertical_evaluation_states_are_reproducible_and_vertical_only() -> None:
    first = sample_vertical_initial_states(CONFIG, episodes=12, seed=20260910)
    second = sample_vertical_initial_states(CONFIG, episodes=12, seed=20260910)
    ranges = CONFIG["vertical_velocity_controller"]["evaluation"]

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, [0, 2, 4, 5]], 0.0)
    assert np.all(ranges["altitude_m"][0] <= first[:, 1])
    assert np.all(first[:, 1] <= ranges["altitude_m"][1])
    assert np.all(ranges["vertical_speed_m_s"][0] <= first[:, 3])
    assert np.all(first[:, 3] <= ranges["vertical_speed_m_s"][1])
    assert np.all(ranges["mass_kg"][0] <= first[:, 6])
    assert np.all(first[:, 6] <= ranges["mass_kg"][1])


def test_selected_velocity_pid_gains_exceed_configured_success_gate() -> None:
    states = sample_vertical_initial_states(CONFIG, episodes=20, seed=20260910)
    gains = CONFIG["vertical_velocity_controller"]["gains"]

    result = evaluate_gains(CONFIG, states, **gains)

    assert result.success_rate >= 0.8
    assert result.successes == result.episodes
    assert result.p95_touchdown_speed_m_s <= CONFIG["landing_success"]["max_abs_vz_m_s"]


@pytest.mark.parametrize(
    "initial_state",
    [
        [0.0, 80.0, 0.0, -15.0, 0.0, 0.0, 950.0],
        [0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0],
        [0.0, 120.0, 0.0, -25.0, 0.0, 0.0, 1000.0],
    ],
)
def test_vertical_suicide_burn_meets_touchdown_speed_limit(initial_state) -> None:
    env = RocketLandingEnv(CONFIG)
    controller = SuicideBurnController.from_config(CONFIG)
    try:
        state, _ = env.reset(options={"initial_state": initial_state})
        throttles = []
        while True:
            action = controller.command(state)
            throttles.append(float(action[0]))
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    assert info["outcome"] == "success"
    assert abs(state[3]) <= CONFIG["landing_success"]["max_abs_vz_m_s"]
    assert controller.ignited
    assert controller.ignition_timing == "on_time"
    assert any(0.0 < throttle < 1.0 for throttle in throttles)
