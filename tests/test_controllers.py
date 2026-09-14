"""Suicide-burn estimate, timing classification and vertical rollout checks."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import load_config
from powered_landing_guidance.controllers import (
    HorizontalAttitudeController,
    IntegratedLandingController,
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
from scripts.evaluate_integrated_control import (
    evaluate_integrated_control,
    sample_integrated_initial_states,
)
from scripts.evaluate_pid_disturbances import (
    DISTURBANCE_ORDER,
    FirstOrderThrottleLag,
    add_sensor_noise,
    evaluate_disturbances,
    save_disturbance_outputs,
)
from scripts.sweep_velocity_gains import evaluate_gains, sample_vertical_initial_states
from scripts.tune_integrated_controller import (
    ScaleCandidate,
    apply_candidate,
    grid_candidates,
    random_candidates,
    save_tuning_outputs,
    tune_controller,
)

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


def test_integrated_controller_holds_action_between_control_updates() -> None:
    controller = IntegratedLandingController.from_config(CONFIG)
    state = State(8.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0)

    first = controller.command(state, time_s=0.0)
    held = controller.command(state, time_s=CONFIG["simulation"]["dt_s"])

    assert controller.update_count == 1
    assert not controller.updated_this_step
    np.testing.assert_array_equal(held, first)

    controller.command(state, time_s=controller.control_interval_s)
    assert controller.update_count == 2
    assert controller.updated_this_step


def test_integrated_controller_switches_phase_by_altitude() -> None:
    controller = IntegratedLandingController.from_config(CONFIG)
    high = State(0.0, 50.0, 0.0, -10.0, 0.0, 0.0, 1000.0)
    low = State(0.0, 20.0, 0.0, -5.0, 0.0, 0.0, 990.0)

    controller.command(high, time_s=0.0)
    assert controller.phase == "approach"

    controller.command(low, time_s=controller.control_interval_s)
    assert controller.phase == "terminal"


def test_integrated_controller_limits_command_slew_rate() -> None:
    controller = IntegratedLandingController.from_config(CONFIG)
    slow = State(10.0, 100.0, 0.0, 10.0, 0.0, 0.0, 1000.0)
    fast = State(-10.0, 100.0, 0.0, -50.0, np.deg2rad(-20.0), 0.0, 1000.0)
    first = controller.command(slow, time_s=0.0)
    second = controller.command(fast, time_s=controller.control_interval_s)

    change = np.abs(second - first)
    assert change[0] <= (
        controller.throttle_slew_rate_per_s * controller.control_interval_s + 1e-12
    )
    assert change[1] <= (controller.gimbal_slew_rate_rad_s * controller.control_interval_s + 1e-12)
    assert controller.throttle_slew_limited
    assert controller.gimbal_slew_limited


def test_integrated_evaluation_is_reproducible_and_lands_safely() -> None:
    first = sample_integrated_initial_states(CONFIG, episodes=12, seed=20260914)
    second = sample_integrated_initial_states(CONFIG, episodes=12, seed=20260914)

    np.testing.assert_array_equal(first, second)
    result = evaluate_integrated_control(CONFIG, first)

    assert result.successes == result.episodes
    assert result.max_abs_touchdown_x_m <= CONFIG["landing_success"]["max_abs_x_m"]
    assert result.max_abs_touchdown_vx_m_s <= CONFIG["landing_success"]["max_abs_vx_m_s"]
    assert result.max_abs_touchdown_vz_m_s <= CONFIG["landing_success"]["max_abs_vz_m_s"]
    assert result.max_abs_touchdown_theta_deg <= CONFIG["landing_success"]["max_abs_theta_deg"]
    assert result.max_abs_touchdown_omega_deg_s <= CONFIG["landing_success"]["max_abs_omega_deg_s"]
    assert (
        result.max_throttle_slew_per_s
        <= CONFIG["integrated_landing_controller"]["slew_rates"]["throttle_per_s"] + 1e-10
    )
    assert (
        result.max_gimbal_slew_deg_s
        <= CONFIG["integrated_landing_controller"]["slew_rates"]["gimbal_deg_s"] + 1e-10
    )


def test_tuning_candidate_generation_is_reproducible() -> None:
    grid = grid_candidates(CONFIG)
    first_random = random_candidates(CONFIG, seed=20260912)
    second_random = random_candidates(CONFIG, seed=20260912)

    assert len(grid) == len(set(grid)) == 13
    assert ScaleCandidate(1.0, 1.0, 1.0) in grid
    assert first_random == second_random
    assert first_random[0] == ScaleCandidate(1.0, 1.0, 1.0)


def test_applying_tuning_candidate_does_not_mutate_source_config() -> None:
    before = CONFIG["integrated_landing_controller"]["phases"]["approach"]["horizontal_outer_loop"][
        "position_kp_s2"
    ]
    tuned = apply_candidate(CONFIG, ScaleCandidate(1.1, 0.95, 1.1))
    after = tuned["integrated_landing_controller"]["phases"]["approach"]["horizontal_outer_loop"][
        "position_kp_s2"
    ]

    assert (
        CONFIG["integrated_landing_controller"]["phases"]["approach"]["horizontal_outer_loop"][
            "position_kp_s2"
        ]
        == before
    )
    assert after == pytest.approx(before * 1.1)


def test_tuning_uses_separate_batches_and_saves_loadable_best_config(tmp_path) -> None:
    config = deepcopy(CONFIG)
    grid = config["baseline_tuning"]["search"]["grid"]
    grid["position_gain_scale"] = [1.0]
    grid["velocity_gain_scale"] = [1.0]
    grid["descent_profile_scale"] = [1.0]

    result, best_config = tune_controller(
        config,
        method="grid",
        train_episodes=2,
        validation_episodes=3,
    )
    report_path = tmp_path / "tuning.json"
    config_path = tmp_path / "best.yaml"
    save_tuning_outputs(
        result,
        best_config,
        report_path=report_path,
        config_path=config_path,
    )

    assert result.train_seed != result.validation_seed
    assert result.candidates_evaluated == len(result.train_results) == 1
    assert result.best_train_result.evaluation.episodes == 2
    assert result.validation_result.episodes == 3
    assert load_config(config_path)["baseline_tuning"]["selected"]["candidate"] == {
        "position_gain_scale": 1.0,
        "velocity_gain_scale": 1.0,
        "descent_profile_scale": 1.0,
    }
    assert json.loads(report_path.read_text(encoding="utf-8"))["validation_episodes"] == 3


def test_first_order_throttle_lag_keeps_gimbal_immediate() -> None:
    actuator = FirstOrderThrottleLag(time_constant_s=0.2)
    applied = actuator.apply([1.0, 0.1], dt_s=0.2)

    assert applied[0] == pytest.approx(1.0 - np.exp(-1.0))
    assert applied[1] == 0.1
    np.testing.assert_array_equal(
        FirstOrderThrottleLag(time_constant_s=0.0).apply([0.7, -0.2], dt_s=0.02),
        [0.7, -0.2],
    )


def test_sensor_noise_is_seeded_and_does_not_mutate_true_state() -> None:
    state = np.asarray([1.0, 2.0, 3.0, -4.0, 0.1, -0.2, 900.0])
    deviations = np.ones(7)
    before = state.copy()
    first = add_sensor_noise(
        state,
        deviations,
        1.0,
        np.random.default_rng(7),
        ground_z_m=0.0,
        dry_mass_kg=750.0,
    )
    second = add_sensor_noise(
        state,
        deviations,
        1.0,
        np.random.default_rng(7),
        ground_z_m=0.0,
        dry_mass_kg=750.0,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(state, before)
    np.testing.assert_array_equal(
        add_sensor_noise(
            state,
            deviations,
            0.0,
            np.random.default_rng(8),
            ground_z_m=0.0,
            dry_mass_kg=750.0,
        ),
        state,
    )


def test_isolated_disturbance_sweep_saves_complete_curves(tmp_path) -> None:
    config = deepcopy(CONFIG)
    settings = config["disturbance_evaluation"]
    settings["constant_wind_m_s"] = [0.0, 10.0]
    settings["gust"]["amplitude_m_s"] = [0.0, 20.0]
    settings["sensor_noise"]["scales"] = [0.0, 1.0]
    settings["thrust_scale"] = [1.0, 0.9]
    settings["engine_lag_s"] = [0.0, 0.1]
    result = evaluate_disturbances(config, episodes_per_level=1)
    repeated = evaluate_disturbances(config, episodes_per_level=1)
    report = tmp_path / "disturbances.json"
    plot = tmp_path / "disturbances.png"
    save_disturbance_outputs(result, report_path=report, plot_path=plot)

    assert tuple(curve.kind for curve in result.curves) == DISTURBANCE_ORDER
    assert repeated == result
    assert all(len(curve.levels) == 2 for curve in result.curves)
    baselines = [curve.levels[0] for curve in result.curves]
    assert all(level.success_rate == baselines[0].success_rate for level in baselines)
    assert json.loads(report.read_text(encoding="utf-8"))["episodes_per_level"] == 1
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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
