"""Analytic checks for translation, rotation, actuators and fuel cutoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    aerodynamic_drag_force_n,
    angular_acceleration_rad_s2,
    applied_thrust_n,
    clip_control,
    integrate_fixed_step,
    propellant_mass_flow_rate_kg_s,
    simulate_planar,
    state_derivative,
    thrust_torque_nm,
    thrust_vector,
    translational_acceleration,
)

PARAMETERS = PlanarDynamicsParameters.from_config(
    load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
)
STATE_WITH_FUEL = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
FULL_THRUST = Control(1.0, 0.0)


def test_free_fall_matches_analytic_solution_with_rk4() -> None:
    initial = State(x=5.0, z=100.0, vx=2.0, vz=-3.0, theta=0.0, omega=0.0, mass=900.0)
    control = Control(throttle=0.0, gimbal_angle=0.0)

    times, states = simulate_planar(
        initial,
        control,
        PARAMETERS,
        duration_s=2.0,
        dt_s=0.05,
        method="rk4",
    )

    expected_x = initial.x + initial.vx * times
    expected_z = initial.z + initial.vz * times - 0.5 * PARAMETERS.gravity_m_s2 * times**2
    expected_vz = initial.vz - PARAMETERS.gravity_m_s2 * times
    np.testing.assert_allclose(states[:, 0], expected_x, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(states[:, 1], expected_z, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(states[:, 2], initial.vx, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(states[:, 3], expected_vz, rtol=0.0, atol=1e-11)
    expected_constants = np.broadcast_to(initial.as_array()[4:], states[:, 4:].shape)
    np.testing.assert_allclose(states[:, 4:], expected_constants, rtol=0.0, atol=1e-12)


def test_vertical_thrust_matches_variable_mass_analytic_solution() -> None:
    initial = State(x=0.0, z=10.0, vx=0.0, vz=1.0, theta=0.0, omega=0.0, mass=1000.0)
    control = Control(throttle=1.0, gimbal_angle=0.0)
    duration_s = 3.0

    times, states = simulate_planar(
        initial,
        control,
        PARAMETERS,
        duration_s=duration_s,
        dt_s=0.02,
        method="rk4",
    )

    exhaust_velocity = PARAMETERS.specific_impulse_s * PARAMETERS.standard_gravity_m_s2
    mass_flow_rate = PARAMETERS.max_thrust_n / exhaust_velocity
    expected_mass = initial.mass - mass_flow_rate * times
    log_mass_ratio = np.log(initial.mass / expected_mass)
    integrated_log_ratio = (
        initial.mass - expected_mass - expected_mass * log_mass_ratio
    ) / mass_flow_rate
    expected_z = (
        initial.z
        + initial.vz * times
        + exhaust_velocity * integrated_log_ratio
        - 0.5 * PARAMETERS.gravity_m_s2 * times**2
    )
    expected_vz = initial.vz + exhaust_velocity * log_mass_ratio - PARAMETERS.gravity_m_s2 * times
    np.testing.assert_allclose(states[:, 1], expected_z, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(states[:, 3], expected_vz, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(states[:, 0], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(states[:, 2], 0.0, rtol=0.0, atol=1e-12)
    expected_angles = np.broadcast_to(initial.as_array()[4:6], states[:, 4:6].shape)
    np.testing.assert_allclose(states[:, 4:6], expected_angles, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(states[:, 6], expected_mass, rtol=0.0, atol=1e-11)


def test_positive_gimbal_produces_positive_x_thrust() -> None:
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, mass=900.0)
    control = Control(throttle=0.5, gimbal_angle=np.deg2rad(10.0))

    force = thrust_vector(state, control, PARAMETERS)

    assert force[0] > 0.0
    assert force[1] > 0.0


def test_acceleration_uses_current_mass() -> None:
    control = Control(throttle=1.0, gimbal_angle=0.0)
    heavy = State(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    light = State(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 800.0)

    heavy_acceleration = translational_acceleration(heavy, control, PARAMETERS)
    light_acceleration = translational_acceleration(light, control, PARAMETERS)

    assert light_acceleration[1] > heavy_acceleration[1]


def test_quadratic_drag_uses_velocity_relative_to_wind() -> None:
    stationary = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    wind = np.asarray((10.0, 0.0))

    force = aerodynamic_drag_force_n(stationary, wind, PARAMETERS)
    expected = (
        0.5
        * PARAMETERS.air_density_kg_m3
        * PARAMETERS.drag_coefficient
        * PARAMETERS.reference_area_m2
        * 10.0**2
    )

    np.testing.assert_allclose(force, [expected, 0.0], rtol=0.0, atol=1e-12)
    matching_airflow = State(0.0, 100.0, 10.0, 0.0, 0.0, 0.0, 1000.0)
    np.testing.assert_array_equal(aerodynamic_drag_force_n(matching_airflow, wind, PARAMETERS), 0.0)


def test_wind_input_changes_translation_without_changing_default_dynamics() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    calm = translational_acceleration(state, [0.0, 0.0], PARAMETERS)
    windy = translational_acceleration(
        state,
        [0.0, 0.0],
        PARAMETERS,
        wind_velocity_m_s=[20.0, 0.0],
    )

    np.testing.assert_allclose(calm, [0.0, -PARAMETERS.gravity_m_s2], rtol=0.0, atol=1e-12)
    assert windy[0] > 0.0
    assert windy[1] == pytest.approx(calm[1])


def test_mass_below_dry_mass_is_rejected() -> None:
    state = State(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, PARAMETERS.dry_mass_kg - 1.0)
    control = Control(throttle=0.0, gimbal_angle=0.0)

    with pytest.raises(ValueError, match="dry mass"):
        translational_acceleration(state, control, PARAMETERS)


def test_smaller_euler_step_reduces_free_fall_position_error() -> None:
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 900.0)
    control = Control(0.0, 0.0)
    duration_s = 1.0
    expected_z = initial.z - 0.5 * PARAMETERS.gravity_m_s2 * duration_s**2

    _, coarse_states = simulate_planar(
        initial, control, PARAMETERS, duration_s, 0.1, method="euler"
    )
    _, fine_states = simulate_planar(initial, control, PARAMETERS, duration_s, 0.01, method="euler")

    coarse_error = abs(coarse_states[-1, 1] - expected_z)
    fine_error = abs(fine_states[-1, 1] - expected_z)
    assert fine_error < coarse_error / 5.0


def test_integrator_shortens_final_step_to_end_exactly() -> None:
    def unit_rate(_time_s: float, state: np.ndarray) -> np.ndarray:
        return np.ones_like(state)

    times, states = integrate_fixed_step(unit_rate, [0.0], (0.0, 1.0), 0.3, method="rk4")

    assert times[-1] == 1.0
    np.testing.assert_allclose(states[-1], [1.0], rtol=0.0, atol=1e-12)


def test_state_derivative_keeps_angular_rate_constant_without_gimbal() -> None:
    state = State(0.0, 20.0, 1.0, -2.0, 0.1, 0.2, 900.0)
    control = Control(0.2, 0.0)

    derivative = state_derivative(0.0, state.as_array(), control, PARAMETERS)

    assert derivative[4] == state.omega
    assert derivative[5] == 0.0


def test_configured_gimbal_limit_is_converted_to_radians() -> None:
    assert PARAMETERS.gimbal_limit_rad == pytest.approx(np.deg2rad(15.0))


@pytest.mark.parametrize(
    ("raw_command", "expected_throttle", "expected_gimbal_deg"),
    [
        ([1.5, np.deg2rad(30.0)], 1.0, 15.0),
        ([-0.5, np.deg2rad(-30.0)], 0.0, -15.0),
        ([0.4, np.deg2rad(5.0)], 0.4, 5.0),
    ],
)
def test_control_command_is_clipped_to_actuator_limits(
    raw_command: list[float],
    expected_throttle: float,
    expected_gimbal_deg: float,
) -> None:
    applied = clip_control(raw_command, PARAMETERS)

    assert applied.throttle == pytest.approx(expected_throttle)
    assert applied.gimbal_angle == pytest.approx(np.deg2rad(expected_gimbal_deg))


@pytest.mark.parametrize("raw_command", [[0.5], [0.5, 0.0, 1.0], [np.nan, 0.0]])
def test_invalid_raw_control_command_is_rejected(raw_command: list[float]) -> None:
    with pytest.raises(ValueError, match="control command"):
        clip_control(raw_command, PARAMETERS)


@pytest.mark.parametrize(
    ("gimbal_deg", "expected_torque_sign"),
    [(5.0, -1.0), (-5.0, 1.0)],
)
def test_gimbal_sign_produces_documented_torque_direction(
    gimbal_deg: float,
    expected_torque_sign: float,
) -> None:
    command = Control(throttle=0.75, gimbal_angle=np.deg2rad(gimbal_deg))

    torque = thrust_torque_nm(STATE_WITH_FUEL, command, PARAMETERS)
    expected_magnitude = (
        PARAMETERS.engine_lever_arm_m
        * command.throttle
        * PARAMETERS.max_thrust_n
        * np.sin(abs(command.gimbal_angle))
    )

    assert np.sign(torque) == expected_torque_sign
    assert abs(torque) == pytest.approx(expected_magnitude)


@pytest.mark.parametrize(
    "command",
    [Control(throttle=0.0, gimbal_angle=np.deg2rad(5.0)), Control(throttle=1.0, gimbal_angle=0.0)],
)
def test_zero_throttle_or_zero_gimbal_produces_no_torque(command: Control) -> None:
    assert thrust_torque_nm(STATE_WITH_FUEL, command, PARAMETERS) == pytest.approx(0.0, abs=1e-12)


def test_angular_acceleration_uses_moment_of_inertia() -> None:
    command = Control(throttle=0.8, gimbal_angle=np.deg2rad(-4.0))

    angular_acceleration = angular_acceleration_rad_s2(STATE_WITH_FUEL, command, PARAMETERS)

    assert angular_acceleration == pytest.approx(
        thrust_torque_nm(STATE_WITH_FUEL, command, PARAMETERS) / PARAMETERS.moment_of_inertia_kg_m2
    )


@pytest.mark.parametrize("gimbal_deg", [5.0, -5.0])
def test_rotation_matches_constant_angular_acceleration_solution(gimbal_deg: float) -> None:
    initial = State(0.0, 100.0, 0.0, 0.0, 0.1, -0.05, 1000.0)
    command = Control(throttle=0.7, gimbal_angle=np.deg2rad(gimbal_deg))
    duration_s = 0.5
    angular_acceleration = angular_acceleration_rad_s2(initial, command, PARAMETERS)

    times, states = simulate_planar(
        initial,
        command,
        PARAMETERS,
        duration_s=duration_s,
        dt_s=0.01,
        method="rk4",
    )

    expected_theta = initial.theta + initial.omega * times + 0.5 * angular_acceleration * times**2
    expected_omega = initial.omega + angular_acceleration * times
    np.testing.assert_allclose(states[:, 4], expected_theta, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(states[:, 5], expected_omega, rtol=0.0, atol=1e-12)


def test_out_of_range_command_is_clipped_inside_state_derivative() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    raw_command = [2.0, np.deg2rad(45.0)]
    clipped_command = Control(1.0, PARAMETERS.gimbal_limit_rad)

    raw_derivative = state_derivative(0.0, state.as_array(), raw_command, PARAMETERS)
    clipped_derivative = state_derivative(0.0, state.as_array(), clipped_command, PARAMETERS)

    np.testing.assert_allclose(raw_derivative, clipped_derivative, rtol=0.0, atol=1e-12)


def test_mass_flow_matches_specific_impulse_equation() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    expected = PARAMETERS.max_thrust_n / (
        PARAMETERS.specific_impulse_s * PARAMETERS.standard_gravity_m_s2
    )

    assert propellant_mass_flow_rate_kg_s(state, FULL_THRUST, PARAMETERS) == pytest.approx(expected)


def test_mass_flow_is_proportional_to_throttle() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    full_rate = propellant_mass_flow_rate_kg_s(state, FULL_THRUST, PARAMETERS)
    half_rate = propellant_mass_flow_rate_kg_s(state, Control(0.5, 0.0), PARAMETERS)

    assert half_rate == pytest.approx(0.5 * full_rate)


def test_state_derivative_contains_negative_mass_flow() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)

    derivative = state_derivative(0.0, state.as_array(), FULL_THRUST, PARAMETERS)

    assert derivative[6] == pytest.approx(
        -propellant_mass_flow_rate_kg_s(state, FULL_THRUST, PARAMETERS)
    )


def test_mass_decreases_linearly_before_burnout() -> None:
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    duration_s = 5.0
    mass_flow_rate = propellant_mass_flow_rate_kg_s(initial, FULL_THRUST, PARAMETERS)

    times, states = simulate_planar(initial, FULL_THRUST, PARAMETERS, duration_s, 0.02)
    expected_mass = initial.mass - mass_flow_rate * times

    np.testing.assert_allclose(states[:, 6], expected_mass, rtol=0.0, atol=2e-11)


def test_zero_throttle_does_not_consume_propellant() -> None:
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 800.0)

    _, states = simulate_planar(initial, Control(0.0, 0.2), PARAMETERS, 10.0, 0.1)

    np.testing.assert_allclose(states[:, 6], initial.mass, rtol=0.0, atol=1e-12)


def test_dry_mass_cuts_off_thrust_torque_and_mass_flow() -> None:
    dry_state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, PARAMETERS.dry_mass_kg)
    command = Control(1.0, np.deg2rad(10.0))

    assert applied_thrust_n(dry_state, command, PARAMETERS) == 0.0
    np.testing.assert_allclose(
        thrust_vector(dry_state, command, PARAMETERS), [0.0, 0.0], rtol=0.0, atol=0.0
    )
    assert thrust_torque_nm(dry_state, command, PARAMETERS) == 0.0
    assert propellant_mass_flow_rate_kg_s(dry_state, command, PARAMETERS) == 0.0

    derivative = state_derivative(0.0, dry_state.as_array(), command, PARAMETERS)
    np.testing.assert_allclose(
        derivative,
        [0.0, 0.0, 0.0, -PARAMETERS.gravity_m_s2, 0.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_below_dry_mass_is_rejected() -> None:
    invalid = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, PARAMETERS.dry_mass_kg - 0.1)

    with pytest.raises(ValueError, match="below dry mass"):
        state_derivative(0.0, invalid.as_array(), FULL_THRUST, PARAMETERS)
    with pytest.raises(ValueError, match="below dry mass"):
        simulate_planar(invalid, FULL_THRUST, PARAMETERS, 1.0, 0.1)


def test_burnout_inside_coarse_step_matches_piecewise_analytic_solution() -> None:
    initial = State(0.0, 100.0, 0.0, -2.0, 0.0, 0.0, PARAMETERS.dry_mass_kg + 1.0)
    duration_s = 1.0
    mass_flow_rate = propellant_mass_flow_rate_kg_s(initial, FULL_THRUST, PARAMETERS)
    burnout_time_s = (initial.mass - PARAMETERS.dry_mass_kg) / mass_flow_rate
    exhaust_velocity = PARAMETERS.specific_impulse_s * PARAMETERS.standard_gravity_m_s2
    log_mass_ratio = np.log(initial.mass / PARAMETERS.dry_mass_kg)
    integrated_log_ratio = (
        initial.mass - PARAMETERS.dry_mass_kg - PARAMETERS.dry_mass_kg * log_mass_ratio
    ) / mass_flow_rate
    burnout_velocity = (
        initial.vz + exhaust_velocity * log_mass_ratio - PARAMETERS.gravity_m_s2 * burnout_time_s
    )
    burnout_height = (
        initial.z
        + initial.vz * burnout_time_s
        + exhaust_velocity * integrated_log_ratio
        - 0.5 * PARAMETERS.gravity_m_s2 * burnout_time_s**2
    )
    coast_time_s = duration_s - burnout_time_s
    expected_velocity = burnout_velocity - PARAMETERS.gravity_m_s2 * coast_time_s
    expected_height = (
        burnout_height
        + burnout_velocity * coast_time_s
        - 0.5 * PARAMETERS.gravity_m_s2 * coast_time_s**2
    )

    _, states = simulate_planar(
        initial,
        FULL_THRUST,
        PARAMETERS,
        duration_s=duration_s,
        dt_s=1.0,
        method="rk4",
    )

    assert states[-1, 6] == PARAMETERS.dry_mass_kg
    assert np.min(states[:, 6]) >= PARAMETERS.dry_mass_kg
    assert states[-1, 3] == pytest.approx(expected_velocity, abs=1e-9)
    assert states[-1, 1] == pytest.approx(expected_height, abs=1e-9)


def test_mass_remains_at_dry_mass_after_burnout() -> None:
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, PARAMETERS.dry_mass_kg + 0.5)

    _, states = simulate_planar(initial, FULL_THRUST, PARAMETERS, 3.0, 0.05)

    assert states[-1, 6] == PARAMETERS.dry_mass_kg
    assert np.all(states[:, 6] >= PARAMETERS.dry_mass_kg)
    assert np.all(np.diff(states[:, 6]) <= 1e-12)
