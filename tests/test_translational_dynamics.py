"""Analytic tests for the Day 2 translational dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    TranslationalParameters,
    integrate_fixed_step,
    simulate_translational,
    state_derivative,
    thrust_vector,
    translational_acceleration,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "default.yaml")
PARAMETERS = TranslationalParameters.from_config(CONFIG)


def test_free_fall_matches_analytic_solution_with_rk4() -> None:
    initial = State(x=5.0, z=100.0, vx=2.0, vz=-3.0, theta=0.0, omega=0.0, mass=900.0)
    control = Control(throttle=0.0, gimbal_angle=0.0)

    times, states = simulate_translational(
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


def test_vertical_thrust_matches_constant_acceleration_solution() -> None:
    initial = State(x=0.0, z=10.0, vx=0.0, vz=1.0, theta=0.0, omega=0.0, mass=1000.0)
    control = Control(throttle=1.0, gimbal_angle=0.0)
    duration_s = 3.0

    times, states = simulate_translational(
        initial,
        control,
        PARAMETERS,
        duration_s=duration_s,
        dt_s=0.02,
        method="rk4",
    )

    acceleration_z = PARAMETERS.max_thrust_n / initial.mass - PARAMETERS.gravity_m_s2
    expected_z = initial.z + initial.vz * times + 0.5 * acceleration_z * times**2
    expected_vz = initial.vz + acceleration_z * times
    np.testing.assert_allclose(states[:, 1], expected_z, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(states[:, 3], expected_vz, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(states[:, 0], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(states[:, 2], 0.0, rtol=0.0, atol=1e-12)
    expected_constants = np.broadcast_to(initial.as_array()[4:], states[:, 4:].shape)
    np.testing.assert_allclose(states[:, 4:], expected_constants, rtol=0.0, atol=1e-12)


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

    _, coarse_states = simulate_translational(
        initial, control, PARAMETERS, duration_s, 0.1, method="euler"
    )
    _, fine_states = simulate_translational(
        initial, control, PARAMETERS, duration_s, 0.01, method="euler"
    )

    coarse_error = abs(coarse_states[-1, 1] - expected_z)
    fine_error = abs(fine_states[-1, 1] - expected_z)
    assert fine_error < coarse_error / 5.0


def test_integrator_shortens_final_step_to_end_exactly() -> None:
    def unit_rate(_time_s: float, state: np.ndarray) -> np.ndarray:
        return np.ones_like(state)

    times, states = integrate_fixed_step(unit_rate, [0.0], (0.0, 1.0), 0.3, method="rk4")

    assert times[-1] == 1.0
    np.testing.assert_allclose(states[-1], [1.0], rtol=0.0, atol=1e-12)


def test_state_derivative_keeps_mass_and_angular_rate_constant() -> None:
    state = State(0.0, 20.0, 1.0, -2.0, 0.1, 0.2, 900.0)
    control = Control(0.2, 0.0)

    derivative = state_derivative(0.0, state.as_array(), control, PARAMETERS)

    assert derivative[4] == state.omega
    assert derivative[5] == 0.0
    assert derivative[6] == 0.0
