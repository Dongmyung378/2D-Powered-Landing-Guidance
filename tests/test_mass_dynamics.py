"""Day 4 tests for propellant consumption, dry mass, and thrust cutoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    applied_thrust_n,
    propellant_mass_flow_rate_kg_s,
    simulate_planar,
    state_derivative,
    thrust_torque_nm,
    thrust_vector,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "default.yaml")
PARAMETERS = PlanarDynamicsParameters.from_config(CONFIG)
FULL_THRUST = Control(1.0, 0.0)


def test_mass_flow_matches_specific_impulse_equation() -> None:
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    expected = PARAMETERS.max_thrust_n / (
        PARAMETERS.specific_impulse_s * PARAMETERS.standard_gravity_m_s2
    )

    assert propellant_mass_flow_rate_kg_s(state, FULL_THRUST, PARAMETERS) == pytest.approx(
        expected
    )


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
        initial.mass
        - PARAMETERS.dry_mass_kg
        - PARAMETERS.dry_mass_kg * log_mass_ratio
    ) / mass_flow_rate
    burnout_velocity = (
        initial.vz
        + exhaust_velocity * log_mass_ratio
        - PARAMETERS.gravity_m_s2 * burnout_time_s
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
