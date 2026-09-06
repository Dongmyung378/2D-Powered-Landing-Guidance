"""Day 3 tests for actuator clipping and planar rotational dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    angular_acceleration_rad_s2,
    clip_control,
    simulate_planar,
    state_derivative,
    thrust_torque_nm,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "default.yaml")
PARAMETERS = PlanarDynamicsParameters.from_config(CONFIG)
STATE_WITH_FUEL = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)


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
        thrust_torque_nm(STATE_WITH_FUEL, command, PARAMETERS)
        / PARAMETERS.moment_of_inertia_kg_m2
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
