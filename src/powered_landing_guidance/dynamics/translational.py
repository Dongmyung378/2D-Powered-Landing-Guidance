"""Day 2 translational dynamics for the planar reusable-rocket model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powered_landing_guidance.dynamics.integrators import IntegrationMethod, integrate_fixed_step
from powered_landing_guidance.model import Control, State


@dataclass(frozen=True, slots=True)
class TranslationalParameters:
    """Physical parameters required by the Day 2 translational model."""

    gravity_m_s2: float
    max_thrust_n: float
    dry_mass_kg: float

    def __post_init__(self) -> None:
        values = (self.gravity_m_s2, self.max_thrust_n, self.dry_mass_kg)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("all translational parameters must be positive finite numbers")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TranslationalParameters:
        """Construct parameters from a validated project configuration."""
        simulation = config["simulation"]
        vehicle = config["vehicle"]
        return cls(
            gravity_m_s2=float(simulation["gravity_m_s2"]),
            max_thrust_n=float(vehicle["max_thrust_n"]),
            dry_mass_kg=float(vehicle["dry_mass_kg"]),
        )


def thrust_vector(
    state: State,
    control: Control,
    parameters: TranslationalParameters,
) -> NDArray[np.float64]:
    """Return inertial ``[Fx, Fz]`` from throttle and thrust-vector angle."""
    thrust_n = control.throttle * parameters.max_thrust_n
    direction_rad = state.theta + control.gimbal_angle
    return thrust_n * np.asarray((np.sin(direction_rad), np.cos(direction_rad)))


def translational_acceleration(
    state: State,
    control: Control,
    parameters: TranslationalParameters,
) -> NDArray[np.float64]:
    """Return inertial ``[ax, az]`` using the state's current mass."""
    if state.mass < parameters.dry_mass_kg:
        raise ValueError("state mass cannot be below dry mass")
    acceleration = thrust_vector(state, control, parameters) / state.mass
    acceleration[1] -= parameters.gravity_m_s2
    return acceleration


def state_derivative(
    _time_s: float,
    state_vector: ArrayLike,
    control: Control,
    parameters: TranslationalParameters,
) -> NDArray[np.float64]:
    """Return the Day 2 derivative of ``[x, z, vx, vz, theta, omega, mass]``."""
    state = State.from_array(state_vector)
    ax, az = translational_acceleration(state, control, parameters)
    return np.asarray(
        (
            state.vx,
            state.vz,
            ax,
            az,
            state.omega,
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )


def simulate_translational(
    initial_state: State,
    control: Control,
    parameters: TranslationalParameters,
    duration_s: float,
    dt_s: float,
    *,
    method: IntegrationMethod = "rk4",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate the Day 2 model under a constant control command."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be a positive finite number")

    def derivative(time_s: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return state_derivative(time_s, state, control, parameters)

    return integrate_fixed_step(
        derivative,
        initial_state.as_array(),
        (0.0, duration_s),
        dt_s,
        method=method,
    )
