"""Planar rigid-body dynamics with variable mass and thrust cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powered_landing_guidance.integrators import (
    IntegrationMethod,
    euler_step,
    rk4_step,
)
from powered_landing_guidance.integrators import (
    integrate_fixed_step as integrate_fixed_step,
)
from powered_landing_guidance.model import Control, State

type ControlInput = Control | ArrayLike


@dataclass(frozen=True, slots=True)
class PlanarDynamicsParameters:
    """Physical and actuator parameters for the planar rigid-body model."""

    gravity_m_s2: float
    max_thrust_n: float
    dry_mass_kg: float
    specific_impulse_s: float
    standard_gravity_m_s2: float
    moment_of_inertia_kg_m2: float
    engine_lever_arm_m: float
    throttle_min: float
    throttle_max: float
    gimbal_limit_rad: float

    @property
    def mass_tolerance_kg(self) -> float:
        """Roundoff allowance shared by fuel cutoff and event detection."""
        return float(np.finfo(np.float64).eps * max(1.0, self.dry_mass_kg) * 8.0)

    def __post_init__(self) -> None:
        positive_values = (
            self.gravity_m_s2,
            self.max_thrust_n,
            self.dry_mass_kg,
            self.specific_impulse_s,
            self.standard_gravity_m_s2,
            self.moment_of_inertia_kg_m2,
            self.engine_lever_arm_m,
            self.gimbal_limit_rad,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive_values):
            raise ValueError("all physical parameters and the gimbal limit must be positive")
        if not np.isfinite(self.throttle_min) or not np.isfinite(self.throttle_max):
            raise ValueError("throttle limits must be finite")
        if not 0.0 <= self.throttle_min < self.throttle_max <= 1.0:
            raise ValueError("throttle limits must satisfy 0 <= min < max <= 1")
        if self.gimbal_limit_rad > np.pi / 2.0:
            raise ValueError("gimbal limit cannot exceed pi/2 radians")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PlanarDynamicsParameters:
        """Construct parameters and convert the configured gimbal limit to radians."""
        simulation = config["simulation"]
        vehicle = config["vehicle"]
        return cls(
            gravity_m_s2=float(simulation["gravity_m_s2"]),
            max_thrust_n=float(vehicle["max_thrust_n"]),
            dry_mass_kg=float(vehicle["dry_mass_kg"]),
            specific_impulse_s=float(vehicle["specific_impulse_s"]),
            standard_gravity_m_s2=float(vehicle["standard_gravity_m_s2"]),
            moment_of_inertia_kg_m2=float(vehicle["moment_of_inertia_kg_m2"]),
            engine_lever_arm_m=float(vehicle["engine_lever_arm_m"]),
            throttle_min=float(vehicle["throttle_min"]),
            throttle_max=float(vehicle["throttle_max"]),
            gimbal_limit_rad=float(np.deg2rad(vehicle["gimbal_limit_deg"])),
        )


def _control_vector(command: ControlInput) -> NDArray[np.float64]:
    vector = (
        command.as_array() if isinstance(command, Control) else np.asarray(command, dtype=float)
    )
    if vector.shape != (2,):
        raise ValueError(f"control command must have shape (2,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("control command must contain only finite values")
    return vector


def clip_control(command: ControlInput, parameters: PlanarDynamicsParameters) -> Control:
    """Clamp a raw ``[throttle, gimbal]`` command to configured actuator limits."""
    throttle, gimbal_angle = _control_vector(command)
    return Control(
        throttle=float(np.clip(throttle, parameters.throttle_min, parameters.throttle_max)),
        gimbal_angle=float(
            np.clip(gimbal_angle, -parameters.gimbal_limit_rad, parameters.gimbal_limit_rad)
        ),
    )


def _validate_mass(state: State, parameters: PlanarDynamicsParameters) -> None:
    if state.mass < parameters.dry_mass_kg - parameters.mass_tolerance_kg:
        raise ValueError("state mass cannot be below dry mass")


def _thrust_after_cutoff(
    state: State, control: Control, parameters: PlanarDynamicsParameters
) -> float:
    """Apply fuel cutoff to an already-clipped command."""
    _validate_mass(state, parameters)
    if state.mass <= parameters.dry_mass_kg + parameters.mass_tolerance_kg:
        return 0.0
    return control.throttle * parameters.max_thrust_n


def applied_thrust_n(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> float:
    """Return thrust after command clipping and dry-mass cutoff."""
    control = clip_control(command, parameters)
    return _thrust_after_cutoff(state, control, parameters)


def propellant_mass_flow_rate_kg_s(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> float:
    """Return positive propellant consumption rate from thrust and specific impulse."""
    thrust_n = applied_thrust_n(state, command, parameters)
    exhaust_velocity_m_s = parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    return thrust_n / exhaust_velocity_m_s


def thrust_vector(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> NDArray[np.float64]:
    """Return inertial ``[Fx, Fz]`` after actuator and fuel limits."""
    control = clip_control(command, parameters)
    thrust_n = _thrust_after_cutoff(state, control, parameters)
    direction_rad = state.theta + control.gimbal_angle
    return thrust_n * np.asarray((np.sin(direction_rad), np.cos(direction_rad)))


def thrust_torque_nm(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> float:
    """Return signed thrust torque about the center of mass.

    Positive attitude is clockwise toward ``+x``. Because the engine is below
    the center of mass, a positive thrust-vector deflection produces a negative
    attitude torque under the documented convention.
    """
    control = clip_control(command, parameters)
    thrust_n = _thrust_after_cutoff(state, control, parameters)
    return float(-parameters.engine_lever_arm_m * thrust_n * np.sin(control.gimbal_angle))


def translational_acceleration(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> NDArray[np.float64]:
    """Return inertial ``[ax, az]`` using the state's current mass."""
    acceleration = thrust_vector(state, command, parameters) / state.mass
    acceleration[1] -= parameters.gravity_m_s2
    return acceleration


def angular_acceleration_rad_s2(
    state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> float:
    """Return angular acceleration from thrust torque and moment of inertia."""
    return thrust_torque_nm(state, command, parameters) / parameters.moment_of_inertia_kg_m2


def _derivative_from_thrust(
    state: State,
    control: Control,
    parameters: PlanarDynamicsParameters,
    thrust_n: float,
) -> NDArray[np.float64]:
    direction_rad = state.theta + control.gimbal_angle
    force = thrust_n * np.asarray((np.sin(direction_rad), np.cos(direction_rad)))
    ax = force[0] / state.mass
    az = force[1] / state.mass - parameters.gravity_m_s2
    torque_nm = -parameters.engine_lever_arm_m * thrust_n * np.sin(control.gimbal_angle)
    angular_acceleration = torque_nm / parameters.moment_of_inertia_kg_m2
    mass_flow_rate = thrust_n / (parameters.specific_impulse_s * parameters.standard_gravity_m_s2)
    return np.asarray(
        (
            state.vx,
            state.vz,
            ax,
            az,
            state.omega,
            angular_acceleration,
            -mass_flow_rate,
        ),
        dtype=np.float64,
    )


def state_derivative(
    _time_s: float,
    state_vector: ArrayLike,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
) -> NDArray[np.float64]:
    """Return ``d/dt [x, z, vx, vz, theta, omega, mass]``."""
    state = State.from_array(state_vector)
    applied_control = clip_control(command, parameters)
    thrust_n = _thrust_after_cutoff(state, applied_control, parameters)
    return _derivative_from_thrust(state, applied_control, parameters, thrust_n)


def simulate_planar(
    initial_state: State,
    command: ControlInput,
    parameters: PlanarDynamicsParameters,
    duration_s: float,
    dt_s: float,
    *,
    method: IntegrationMethod = "rk4",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate planar motion with exact dry-mass event splitting.

    If propellant runs out inside a numerical step, the powered and ballistic
    portions are integrated separately. The returned trajectory remains on the
    requested time grid and mass is projected exactly to dry mass at burnout.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be a positive finite number")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be a positive finite number")
    steppers = {"euler": euler_step, "rk4": rk4_step}
    try:
        stepper = steppers[method]
    except KeyError as error:
        raise ValueError(f"unsupported integration method: {method}") from error

    applied_control = clip_control(command, parameters)
    initial_vector = initial_state.as_array()
    _validate_mass(initial_state, parameters)
    nominal_thrust_n = applied_control.throttle * parameters.max_thrust_n
    nominal_mass_flow = nominal_thrust_n / (
        parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, duration_s) * 8.0
    mass_tolerance = parameters.mass_tolerance_kg

    times = [0.0]
    states = [initial_vector.copy()]

    def powered_derivative(_time_s: float, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        state = State.from_array(vector)
        return _derivative_from_thrust(state, applied_control, parameters, nominal_thrust_n)

    def unpowered_derivative(_time_s: float, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return _derivative_from_thrust(State.from_array(vector), applied_control, parameters, 0.0)

    while times[-1] < duration_s - tolerance:
        step_start_s = times[-1]
        step_end_s = min(step_start_s + dt_s, duration_s)
        remaining_step_s = step_end_s - step_start_s
        local_time_s = step_start_s
        next_state = states[-1].copy()
        fuel_mass_kg = max(0.0, next_state[6] - parameters.dry_mass_kg)
        if fuel_mass_kg <= mass_tolerance:
            next_state[6] = parameters.dry_mass_kg

        if nominal_mass_flow > 0.0 and fuel_mass_kg > mass_tolerance:
            burnout_time_s = fuel_mass_kg / nominal_mass_flow
            powered_step_s = min(remaining_step_s, burnout_time_s)
            next_state = stepper(
                powered_derivative,
                local_time_s,
                next_state,
                powered_step_s,
            )
            local_time_s += powered_step_s
            remaining_step_s -= powered_step_s
            if powered_step_s >= burnout_time_s - tolerance:
                next_state[6] = parameters.dry_mass_kg

        if remaining_step_s > tolerance:
            next_state[6] = max(next_state[6], parameters.dry_mass_kg)
            next_state = stepper(
                unpowered_derivative,
                local_time_s,
                next_state,
                remaining_step_s,
            )

        next_state[6] = max(next_state[6], parameters.dry_mass_kg)
        times.append(step_end_s)
        states.append(next_state)

    times[-1] = duration_s
    return np.asarray(times, dtype=np.float64), np.stack(states)
