"""Heuristic controllers for powered-landing baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from powered_landing_guidance.dynamics import PlanarDynamicsParameters
from powered_landing_guidance.model import State

type IgnitionTiming = Literal["early_burn", "on_time", "late_burn", "infeasible"]


@dataclass(frozen=True, slots=True)
class SuicideBurnEstimate:
    """Estimate for a vertical full-thrust braking burn."""

    altitude_m: float
    downward_speed_m_s: float
    target_touchdown_speed_m_s: float
    net_deceleration_m_s2: float
    stopping_distance_m: float
    reaction_distance_m: float
    ignition_height_m: float
    burn_time_s: float
    propellant_required_kg: float
    propellant_available_kg: float
    feasible: bool
    mass_model: Literal["constant", "variable"]


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def estimate_suicide_burn(
    state: State | ArrayLike,
    parameters: PlanarDynamicsParameters,
    *,
    ground_z_m: float = 0.0,
    target_touchdown_speed_m_s: float = 2.0,
    reaction_time_s: float = 0.0,
) -> SuicideBurnEstimate:
    """Estimate the full-thrust ignition height for vertical descent.

    Mass is held at its current value throughout the estimate. The simulator
    still uses the variable-mass dynamics, so this intentionally conservative
    estimate is only the first non-learning baseline.
    """
    current = state if isinstance(state, State) else State.from_array(state)
    ground = float(ground_z_m)
    if not np.isfinite(ground):
        raise ValueError("ground_z_m must be finite")
    altitude = current.z - ground
    if altitude < 0.0:
        raise ValueError("state cannot be below ground")
    if current.mass < parameters.dry_mass_kg - parameters.mass_tolerance_kg:
        raise ValueError("state mass cannot be below dry mass")

    target_speed = _nonnegative(target_touchdown_speed_m_s, "target_touchdown_speed_m_s")
    reaction_time = _nonnegative(reaction_time_s, "reaction_time_s")
    downward_speed = max(0.0, -current.vz)
    required_delta_v = max(0.0, downward_speed - target_speed)
    net_deceleration = parameters.max_thrust_n / current.mass - parameters.gravity_m_s2
    available_propellant = max(0.0, current.mass - parameters.dry_mass_kg)

    if net_deceleration <= 0.0:
        stopping_distance = np.inf
        burn_time = np.inf
        required_propellant = np.inf
        ignition_height = np.inf
        feasible = False
    else:
        stopping_distance = max(0.0, downward_speed**2 - target_speed**2) / (2.0 * net_deceleration)
        burn_time = required_delta_v / net_deceleration
        mass_flow_rate = parameters.max_thrust_n / (
            parameters.specific_impulse_s * parameters.standard_gravity_m_s2
        )
        required_propellant = mass_flow_rate * burn_time
        reaction_distance = (
            downward_speed * reaction_time + 0.5 * parameters.gravity_m_s2 * reaction_time**2
        )
        ignition_height = stopping_distance + reaction_distance
        feasible = required_propellant <= available_propellant + parameters.mass_tolerance_kg

    if net_deceleration <= 0.0:
        reaction_distance = 0.0

    return SuicideBurnEstimate(
        altitude_m=float(altitude),
        downward_speed_m_s=float(downward_speed),
        target_touchdown_speed_m_s=target_speed,
        net_deceleration_m_s2=float(net_deceleration),
        stopping_distance_m=float(stopping_distance),
        reaction_distance_m=float(reaction_distance),
        ignition_height_m=float(ignition_height),
        burn_time_s=float(burn_time),
        propellant_required_kg=float(required_propellant),
        propellant_available_kg=float(available_propellant),
        feasible=bool(feasible),
        mass_model="constant",
    )


def estimate_variable_mass_suicide_burn(
    state: State | ArrayLike,
    parameters: PlanarDynamicsParameters,
    *,
    ground_z_m: float = 0.0,
    target_touchdown_speed_m_s: float = 2.0,
    reaction_time_s: float = 0.0,
) -> SuicideBurnEstimate:
    """Estimate ignition height with the analytic variable-mass vertical solution."""
    current = state if isinstance(state, State) else State.from_array(state)
    ground = float(ground_z_m)
    if not np.isfinite(ground):
        raise ValueError("ground_z_m must be finite")
    altitude = current.z - ground
    if altitude < 0.0:
        raise ValueError("state cannot be below ground")
    if current.mass < parameters.dry_mass_kg - parameters.mass_tolerance_kg:
        raise ValueError("state mass cannot be below dry mass")

    target_speed = _nonnegative(target_touchdown_speed_m_s, "target_touchdown_speed_m_s")
    reaction_time = _nonnegative(reaction_time_s, "reaction_time_s")
    downward_speed = max(0.0, -current.vz)
    required_delta_v = max(0.0, downward_speed - target_speed)
    net_deceleration = parameters.max_thrust_n / current.mass - parameters.gravity_m_s2
    available_propellant = max(0.0, current.mass - parameters.dry_mass_kg)
    mass_flow_rate = parameters.max_thrust_n / (
        parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    )
    exhaust_velocity = parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    available_burn_time = available_propellant / mass_flow_rate

    def velocity_after_burn(time_s: float) -> float:
        final_mass = current.mass - mass_flow_rate * time_s
        return (
            current.vz
            + exhaust_velocity * np.log(current.mass / final_mass)
            - parameters.gravity_m_s2 * time_s
        )

    feasible = net_deceleration > 0.0
    if required_delta_v == 0.0 and feasible:
        burn_time = 0.0
        stopping_distance = 0.0
        required_propellant = 0.0
    elif (
        feasible
        and available_burn_time > 0.0
        and velocity_after_burn(available_burn_time) >= -target_speed
    ):
        burn_time = brentq(
            lambda time_s: velocity_after_burn(time_s) + target_speed,
            0.0,
            available_burn_time,
            xtol=1e-12,
        )
        final_mass = current.mass - mass_flow_rate * burn_time
        log_mass_ratio = np.log(current.mass / final_mass)
        integrated_log_ratio = (
            current.mass - final_mass - final_mass * log_mass_ratio
        ) / mass_flow_rate
        displacement = (
            current.vz * burn_time
            + exhaust_velocity * integrated_log_ratio
            - 0.5 * parameters.gravity_m_s2 * burn_time**2
        )
        stopping_distance = max(0.0, -displacement)
        required_propellant = mass_flow_rate * burn_time
    else:
        feasible = False
        burn_time = np.inf
        stopping_distance = np.inf
        required_propellant = np.inf

    reaction_distance = (
        downward_speed * reaction_time + 0.5 * parameters.gravity_m_s2 * reaction_time**2
    )
    ignition_height = stopping_distance + reaction_distance
    return SuicideBurnEstimate(
        altitude_m=float(altitude),
        downward_speed_m_s=float(downward_speed),
        target_touchdown_speed_m_s=target_speed,
        net_deceleration_m_s2=float(net_deceleration),
        stopping_distance_m=float(stopping_distance),
        reaction_distance_m=float(reaction_distance),
        ignition_height_m=float(ignition_height),
        burn_time_s=float(burn_time),
        propellant_required_kg=float(required_propellant),
        propellant_available_kg=float(available_propellant),
        feasible=bool(feasible),
        mass_model="variable",
    )


def classify_ignition_timing(
    actual_ignition_height_m: float,
    required_ignition_height_m: float,
    *,
    tolerance_m: float = 0.5,
) -> Literal["early_burn", "on_time", "late_burn"]:
    """Classify an ignition relative to the estimated required height."""
    actual = _nonnegative(actual_ignition_height_m, "actual_ignition_height_m")
    required = _nonnegative(required_ignition_height_m, "required_ignition_height_m")
    tolerance = _nonnegative(tolerance_m, "tolerance_m")
    error = actual - required
    if error > tolerance:
        return "early_burn"
    if error < -tolerance:
        return "late_burn"
    return "on_time"


@dataclass(slots=True)
class SuicideBurnController:
    """Vertical bang-bang controller using a stopping-distance trigger."""

    parameters: PlanarDynamicsParameters
    ground_z_m: float
    target_touchdown_speed_m_s: float
    control_interval_s: float
    reaction_time_s: float
    timing_tolerance_m: float = 0.5
    account_for_mass_loss: bool = True
    ignited: bool = field(init=False, default=False)
    actual_ignition_height_m: float | None = field(init=False, default=None)
    required_ignition_height_m: float | None = field(init=False, default=None)
    ignition_timing: IgnitionTiming | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.ground_z_m = float(self.ground_z_m)
        if not np.isfinite(self.ground_z_m):
            raise ValueError("ground_z_m must be finite")
        self.target_touchdown_speed_m_s = _nonnegative(
            self.target_touchdown_speed_m_s, "target_touchdown_speed_m_s"
        )
        self.control_interval_s = _nonnegative(self.control_interval_s, "control_interval_s")
        if self.control_interval_s == 0.0:
            raise ValueError("control_interval_s must be positive")
        self.reaction_time_s = _nonnegative(self.reaction_time_s, "reaction_time_s")
        self.timing_tolerance_m = _nonnegative(self.timing_tolerance_m, "timing_tolerance_m")
        if not isinstance(self.account_for_mass_loss, bool):
            raise ValueError("account_for_mass_loss must be boolean")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SuicideBurnController:
        """Build the baseline from the shared project configuration."""
        return cls(
            parameters=PlanarDynamicsParameters.from_config(config),
            ground_z_m=float(config["simulation"]["ground_z_m"]),
            target_touchdown_speed_m_s=float(config["landing_success"]["max_abs_vz_m_s"]),
            control_interval_s=float(config["simulation"]["dt_s"]),
            reaction_time_s=0.0,
        )

    def reset(self) -> None:
        """Clear the ignition latch before a new episode."""
        self.ignited = False
        self.actual_ignition_height_m = None
        self.required_ignition_height_m = None
        self.ignition_timing = None

    def estimate(self, state: State | ArrayLike) -> SuicideBurnEstimate:
        """Return the current burn estimate."""
        estimator = (
            estimate_variable_mass_suicide_burn
            if self.account_for_mass_loss
            else estimate_suicide_burn
        )
        return estimator(
            state,
            self.parameters,
            ground_z_m=self.ground_z_m,
            target_touchdown_speed_m_s=self.target_touchdown_speed_m_s,
            reaction_time_s=self.reaction_time_s,
        )

    def _coast_state(self, state: State, duration_s: float) -> State:
        gravity = self.parameters.gravity_m_s2
        return State(
            x=state.x + state.vx * duration_s,
            z=state.z + state.vz * duration_s - 0.5 * gravity * duration_s**2,
            vx=state.vx,
            vz=state.vz - gravity * duration_s,
            theta=state.theta + state.omega * duration_s,
            omega=state.omega,
            mass=state.mass,
        )

    def _record_ignition(
        self,
        actual_height_m: float,
        required_height_m: float,
        timing: IgnitionTiming | None = None,
    ) -> None:
        self.ignited = True
        self.actual_ignition_height_m = actual_height_m
        self.required_ignition_height_m = required_height_m
        self.ignition_timing = timing or classify_ignition_timing(
            actual_height_m,
            required_height_m,
            tolerance_m=self.timing_tolerance_m,
        )

    def command(self, state: State | ArrayLike) -> NDArray[np.float64]:
        """Return [throttle, gimbal] with sub-step ignition compensation."""
        current = state if isinstance(state, State) else State.from_array(state)
        estimate = self.estimate(current)
        if self.ignited or estimate.downward_speed_m_s == 0.0:
            return np.asarray((1.0 if self.ignited else 0.0, 0.0), dtype=np.float64)

        if not estimate.feasible:
            self._record_ignition(estimate.altitude_m, estimate.ignition_height_m, "infeasible")
            return np.asarray((1.0, 0.0), dtype=np.float64)
        if estimate.altitude_m <= estimate.ignition_height_m:
            self._record_ignition(estimate.altitude_m, estimate.ignition_height_m)
            return np.asarray((1.0, 0.0), dtype=np.float64)

        coasted = self._coast_state(current, self.control_interval_s)
        if coasted.z <= self.ground_z_m:
            self._record_ignition(estimate.altitude_m, estimate.ignition_height_m, "late_burn")
            return np.asarray((1.0, 0.0), dtype=np.float64)
        coasted_estimate = self.estimate(coasted)
        if not coasted_estimate.feasible:
            self._record_ignition(estimate.altitude_m, estimate.ignition_height_m, "infeasible")
            return np.asarray((1.0, 0.0), dtype=np.float64)
        if coasted_estimate.altitude_m > coasted_estimate.ignition_height_m:
            return np.asarray((0.0, 0.0), dtype=np.float64)

        def ignition_margin(time_s: float) -> float:
            coast_state = self._coast_state(current, time_s)
            coast_estimate = self.estimate(coast_state)
            return coast_estimate.altitude_m - coast_estimate.ignition_height_m

        ignition_time = brentq(
            ignition_margin,
            0.0,
            self.control_interval_s,
            xtol=1e-12,
        )
        ignition_state = self._coast_state(current, ignition_time)
        ignition_estimate = self.estimate(ignition_state)
        self._record_ignition(
            ignition_estimate.altitude_m,
            ignition_estimate.ignition_height_m,
        )
        throttle = 1.0 - ignition_time / self.control_interval_s
        return np.asarray((throttle, 0.0), dtype=np.float64)
