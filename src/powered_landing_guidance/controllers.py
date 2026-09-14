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


def _positive(value: float, name: str) -> float:
    number = _nonnegative(value, name)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True, slots=True)
class VerticalVelocityProfile:
    """Altitude-dependent vertical-speed reference for terminal descent."""

    touchdown_speed_m_s: float
    max_descent_speed_m_s: float
    deceleration_m_s2: float

    def __post_init__(self) -> None:
        touchdown = _positive(self.touchdown_speed_m_s, "touchdown_speed_m_s")
        maximum = _positive(self.max_descent_speed_m_s, "max_descent_speed_m_s")
        deceleration = _positive(self.deceleration_m_s2, "deceleration_m_s2")
        if maximum < touchdown:
            raise ValueError("max_descent_speed_m_s must be at least touchdown_speed_m_s")
        object.__setattr__(self, "touchdown_speed_m_s", touchdown)
        object.__setattr__(self, "max_descent_speed_m_s", maximum)
        object.__setattr__(self, "deceleration_m_s2", deceleration)

    def target_velocity_m_s(self, altitude_m: float) -> float:
        """Return the downward reference velocity at an altitude above ground."""
        altitude = _nonnegative(altitude_m, "altitude_m")
        speed = np.sqrt(self.touchdown_speed_m_s**2 + 2.0 * self.deceleration_m_s2 * altitude)
        return -min(self.max_descent_speed_m_s, float(speed))

    def target_acceleration_m_s2(self, altitude_m: float) -> float:
        """Return the feed-forward acceleration along the active profile branch."""
        altitude = _nonnegative(altitude_m, "altitude_m")
        unconstrained_speed = np.sqrt(
            self.touchdown_speed_m_s**2 + 2.0 * self.deceleration_m_s2 * altitude
        )
        if unconstrained_speed >= self.max_descent_speed_m_s:
            return 0.0
        return self.deceleration_m_s2


@dataclass(slots=True)
class VerticalVelocityPIDController:
    """Track a vertical-speed profile with feed-forward PID throttle control."""

    parameters: PlanarDynamicsParameters
    ground_z_m: float
    control_interval_s: float
    profile: VerticalVelocityProfile
    kp: float
    ki: float
    kd: float
    integral_limit_m: float
    integral_error_m: float = field(init=False, default=0.0)
    previous_error_m_s: float | None = field(init=False, default=None)
    target_vertical_speed_m_s: float | None = field(init=False, default=None)
    unsaturated_throttle: float | None = field(init=False, default=None)
    throttle: float | None = field(init=False, default=None)
    saturated: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.ground_z_m = float(self.ground_z_m)
        if not np.isfinite(self.ground_z_m):
            raise ValueError("ground_z_m must be finite")
        self.control_interval_s = _positive(self.control_interval_s, "control_interval_s")
        self.kp = _nonnegative(self.kp, "kp")
        self.ki = _nonnegative(self.ki, "ki")
        self.kd = _nonnegative(self.kd, "kd")
        self.integral_limit_m = _positive(self.integral_limit_m, "integral_limit_m")

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
    ) -> VerticalVelocityPIDController:
        """Build the controller, optionally overriding gains for a sweep."""
        settings = config["vertical_velocity_controller"]
        profile = settings["profile"]
        gains = settings["gains"]
        return cls(
            parameters=PlanarDynamicsParameters.from_config(config),
            ground_z_m=float(config["simulation"]["ground_z_m"]),
            control_interval_s=float(config["simulation"]["dt_s"]),
            profile=VerticalVelocityProfile(
                touchdown_speed_m_s=float(profile["touchdown_speed_m_s"]),
                max_descent_speed_m_s=float(profile["max_descent_speed_m_s"]),
                deceleration_m_s2=float(profile["deceleration_m_s2"]),
            ),
            kp=float(gains["kp"] if kp is None else kp),
            ki=float(gains["ki"] if ki is None else ki),
            kd=float(gains["kd"] if kd is None else kd),
            integral_limit_m=float(settings["anti_windup"]["integral_limit_m"]),
        )

    def reset(self) -> None:
        """Clear state accumulated during the previous episode."""
        self.integral_error_m = 0.0
        self.previous_error_m_s = None
        self.target_vertical_speed_m_s = None
        self.unsaturated_throttle = None
        self.throttle = None
        self.saturated = False

    def command(self, state: State | ArrayLike) -> NDArray[np.float64]:
        """Return a saturated [throttle, gimbal] command and update PID state."""
        current = state if isinstance(state, State) else State.from_array(state)
        altitude = current.z - self.ground_z_m
        if altitude < 0.0:
            raise ValueError("state cannot be below ground")

        target_velocity = self.profile.target_velocity_m_s(altitude)
        error = target_velocity - current.vz
        derivative = (
            0.0
            if self.previous_error_m_s is None
            else (error - self.previous_error_m_s) / self.control_interval_s
        )
        candidate_integral = float(
            np.clip(
                self.integral_error_m + error * self.control_interval_s,
                -self.integral_limit_m,
                self.integral_limit_m,
            )
        )
        target_acceleration = self.profile.target_acceleration_m_s2(altitude)
        feed_forward = (
            current.mass
            * (self.parameters.gravity_m_s2 + target_acceleration)
            / self.parameters.max_thrust_n
        )

        def raw_throttle(integral: float) -> float:
            return feed_forward + self.kp * error + self.ki * integral + self.kd * derivative

        unsaturated = raw_throttle(candidate_integral)
        minimum = self.parameters.throttle_min
        maximum = self.parameters.throttle_max
        blocks_integration = (unsaturated > maximum and error > 0.0) or (
            unsaturated < minimum and error < 0.0
        )
        if blocks_integration:
            candidate_integral = self.integral_error_m
            unsaturated = raw_throttle(candidate_integral)

        throttle = float(np.clip(unsaturated, minimum, maximum))
        self.integral_error_m = candidate_integral
        self.previous_error_m_s = error
        self.target_vertical_speed_m_s = target_velocity
        self.unsaturated_throttle = float(unsaturated)
        self.throttle = throttle
        self.saturated = not np.isclose(throttle, unsaturated, rtol=0.0, atol=1e-12)
        return np.asarray((throttle, 0.0), dtype=np.float64)


@dataclass(slots=True)
class HorizontalAttitudeController:
    """Drive horizontal error toward a target with cascaded position and attitude loops."""

    parameters: PlanarDynamicsParameters
    target_x_m: float
    position_kp_s2: float
    velocity_kd_s: float
    max_horizontal_acceleration_m_s2: float
    max_tilt_rad: float
    attitude_kp_s2: float
    angular_rate_kd_s: float
    compensate_vertical_thrust: bool = True
    horizontal_acceleration_m_s2: float | None = field(init=False, default=None)
    target_attitude_rad: float | None = field(init=False, default=None)
    desired_angular_acceleration_rad_s2: float | None = field(init=False, default=None)
    uncompensated_throttle: float | None = field(init=False, default=None)
    throttle: float | None = field(init=False, default=None)
    gimbal_angle_rad: float | None = field(init=False, default=None)
    tilt_limited: bool = field(init=False, default=False)
    gimbal_limited: bool = field(init=False, default=False)
    throttle_limited: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.target_x_m = float(self.target_x_m)
        if not np.isfinite(self.target_x_m):
            raise ValueError("target_x_m must be finite")
        self.position_kp_s2 = _nonnegative(self.position_kp_s2, "position_kp_s2")
        self.velocity_kd_s = _nonnegative(self.velocity_kd_s, "velocity_kd_s")
        self.max_horizontal_acceleration_m_s2 = _positive(
            self.max_horizontal_acceleration_m_s2,
            "max_horizontal_acceleration_m_s2",
        )
        self.max_tilt_rad = _positive(self.max_tilt_rad, "max_tilt_rad")
        if self.max_tilt_rad >= np.pi / 2.0:
            raise ValueError("max_tilt_rad must be less than pi / 2")
        self.attitude_kp_s2 = _positive(self.attitude_kp_s2, "attitude_kp_s2")
        self.angular_rate_kd_s = _nonnegative(
            self.angular_rate_kd_s,
            "angular_rate_kd_s",
        )
        if not isinstance(self.compensate_vertical_thrust, bool):
            raise ValueError("compensate_vertical_thrust must be boolean")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HorizontalAttitudeController:
        """Build the cascaded controller from the shared project configuration."""
        settings = config["horizontal_attitude_controller"]
        outer = settings["outer_loop"]
        inner = settings["inner_loop"]
        return cls(
            parameters=PlanarDynamicsParameters.from_config(config),
            target_x_m=float(settings["target_x_m"]),
            position_kp_s2=float(outer["position_kp_s2"]),
            velocity_kd_s=float(outer["velocity_kd_s"]),
            max_horizontal_acceleration_m_s2=float(outer["max_horizontal_acceleration_m_s2"]),
            max_tilt_rad=float(np.deg2rad(outer["max_tilt_deg"])),
            attitude_kp_s2=float(inner["attitude_kp_s2"]),
            angular_rate_kd_s=float(inner["angular_rate_kd_s"]),
            compensate_vertical_thrust=bool(settings["coupling"]["compensate_vertical_thrust"]),
        )

    def reset(self) -> None:
        """Clear telemetry retained from the previous command."""
        self.horizontal_acceleration_m_s2 = None
        self.target_attitude_rad = None
        self.desired_angular_acceleration_rad_s2 = None
        self.uncompensated_throttle = None
        self.throttle = None
        self.gimbal_angle_rad = None
        self.tilt_limited = False
        self.gimbal_limited = False
        self.throttle_limited = False

    def _gimbal_for_angular_acceleration(
        self,
        desired_angular_acceleration_rad_s2: float,
        throttle: float,
    ) -> tuple[float, bool]:
        thrust_n = throttle * self.parameters.max_thrust_n
        if thrust_n <= 0.0:
            return 0.0, not np.isclose(desired_angular_acceleration_rad_s2, 0.0)
        sine_command = (
            -desired_angular_acceleration_rad_s2
            * self.parameters.moment_of_inertia_kg_m2
            / (self.parameters.engine_lever_arm_m * thrust_n)
        )
        force_limited = abs(sine_command) > 1.0
        raw_gimbal = float(np.arcsin(np.clip(sine_command, -1.0, 1.0)))
        gimbal = float(
            np.clip(
                raw_gimbal,
                -self.parameters.gimbal_limit_rad,
                self.parameters.gimbal_limit_rad,
            )
        )
        return gimbal, force_limited or not np.isclose(gimbal, raw_gimbal, atol=1e-12)

    def command(
        self,
        state: State | ArrayLike,
        *,
        base_throttle: float,
    ) -> NDArray[np.float64]:
        """Return throttle and gimbal commands while preserving requested vertical thrust."""
        current = state if isinstance(state, State) else State.from_array(state)
        base = float(base_throttle)
        minimum = self.parameters.throttle_min
        maximum = self.parameters.throttle_max
        if not np.isfinite(base) or not minimum <= base <= maximum:
            raise ValueError("base_throttle must be finite and within actuator limits")

        raw_acceleration = (
            self.position_kp_s2 * (self.target_x_m - current.x) - self.velocity_kd_s * current.vx
        )
        acceleration = float(
            np.clip(
                raw_acceleration,
                -self.max_horizontal_acceleration_m_s2,
                self.max_horizontal_acceleration_m_s2,
            )
        )
        raw_target_attitude = float(np.arctan2(acceleration, self.parameters.gravity_m_s2))
        target_attitude = float(np.clip(raw_target_attitude, -self.max_tilt_rad, self.max_tilt_rad))
        desired_angular_acceleration = (
            self.attitude_kp_s2 * (target_attitude - current.theta)
            - self.angular_rate_kd_s * current.omega
        )

        throttle = base
        gimbal, gimbal_limited = self._gimbal_for_angular_acceleration(
            desired_angular_acceleration,
            throttle,
        )
        if self.compensate_vertical_thrust and base > 0.0:
            for _ in range(8):
                vertical_fraction = max(float(np.cos(current.theta + gimbal)), 1e-9)
                throttle = float(np.clip(base / vertical_fraction, minimum, maximum))
                gimbal, gimbal_limited = self._gimbal_for_angular_acceleration(
                    desired_angular_acceleration,
                    throttle,
                )

        self.horizontal_acceleration_m_s2 = acceleration
        self.target_attitude_rad = target_attitude
        self.desired_angular_acceleration_rad_s2 = float(desired_angular_acceleration)
        self.uncompensated_throttle = base
        self.throttle = throttle
        self.gimbal_angle_rad = gimbal
        self.tilt_limited = not np.isclose(target_attitude, raw_target_attitude, atol=1e-12)
        self.gimbal_limited = gimbal_limited
        self.throttle_limited = not np.isclose(
            throttle,
            base / max(float(np.cos(current.theta + gimbal)), 1e-9)
            if self.compensate_vertical_thrust and base > 0.0
            else base,
            atol=1e-12,
        )
        return np.asarray((throttle, gimbal), dtype=np.float64)


@dataclass(slots=True)
class IntegratedLandingController:
    """Combine phase-specific vertical and horizontal controllers at a fixed control rate."""

    parameters: PlanarDynamicsParameters
    ground_z_m: float
    simulation_interval_s: float
    control_interval_s: float
    terminal_phase_below_m: float
    approach_vertical: VerticalVelocityPIDController
    terminal_vertical: VerticalVelocityPIDController
    approach_horizontal: HorizontalAttitudeController
    terminal_horizontal: HorizontalAttitudeController
    throttle_slew_rate_per_s: float
    gimbal_slew_rate_rad_s: float
    held_action: NDArray[np.float64] | None = field(init=False, default=None)
    raw_action: NDArray[np.float64] | None = field(init=False, default=None)
    phase: Literal["approach", "terminal"] | None = field(init=False, default=None)
    next_update_time_s: float = field(init=False, default=0.0)
    last_update_time_s: float | None = field(init=False, default=None)
    last_command_time_s: float | None = field(init=False, default=None)
    update_count: int = field(init=False, default=0)
    updated_this_step: bool = field(init=False, default=False)
    throttle_slew_limited: bool = field(init=False, default=False)
    gimbal_slew_limited: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.ground_z_m = float(self.ground_z_m)
        if not np.isfinite(self.ground_z_m):
            raise ValueError("ground_z_m must be finite")
        self.simulation_interval_s = _positive(
            self.simulation_interval_s,
            "simulation_interval_s",
        )
        self.control_interval_s = _positive(self.control_interval_s, "control_interval_s")
        ratio = self.control_interval_s / self.simulation_interval_s
        if ratio < 1.0 or not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
            raise ValueError(
                "control_interval_s must be an integer multiple of simulation_interval_s"
            )
        self.terminal_phase_below_m = _positive(
            self.terminal_phase_below_m,
            "terminal_phase_below_m",
        )
        self.throttle_slew_rate_per_s = _positive(
            self.throttle_slew_rate_per_s,
            "throttle_slew_rate_per_s",
        )
        self.gimbal_slew_rate_rad_s = _positive(
            self.gimbal_slew_rate_rad_s,
            "gimbal_slew_rate_rad_s",
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IntegratedLandingController:
        """Build both control phases from the shared project configuration."""
        parameters = PlanarDynamicsParameters.from_config(config)
        settings = config["integrated_landing_controller"]
        control_interval = float(settings["control_interval_s"])
        integral_limit = float(settings["anti_windup_integral_limit_m"])
        target_x = float(settings["target_x_m"])
        compensate = bool(settings["compensate_vertical_thrust"])

        def vertical_controller(phase: dict[str, Any]) -> VerticalVelocityPIDController:
            profile = phase["vertical_profile"]
            gains = phase["vertical_gains"]
            return VerticalVelocityPIDController(
                parameters=parameters,
                ground_z_m=float(config["simulation"]["ground_z_m"]),
                control_interval_s=control_interval,
                profile=VerticalVelocityProfile(
                    touchdown_speed_m_s=float(profile["touchdown_speed_m_s"]),
                    max_descent_speed_m_s=float(profile["max_descent_speed_m_s"]),
                    deceleration_m_s2=float(profile["deceleration_m_s2"]),
                ),
                kp=float(gains["kp"]),
                ki=float(gains["ki"]),
                kd=float(gains["kd"]),
                integral_limit_m=integral_limit,
            )

        def horizontal_controller(phase: dict[str, Any]) -> HorizontalAttitudeController:
            outer = phase["horizontal_outer_loop"]
            inner = phase["attitude_inner_loop"]
            return HorizontalAttitudeController(
                parameters=parameters,
                target_x_m=target_x,
                position_kp_s2=float(outer["position_kp_s2"]),
                velocity_kd_s=float(outer["velocity_kd_s"]),
                max_horizontal_acceleration_m_s2=float(outer["max_horizontal_acceleration_m_s2"]),
                max_tilt_rad=float(np.deg2rad(outer["max_tilt_deg"])),
                attitude_kp_s2=float(inner["attitude_kp_s2"]),
                angular_rate_kd_s=float(inner["angular_rate_kd_s"]),
                compensate_vertical_thrust=compensate,
            )

        phases = settings["phases"]
        approach = phases["approach"]
        terminal = phases["terminal"]
        return cls(
            parameters=parameters,
            ground_z_m=float(config["simulation"]["ground_z_m"]),
            simulation_interval_s=float(config["simulation"]["dt_s"]),
            control_interval_s=control_interval,
            terminal_phase_below_m=float(settings["terminal_phase_below_m"]),
            approach_vertical=vertical_controller(approach),
            terminal_vertical=vertical_controller(terminal),
            approach_horizontal=horizontal_controller(approach),
            terminal_horizontal=horizontal_controller(terminal),
            throttle_slew_rate_per_s=float(settings["slew_rates"]["throttle_per_s"]),
            gimbal_slew_rate_rad_s=float(np.deg2rad(settings["slew_rates"]["gimbal_deg_s"])),
        )

    def reset(self) -> None:
        """Clear both control phases, the held command, and timing state."""
        self.approach_vertical.reset()
        self.terminal_vertical.reset()
        self.approach_horizontal.reset()
        self.terminal_horizontal.reset()
        self.held_action = None
        self.raw_action = None
        self.phase = None
        self.next_update_time_s = 0.0
        self.last_update_time_s = None
        self.last_command_time_s = None
        self.update_count = 0
        self.updated_this_step = False
        self.throttle_slew_limited = False
        self.gimbal_slew_limited = False

    def _phase_controllers(
        self,
        altitude_m: float,
    ) -> tuple[
        Literal["approach", "terminal"],
        VerticalVelocityPIDController,
        HorizontalAttitudeController,
    ]:
        if altitude_m <= self.terminal_phase_below_m:
            return "terminal", self.terminal_vertical, self.terminal_horizontal
        return "approach", self.approach_vertical, self.approach_horizontal

    def _apply_slew_limits(
        self,
        raw_action: NDArray[np.float64],
        elapsed_s: float,
    ) -> NDArray[np.float64]:
        if self.held_action is None:
            self.throttle_slew_limited = False
            self.gimbal_slew_limited = False
            return raw_action.copy()
        limits = np.asarray(
            (
                self.throttle_slew_rate_per_s * elapsed_s,
                self.gimbal_slew_rate_rad_s * elapsed_s,
            )
        )
        lower = self.held_action - limits
        upper = self.held_action + limits
        limited = np.clip(raw_action, lower, upper)
        self.throttle_slew_limited = not np.isclose(limited[0], raw_action[0], atol=1e-12)
        self.gimbal_slew_limited = not np.isclose(limited[1], raw_action[1], atol=1e-12)
        return limited

    def command(
        self,
        state: State | ArrayLike,
        *,
        time_s: float,
    ) -> NDArray[np.float64]:
        """Update on the control clock and hold the command on intermediate simulation steps."""
        current = state if isinstance(state, State) else State.from_array(state)
        time = _nonnegative(time_s, "time_s")
        if self.last_command_time_s is not None and time < self.last_command_time_s - 1e-12:
            raise ValueError("time_s must be nondecreasing")
        self.last_command_time_s = time
        self.updated_this_step = False
        if self.held_action is not None and time < self.next_update_time_s - 1e-12:
            return self.held_action.copy()

        altitude = current.z - self.ground_z_m
        if altitude < 0.0:
            raise ValueError("state cannot be below ground")
        phase, vertical, horizontal = self._phase_controllers(altitude)
        base_throttle = float(vertical.command(current)[0])
        raw_action = horizontal.command(current, base_throttle=base_throttle)
        elapsed = (
            self.control_interval_s
            if self.last_update_time_s is None
            else time - self.last_update_time_s
        )
        action = self._apply_slew_limits(raw_action, elapsed)

        self.raw_action = raw_action.copy()
        self.held_action = action.copy()
        self.phase = phase
        self.last_update_time_s = time
        while self.next_update_time_s <= time + 1e-12:
            self.next_update_time_s += self.control_interval_s
        self.update_count += 1
        self.updated_this_step = True
        return action


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
