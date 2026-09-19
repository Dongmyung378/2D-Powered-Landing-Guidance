"""Solver-independent nominal landing problem for the Week 3 teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.dynamics import PlanarDynamicsParameters
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.model import State

TERMINAL_RESIDUAL_NAMES = ("x", "z", "vx", "vz", "theta", "omega")
TERMINAL_HEIGHT_SCALE_M = 1.0
PATH_MARGIN_NAMES = (
    "altitude",
    "propellant_reserve",
    "throttle_low",
    "throttle_high",
    "gimbal_low",
    "gimbal_high",
    "tilt_low",
    "tilt_high",
    "angular_rate_low",
    "angular_rate_high",
)


def _vector(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have shape ({size},) and contain finite values")
    return vector


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Dimensionless fuel, terminal-tracking, smoothness, and weighted total."""

    fuel: float
    touchdown: float
    smoothness: float
    weighted_total: float


@dataclass(frozen=True, slots=True)
class LandingOptimalControlProblem:
    """Continuous model and constraints for a free-duration, fixed-node OCP.

    The model is smooth only in the feasible region: mass stays above dry mass
    by the configured reserve, so the simulator's fuel-cutoff branch is never
    used. A solver is deliberately not part of the Day 15 formulation.
    """

    initial_state: State
    parameters: PlanarDynamicsParameters
    ground_z_m: float
    target_x_m: float
    intervals: int
    duration_bounds_s: tuple[float, float]
    max_abs_thrust_angle_rad: float
    max_abs_tilt_rad: float
    max_abs_angular_rate_rad_s: float
    min_propellant_reserve_kg: float
    terminal_limits: NDArray[np.float64]
    target_touchdown_vz_m_s: float
    feasibility_tolerance: float
    fuel_weight: float
    touchdown_weight: float
    smoothness_weight: float

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        initial_state: State | ArrayLike,
    ) -> LandingOptimalControlProblem:
        """Construct a nominal problem using frozen simulator and landing limits."""
        validate_config(config)
        settings = config.get("optimal_control")
        if not isinstance(settings, dict):
            raise ValueError("configuration is missing optimal_control")
        state = (
            initial_state if isinstance(initial_state, State) else State.from_array(initial_state)
        )
        parameters = PlanarDynamicsParameters.from_config(config)
        ground = float(config["simulation"]["ground_z_m"])
        reserve = float(settings["min_propellant_reserve_kg"])
        if state.z <= ground:
            raise ValueError("initial state must be above the ground")
        if state.mass <= parameters.dry_mass_kg + reserve:
            raise ValueError("initial state must exceed dry mass plus the reserve")
        landing = config["landing_success"]
        limits = np.asarray(
            (
                landing["max_abs_x_m"],
                landing["max_abs_vx_m_s"],
                landing["max_abs_vz_m_s"],
                np.deg2rad(landing["max_abs_theta_deg"]),
                np.deg2rad(landing["max_abs_omega_deg_s"]),
            ),
            dtype=np.float64,
        )
        limits.flags.writeable = False
        weights = settings["objective_weights"]
        return cls(
            initial_state=state,
            parameters=parameters,
            ground_z_m=ground,
            target_x_m=float(config["integrated_landing_controller"]["target_x_m"]),
            intervals=int(settings["intervals"]),
            duration_bounds_s=tuple(float(value) for value in settings["duration_s"]),
            max_abs_thrust_angle_rad=float(np.deg2rad(settings["max_abs_thrust_angle_deg"])),
            max_abs_tilt_rad=float(np.deg2rad(settings["max_abs_tilt_deg"])),
            max_abs_angular_rate_rad_s=float(np.deg2rad(settings["max_abs_angular_rate_deg_s"])),
            min_propellant_reserve_kg=reserve,
            terminal_limits=limits,
            target_touchdown_vz_m_s=float(settings["target_touchdown_vz_m_s"]),
            feasibility_tolerance=float(settings["feasibility_tolerance"]),
            fuel_weight=float(weights["fuel"]),
            touchdown_weight=float(weights["touchdown"]),
            smoothness_weight=float(weights["smoothness"]),
        )

    def mesh_step_s(self, duration_s: float) -> float:
        """Return h = T/N after checking the free-horizon decision bound."""
        duration = float(duration_s)
        low, high = self.duration_bounds_s
        if not np.isfinite(duration) or not low <= duration <= high:
            raise ValueError(f"duration_s must be finite and within [{low}, {high}]")
        return duration / self.intervals

    def dynamics(self, state: ArrayLike, action: ArrayLike) -> NDArray[np.float64]:
        """Return the smooth nominal derivative in simulator state order."""
        x = _vector(state, 7, "state")
        u = _vector(action, 2, "action")
        if x[6] <= 0.0:
            raise ValueError("state mass must be positive")
        p = self.parameters
        thrust = p.max_thrust_n * u[0]
        direction = x[4] + u[1]
        return np.asarray(
            (
                x[2],
                x[3],
                thrust * np.sin(direction) / x[6],
                thrust * np.cos(direction) / x[6] - p.gravity_m_s2,
                x[5],
                -p.engine_lever_arm_m * thrust * np.sin(u[1]) / p.moment_of_inertia_kg_m2,
                -thrust / (p.specific_impulse_s * p.standard_gravity_m_s2),
            ),
            dtype=np.float64,
        )

    def path_margins(self, state: ArrayLike, action: ArrayLike) -> NDArray[np.float64]:
        """Return margins in PATH_MARGIN_NAMES order; every value must be >= 0."""
        x = _vector(state, 7, "state")
        u = _vector(action, 2, "action")
        p = self.parameters
        return np.asarray(
            (
                x[1] - self.ground_z_m,
                x[6] - p.dry_mass_kg - self.min_propellant_reserve_kg,
                u[0] - p.throttle_min,
                p.throttle_max - u[0],
                u[1] + p.gimbal_limit_rad,
                p.gimbal_limit_rad - u[1],
                x[4] + self.max_abs_tilt_rad,
                self.max_abs_tilt_rad - x[4],
                x[5] + self.max_abs_angular_rate_rad_s,
                self.max_abs_angular_rate_rad_s - x[5],
            ),
            dtype=np.float64,
        )

    def terminal_violations(self, state: ArrayLike) -> NDArray[np.float64]:
        """Return normalized nonnegative violations in TERMINAL_RESIDUAL_NAMES order."""
        x = _vector(state, 7, "state")
        x_limit, vx_limit, vz_limit, theta_limit, omega_limit = self.terminal_limits
        return np.asarray(
            (
                max(0.0, abs(x[0] - self.target_x_m) - x_limit) / x_limit,
                abs(x[1] - self.ground_z_m) / TERMINAL_HEIGHT_SCALE_M,
                max(0.0, abs(x[2]) - vx_limit) / vx_limit,
                max(0.0, -vz_limit - x[3], x[3]) / vz_limit,
                max(0.0, abs(x[4]) - theta_limit) / theta_limit,
                max(0.0, abs(x[5]) - omega_limit) / omega_limit,
            ),
            dtype=np.float64,
        )

    def feasibility_score(self, terminal_state: ArrayLike) -> float:
        """Stage A objective: minimize terminal violation before economic costs."""
        residuals = self.terminal_violations(terminal_state)
        return float(np.dot(residuals, residuals))

    def terminal_is_feasible(self, terminal_state: ArrayLike) -> bool:
        """Check Stage A's maximum normalized terminal-violation gate."""
        return bool(np.max(self.terminal_violations(terminal_state)) <= self.feasibility_tolerance)

    def objective(
        self,
        terminal_state: ArrayLike,
        controls: ArrayLike,
    ) -> ObjectiveTerms:
        """Stage B cost, meaningful only after all hard constraints are satisfied."""
        x = _vector(terminal_state, 7, "terminal_state")
        u = np.asarray(controls, dtype=np.float64)
        if u.shape != (self.intervals, 2) or not np.all(np.isfinite(u)):
            raise ValueError(f"controls must have shape ({self.intervals}, 2) and be finite")
        p = self.parameters
        available_fuel = self.initial_state.mass - p.dry_mass_kg
        fuel = float((self.initial_state.mass - x[6]) / available_fuel)
        x_limit, vx_limit, vz_limit, theta_limit, omega_limit = self.terminal_limits
        terminal_error = np.asarray(
            (
                (x[0] - self.target_x_m) / x_limit,
                x[2] / vx_limit,
                (x[3] - self.target_touchdown_vz_m_s) / vz_limit,
                x[4] / theta_limit,
                x[5] / omega_limit,
            ),
            dtype=np.float64,
        )
        touchdown = float(np.mean(terminal_error**2))
        if self.intervals > 1:
            changes = np.diff(u, axis=0)
            changes[:, 0] /= p.throttle_max - p.throttle_min
            changes[:, 1] /= 2.0 * p.gimbal_limit_rad
            smoothness = float(np.mean(np.sum(changes**2, axis=1)))
        else:
            smoothness = 0.0
        return ObjectiveTerms(
            fuel=fuel,
            touchdown=touchdown,
            smoothness=smoothness,
            weighted_total=(
                self.fuel_weight * fuel
                + self.touchdown_weight * touchdown
                + self.smoothness_weight * smoothness
            ),
        )

    def rk4_defects(
        self,
        states: ArrayLike,
        controls: ArrayLike,
        duration_s: float,
    ) -> NDArray[np.float64]:
        """Return X[k+1] - RK4(X[k], U[k], T/N), one row per interval."""
        x = np.asarray(states, dtype=np.float64)
        u = np.asarray(controls, dtype=np.float64)
        if x.shape != (self.intervals + 1, 7) or not np.all(np.isfinite(x)):
            raise ValueError(f"states must have shape ({self.intervals + 1}, 7) and be finite")
        if u.shape != (self.intervals, 2) or not np.all(np.isfinite(u)):
            raise ValueError(f"controls must have shape ({self.intervals}, 2) and be finite")
        h = self.mesh_step_s(duration_s)
        defects = np.empty((self.intervals, 7), dtype=np.float64)
        for k in range(self.intervals):
            action = u[k]
            predicted = rk4_step(
                lambda _t, value, command=action: self.dynamics(value, command),
                k * h,
                x[k],
                h,
            )
            defects[k] = x[k + 1] - predicted
        return defects
