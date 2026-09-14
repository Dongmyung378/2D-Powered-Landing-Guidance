"""YAML configuration loading and validation."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a project configuration violates the documented schema."""


def _mapping(parent: Config, key: str) -> Config:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _positive(mapping: Config, key: str) -> float:
    value = _finite(mapping, key)
    if value <= 0:
        raise ConfigError(f"'{key}' must be a positive number")
    return value


def _finite(mapping: Config, key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise ConfigError(f"'{key}' must be a finite number")
    return float(value)


def _nonnegative(mapping: Config, key: str) -> float:
    value = _finite(mapping, key)
    if value < 0:
        raise ConfigError(f"'{key}' must be a nonnegative number")
    return value


def _range(mapping: Config, key: str) -> tuple[float, float]:
    values = mapping.get(key)
    if not isinstance(values, list | tuple) or len(values) != 2:
        raise ConfigError(f"'{key}' must contain [low, high]")
    low, high = values
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value)
        for value in values
    ):
        raise ConfigError(f"'{key}' bounds must be finite numbers")
    if low > high:
        raise ConfigError(f"'{key}' must satisfy low <= high")
    return float(low), float(high)


def validate_config(config: Config) -> None:
    """Check model conventions, physical parameters and initial-state fields."""
    project = _mapping(config, "project")
    conventions = _mapping(config, "conventions")
    simulation = _mapping(config, "simulation")
    vehicle = _mapping(config, "vehicle")
    initial_state = _mapping(config, "initial_state")
    landing = _mapping(config, "landing_success")
    velocity_controller = _mapping(config, "vertical_velocity_controller")
    horizontal_controller = _mapping(config, "horizontal_attitude_controller")

    if str(project.get("python_version")) != "3.12.7":
        raise ConfigError("project.python_version must be '3.12.7'")

    expected_conventions = {
        "x_positive": "right",
        "z_positive": "up",
        "theta_zero": "body_axis_aligned_with_positive_z",
        "theta_positive": "clockwise_toward_positive_x",
        "gimbal_positive": "clockwise_thrust_vector_from_body_axis",
        "internal_angle_unit": "rad",
    }
    for key, expected in expected_conventions.items():
        if conventions.get(key) != expected:
            raise ConfigError(f"conventions.{key} must be '{expected}'")

    dt = _positive(simulation, "dt_s")
    max_time = _positive(simulation, "max_time_s")
    _positive(simulation, "gravity_m_s2")
    ground = _finite(simulation, "ground_z_m")
    if max_time <= dt:
        raise ConfigError("simulation.max_time_s must be greater than simulation.dt_s")
    if simulation.get("integrator") not in {"euler", "rk4"}:
        raise ConfigError("simulation.integrator must be 'euler' or 'rk4'")

    dry_mass = _positive(vehicle, "dry_mass_kg")
    initial_mass = _positive(vehicle, "initial_mass_kg")
    _positive(vehicle, "max_thrust_n")
    _positive(vehicle, "specific_impulse_s")
    _positive(vehicle, "standard_gravity_m_s2")
    _positive(vehicle, "moment_of_inertia_kg_m2")
    _positive(vehicle, "engine_lever_arm_m")
    gimbal_limit = _positive(vehicle, "gimbal_limit_deg")
    if initial_mass < dry_mass:
        raise ConfigError("vehicle.initial_mass_kg must be at or above dry_mass_kg")
    if gimbal_limit > 90.0:
        raise ConfigError("vehicle.gimbal_limit_deg cannot exceed 90")

    throttle_min = _finite(vehicle, "throttle_min")
    throttle_max = _finite(vehicle, "throttle_max")
    if not 0 <= throttle_min < throttle_max <= 1:
        raise ConfigError("vehicle throttle bounds must satisfy 0 <= min < max <= 1")

    required_initial_state = {
        "x_m",
        "z_m",
        "vx_m_s",
        "vz_m_s",
        "theta_deg",
        "omega_deg_s",
        "mass_kg",
    }
    if set(initial_state) != required_initial_state:
        missing = sorted(required_initial_state - set(initial_state))
        extra = sorted(set(initial_state) - required_initial_state)
        raise ConfigError(f"initial_state keys mismatch; missing={missing}, extra={extra}")
    initial_values = {key: _finite(initial_state, key) for key in required_initial_state}
    if initial_values["z_m"] < ground:
        raise ConfigError("initial_state.z_m must be at or above simulation.ground_z_m")
    if initial_values["mass_kg"] < dry_mass:
        raise ConfigError("initial_state.mass_kg must be at or above vehicle.dry_mass_kg")

    for key in (
        "max_abs_x_m",
        "max_abs_vx_m_s",
        "max_abs_vz_m_s",
        "max_abs_theta_deg",
        "max_abs_omega_deg_s",
    ):
        _positive(landing, key)
    for key in (
        "require_mass_at_or_above_dry_mass",
        "evaluate_immediately_before_ground_contact",
    ):
        if landing.get(key) is not True:
            raise ConfigError(f"landing_success.{key} must be true")

    profile = _mapping(velocity_controller, "profile")
    touchdown_speed = _positive(profile, "touchdown_speed_m_s")
    max_descent_speed = _positive(profile, "max_descent_speed_m_s")
    _positive(profile, "deceleration_m_s2")
    if max_descent_speed < touchdown_speed:
        raise ConfigError("profile.max_descent_speed_m_s must be at least touchdown_speed_m_s")

    gains = _mapping(velocity_controller, "gains")
    for key in ("kp", "ki", "kd"):
        _nonnegative(gains, key)

    anti_windup = _mapping(velocity_controller, "anti_windup")
    _positive(anti_windup, "integral_limit_m")

    evaluation = _mapping(velocity_controller, "evaluation")
    altitude_range = _range(evaluation, "altitude_m")
    vertical_speed_range = _range(evaluation, "vertical_speed_m_s")
    mass_range = _range(evaluation, "mass_kg")
    if altitude_range[0] < ground:
        raise ConfigError("evaluation.altitude_m cannot extend below ground")
    if vertical_speed_range[1] > 0:
        raise ConfigError("evaluation.vertical_speed_m_s must contain descending speeds")
    if mass_range[0] < dry_mass:
        raise ConfigError("evaluation.mass_kg cannot extend below dry mass")

    gain_sweep = _mapping(velocity_controller, "gain_sweep")
    for key in ("kp", "ki", "kd"):
        values = gain_sweep.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"gain_sweep.{key} must be a nonempty list")
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(value)
                or value < 0
            ):
                raise ConfigError(f"gain_sweep.{key} must contain nonnegative finite numbers")

    target_x = _finite(horizontal_controller, "target_x_m")
    outer_loop = _mapping(horizontal_controller, "outer_loop")
    _nonnegative(outer_loop, "position_kp_s2")
    _nonnegative(outer_loop, "velocity_kd_s")
    _positive(outer_loop, "max_horizontal_acceleration_m_s2")
    max_tilt_deg = _positive(outer_loop, "max_tilt_deg")
    if max_tilt_deg > 45.0:
        raise ConfigError("outer_loop.max_tilt_deg cannot exceed 45")

    inner_loop = _mapping(horizontal_controller, "inner_loop")
    _positive(inner_loop, "attitude_kp_s2")
    _nonnegative(inner_loop, "angular_rate_kd_s")

    coupling = _mapping(horizontal_controller, "coupling")
    if not isinstance(coupling.get("compensate_vertical_thrust"), bool):
        raise ConfigError("coupling.compensate_vertical_thrust must be boolean")

    horizontal_evaluation = _mapping(horizontal_controller, "evaluation")
    _positive(horizontal_evaluation, "duration_s")
    altitude = _finite(horizontal_evaluation, "altitude_m")
    min_abs_x = _positive(horizontal_evaluation, "min_abs_x_m")
    x_range = _range(horizontal_evaluation, "x_m")
    _range(horizontal_evaluation, "vx_m_s")
    theta_range = _range(horizontal_evaluation, "theta_deg")
    _range(horizontal_evaluation, "omega_deg_s")
    horizontal_mass_range = _range(horizontal_evaluation, "mass_kg")
    if altitude <= ground:
        raise ConfigError("horizontal evaluation altitude must be above ground")
    if x_range[0] > target_x - min_abs_x or x_range[1] < target_x + min_abs_x:
        raise ConfigError("horizontal evaluation x_m must cover both sides of min_abs_x_m")
    if max(abs(theta_range[0]), abs(theta_range[1])) > max_tilt_deg:
        raise ConfigError("horizontal evaluation theta_deg cannot exceed max_tilt_deg")
    if horizontal_mass_range[0] < dry_mass:
        raise ConfigError("horizontal evaluation mass_kg cannot extend below dry mass")


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration and validate it before returning it."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded
