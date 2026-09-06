"""YAML configuration loading and Day 1 schema validation."""

from __future__ import annotations

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
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive number")
    return float(value)


def validate_config(config: Config) -> None:
    """Validate the stable Day 1 configuration contract."""
    project = _mapping(config, "project")
    conventions = _mapping(config, "conventions")
    simulation = _mapping(config, "simulation")
    vehicle = _mapping(config, "vehicle")
    initial_state = _mapping(config, "initial_state")
    landing = _mapping(config, "landing_success")

    if str(project.get("python_version")) != "3.12.7":
        raise ConfigError("project.python_version must be '3.12.7'")

    expected_conventions = {
        "x_positive": "right",
        "z_positive": "up",
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
    if max_time <= dt:
        raise ConfigError("simulation.max_time_s must be greater than simulation.dt_s")

    dry_mass = _positive(vehicle, "dry_mass_kg")
    initial_mass = _positive(vehicle, "initial_mass_kg")
    _positive(vehicle, "max_thrust_n")
    _positive(vehicle, "specific_impulse_s")
    _positive(vehicle, "standard_gravity_m_s2")
    _positive(vehicle, "moment_of_inertia_kg_m2")
    _positive(vehicle, "engine_lever_arm_m")
    _positive(vehicle, "gimbal_limit_deg")
    if initial_mass < dry_mass:
        raise ConfigError("vehicle.initial_mass_kg must be at or above dry_mass_kg")

    throttle_min = vehicle.get("throttle_min")
    throttle_max = vehicle.get("throttle_max")
    if not isinstance(throttle_min, int | float) or not isinstance(throttle_max, int | float):
        raise ConfigError("vehicle throttle bounds must be numeric")
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
    if initial_state["mass_kg"] < dry_mass:
        raise ConfigError("initial_state.mass_kg must be at or above vehicle.dry_mass_kg")

    for key in (
        "max_abs_x_m",
        "max_abs_vx_m_s",
        "max_abs_vz_m_s",
        "max_abs_theta_deg",
        "max_abs_omega_deg_s",
    ):
        _positive(landing, key)


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration and validate it before returning it."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded
