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


def _positive_integer(mapping: Config, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer")
    return value


def _nonnegative_integer(mapping: Config, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"'{key}' must be a nonnegative integer")
    return value


def _sha256(mapping: Config, key: str) -> str:
    value = mapping.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfigError(f"'{key}' must be a lowercase SHA-256 hex digest")
    return value


def _positive_list(mapping: Config, key: str) -> list[float]:
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        raise ConfigError(f"'{key}' must be a nonempty list")
    result: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value <= 0.0
        ):
            raise ConfigError(f"'{key}' must contain positive finite numbers")
        result.append(float(value))
    return result


def _nonnegative_list(mapping: Config, key: str) -> list[float]:
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        raise ConfigError(f"'{key}' must be a nonempty list")
    result: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0.0
        ):
            raise ConfigError(f"'{key}' must contain nonnegative finite numbers")
        result.append(float(value))
    return result


def validate_config(config: Config) -> None:
    """Check model conventions, physical parameters and initial-state fields."""
    project = _mapping(config, "project")
    conventions = _mapping(config, "conventions")
    simulation = _mapping(config, "simulation")
    vehicle = _mapping(config, "vehicle")
    environment = _mapping(config, "environment")
    initial_state = _mapping(config, "initial_state")
    landing = _mapping(config, "landing_success")
    velocity_controller = _mapping(config, "vertical_velocity_controller")
    horizontal_controller = _mapping(config, "horizontal_attitude_controller")
    integrated_controller = _mapping(config, "integrated_landing_controller")
    baseline_tuning = _mapping(config, "baseline_tuning")

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
    _nonnegative(vehicle, "drag_coefficient")
    _positive(vehicle, "reference_area_m2")
    gimbal_limit = _positive(vehicle, "gimbal_limit_deg")
    if initial_mass < dry_mass:
        raise ConfigError("vehicle.initial_mass_kg must be at or above dry_mass_kg")
    if gimbal_limit > 90.0:
        raise ConfigError("vehicle.gimbal_limit_deg cannot exceed 90")

    throttle_min = _finite(vehicle, "throttle_min")
    throttle_max = _finite(vehicle, "throttle_max")
    if not 0 <= throttle_min < throttle_max <= 1:
        raise ConfigError("vehicle throttle bounds must satisfy 0 <= min < max <= 1")

    _positive(environment, "air_density_kg_m3")
    _finite(environment, "horizontal_wind_m_s")

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

    integrated_target_x = _finite(integrated_controller, "target_x_m")
    _positive(integrated_controller, "control_interval_s")
    _positive(integrated_controller, "terminal_phase_below_m")
    _positive(integrated_controller, "anti_windup_integral_limit_m")
    if not isinstance(integrated_controller.get("compensate_vertical_thrust"), bool):
        raise ConfigError("integrated compensate_vertical_thrust must be boolean")

    slew_rates = _mapping(integrated_controller, "slew_rates")
    _positive(slew_rates, "throttle_per_s")
    _positive(slew_rates, "gimbal_deg_s")

    phases = _mapping(integrated_controller, "phases")
    if set(phases) != {"approach", "terminal"}:
        raise ConfigError("integrated phases must contain exactly approach and terminal")
    phase_max_tilts: list[float] = []
    for phase_name in ("approach", "terminal"):
        phase = _mapping(phases, phase_name)
        phase_profile = _mapping(phase, "vertical_profile")
        phase_touchdown_speed = _positive(phase_profile, "touchdown_speed_m_s")
        phase_max_descent_speed = _positive(phase_profile, "max_descent_speed_m_s")
        _positive(phase_profile, "deceleration_m_s2")
        if phase_max_descent_speed < phase_touchdown_speed:
            raise ConfigError(
                f"integrated {phase_name} max_descent_speed_m_s must be at least "
                "touchdown_speed_m_s"
            )

        phase_vertical_gains = _mapping(phase, "vertical_gains")
        for key in ("kp", "ki", "kd"):
            _nonnegative(phase_vertical_gains, key)

        phase_outer = _mapping(phase, "horizontal_outer_loop")
        _nonnegative(phase_outer, "position_kp_s2")
        _nonnegative(phase_outer, "velocity_kd_s")
        _positive(phase_outer, "max_horizontal_acceleration_m_s2")
        phase_max_tilt = _positive(phase_outer, "max_tilt_deg")
        if phase_max_tilt > 45.0:
            raise ConfigError(f"integrated {phase_name} max_tilt_deg cannot exceed 45")
        phase_max_tilts.append(phase_max_tilt)

        phase_inner = _mapping(phase, "attitude_inner_loop")
        _positive(phase_inner, "attitude_kp_s2")
        _nonnegative(phase_inner, "angular_rate_kd_s")

    integrated_evaluation = _mapping(integrated_controller, "evaluation")
    integrated_min_abs_x = _positive(integrated_evaluation, "min_abs_x_m")
    integrated_x_range = _range(integrated_evaluation, "x_m")
    integrated_z_range = _range(integrated_evaluation, "z_m")
    _range(integrated_evaluation, "vx_m_s")
    integrated_vz_range = _range(integrated_evaluation, "vz_m_s")
    integrated_theta_range = _range(integrated_evaluation, "theta_deg")
    _range(integrated_evaluation, "omega_deg_s")
    integrated_mass_range = _range(integrated_evaluation, "mass_kg")
    if (
        integrated_x_range[0] > integrated_target_x - integrated_min_abs_x
        or integrated_x_range[1] < integrated_target_x + integrated_min_abs_x
    ):
        raise ConfigError("integrated evaluation x_m must cover both sides of min_abs_x_m")
    if integrated_z_range[0] <= ground:
        raise ConfigError("integrated evaluation z_m must be above ground")
    if integrated_vz_range[1] > 0.0:
        raise ConfigError("integrated evaluation vz_m_s must contain descending speeds")
    if max(abs(integrated_theta_range[0]), abs(integrated_theta_range[1])) > max(phase_max_tilts):
        raise ConfigError("integrated evaluation theta_deg exceeds the configured phase tilt limit")
    if integrated_mass_range[0] < dry_mass:
        raise ConfigError("integrated evaluation mass_kg cannot extend below dry mass")

    tuning_search = _mapping(baseline_tuning, "search")
    if tuning_search.get("method") not in {"grid", "random"}:
        raise ConfigError("baseline_tuning search method must be 'grid' or 'random'")
    _positive_integer(tuning_search, "train_episodes")
    _positive_integer(tuning_search, "validation_episodes")
    _positive_integer(tuning_search, "random_candidates")
    for key in ("train_seed", "validation_seed"):
        value = tuning_search.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"baseline_tuning '{key}' must be a nonnegative integer")
    if tuning_search["train_seed"] == tuning_search["validation_seed"]:
        raise ConfigError("baseline_tuning train and validation seeds must differ")

    tuning_grid = _mapping(tuning_search, "grid")
    tuning_random = _mapping(tuning_search, "random")
    scale_keys = (
        "position_gain_scale",
        "velocity_gain_scale",
        "descent_profile_scale",
    )
    for key in scale_keys:
        _positive_list(tuning_grid, key)
        random_range = _range(tuning_random, key)
        if random_range[0] <= 0.0:
            raise ConfigError(f"baseline_tuning random '{key}' must be positive")

    tuning_objective = _mapping(baseline_tuning, "objective")
    _positive(tuning_objective, "failure_penalty")
    _nonnegative(tuning_objective, "fuel_weight")
    _nonnegative(tuning_objective, "landing_error_weight")
    if tuning_objective["fuel_weight"] == 0.0 and tuning_objective["landing_error_weight"] == 0.0:
        raise ConfigError("baseline_tuning objective must weight fuel or landing error")

    disturbance_settings = config.get("disturbance_evaluation")
    if disturbance_settings is not None:
        if not isinstance(disturbance_settings, dict):
            raise ConfigError("'disturbance_evaluation' must be a mapping")
        _positive_integer(disturbance_settings, "episodes_per_level")
        disturbance_seed = disturbance_settings.get("seed")
        if (
            not isinstance(disturbance_seed, int)
            or isinstance(disturbance_seed, bool)
            or disturbance_seed < 0
        ):
            raise ConfigError("disturbance_evaluation seed must be a nonnegative integer")
        collapse_rate = _positive(disturbance_settings, "collapse_success_rate")
        if collapse_rate > 1.0:
            raise ConfigError("collapse_success_rate cannot exceed 1")

        controller_scales = _mapping(disturbance_settings, "controller_scales")
        for key in scale_keys:
            _positive(controller_scales, key)

        constant_wind = _nonnegative_list(disturbance_settings, "constant_wind_m_s")
        if (
            len(constant_wind) < 2
            or constant_wind[0] != 0.0
            or any(
                later <= earlier
                for earlier, later in zip(constant_wind, constant_wind[1:], strict=False)
            )
        ):
            raise ConfigError("constant_wind_m_s must start at 0 and increase")

        gust = _mapping(disturbance_settings, "gust")
        gust_amplitudes = _nonnegative_list(gust, "amplitude_m_s")
        if (
            len(gust_amplitudes) < 2
            or gust_amplitudes[0] != 0.0
            or any(
                later <= earlier
                for earlier, later in zip(gust_amplitudes, gust_amplitudes[1:], strict=False)
            )
        ):
            raise ConfigError("gust amplitude_m_s must start at 0 and increase")
        gust_start_range = _range(gust, "start_time_s")
        if gust_start_range[0] < 0.0:
            raise ConfigError("gust start_time_s must be nonnegative")
        _positive(gust, "duration_s")

        sensor_noise = _mapping(disturbance_settings, "sensor_noise")
        noise_scales = _nonnegative_list(sensor_noise, "scales")
        if (
            len(noise_scales) < 2
            or noise_scales[0] != 0.0
            or any(
                later <= earlier
                for earlier, later in zip(noise_scales, noise_scales[1:], strict=False)
            )
        ):
            raise ConfigError("sensor noise scales must start at 0 and increase")
        noise_standard_deviations = _mapping(sensor_noise, "standard_deviations")
        noise_keys = ("x_m", "z_m", "vx_m_s", "vz_m_s", "theta_deg", "omega_deg_s", "mass_kg")
        noise_values = [_nonnegative(noise_standard_deviations, key) for key in noise_keys]
        if not any(noise_values):
            raise ConfigError("sensor noise must have a positive standard deviation")

        thrust_scales = _positive_list(disturbance_settings, "thrust_scale")
        if (
            len(thrust_scales) < 2
            or thrust_scales[0] != 1.0
            or any(
                later >= earlier
                for earlier, later in zip(thrust_scales, thrust_scales[1:], strict=False)
            )
        ):
            raise ConfigError("thrust_scale must start at 1 and decrease")

        engine_lag = _nonnegative_list(disturbance_settings, "engine_lag_s")
        if (
            len(engine_lag) < 2
            or engine_lag[0] != 0.0
            or any(
                later <= earlier for earlier, later in zip(engine_lag, engine_lag[1:], strict=False)
            )
        ):
            raise ConfigError("engine_lag_s must start at 0 and increase")

    optimal = config.get("optimal_control")
    if optimal is not None:
        if not isinstance(optimal, dict):
            raise ConfigError("'optimal_control' must be a mapping")
        _positive_integer(optimal, "intervals")
        duration_low, duration_high = _range(optimal, "duration_s")
        if duration_low <= 0.0 or duration_low == duration_high:
            raise ConfigError("optimal_control.duration_s must have 0 < low < high")
        angle_limit = _positive(optimal, "max_abs_thrust_angle_deg")
        if angle_limit > 45.0:
            raise ConfigError("optimal_control.max_abs_thrust_angle_deg must not exceed 45")
        tilt_limit = _positive(optimal, "max_abs_tilt_deg")
        if tilt_limit > 90.0:
            raise ConfigError("optimal_control.max_abs_tilt_deg must not exceed 90")
        if tilt_limit < landing["max_abs_theta_deg"]:
            raise ConfigError("optimal_control max tilt must include the landing attitude limit")
        angular_rate_limit = _positive(optimal, "max_abs_angular_rate_deg_s")
        if angular_rate_limit < landing["max_abs_omega_deg_s"]:
            raise ConfigError(
                "optimal_control angular-rate limit must include the landing angular-rate limit"
            )
        reserve = _positive(optimal, "min_propellant_reserve_kg")
        if reserve >= initial_values["mass_kg"] - dry_mass:
            raise ConfigError("optimal_control reserve must be below initial propellant")
        target_vz = _finite(optimal, "target_touchdown_vz_m_s")
        if not -landing["max_abs_vz_m_s"] <= target_vz < 0.0:
            raise ConfigError("optimal_control target touchdown vz must be descending and safe")
        tolerance = _positive(optimal, "feasibility_tolerance")
        if tolerance >= 1.0:
            raise ConfigError("optimal_control feasibility_tolerance must be below 1")
        weights = _mapping(optimal, "objective_weights")
        if set(weights) != {"fuel", "touchdown", "smoothness"}:
            raise ConfigError(
                "optimal_control objective_weights keys must be fuel, touchdown, smoothness"
            )
        for key in ("fuel", "touchdown", "smoothness"):
            _positive(weights, key)

    baseline_protocol = config.get("baseline_protocol")
    if baseline_protocol is not None:
        if not isinstance(baseline_protocol, dict):
            raise ConfigError("'baseline_protocol' must be a mapping")
        protocol_id = baseline_protocol.get("id")
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ConfigError("baseline_protocol.id must be a nonempty string")
        if baseline_protocol.get("controller") != "integrated-pid":
            raise ConfigError("baseline_protocol.controller must be 'integrated-pid'")
        if baseline_protocol.get("frozen") is not True:
            raise ConfigError("baseline_protocol.frozen must be true")
        _sha256(baseline_protocol, "controller_sha256")

        fixed_conditions = _mapping(baseline_protocol, "initial_conditions")
        if fixed_conditions.get("sampler") != "integrated_uniform_v1":
            raise ConfigError("baseline initial-condition sampler must be 'integrated_uniform_v1'")
        fixed_episodes = _positive_integer(fixed_conditions, "episodes")
        _nonnegative_integer(fixed_conditions, "seed")
        _sha256(fixed_conditions, "sha256")

        acceptance = _mapping(baseline_protocol, "acceptance")
        minimum_success = _positive(acceptance, "minimum_success_rate")
        if minimum_success > 1.0:
            raise ConfigError("baseline minimum_success_rate cannot exceed 1")

        cases = _mapping(baseline_protocol, "representative_cases")
        success_index = _nonnegative_integer(cases, "success_initial_condition_index")
        failure = _mapping(cases, "failure")
        failure_index = _nonnegative_integer(failure, "initial_condition_index")
        failure_wind = _finite(failure, "horizontal_wind_m_s")
        if success_index >= fixed_episodes or failure_index >= fixed_episodes:
            raise ConfigError("representative case index must be within the fixed evaluation set")
        if failure_wind == 0.0:
            raise ConfigError("representative failure wind must be nonzero")

        provenance = _mapping(baseline_protocol, "provenance")
        if provenance.get("tuning_method") not in {"grid", "random"}:
            raise ConfigError("baseline provenance tuning_method must be 'grid' or 'random'")
        _nonnegative_integer(provenance, "train_seed")
        _nonnegative_integer(provenance, "validation_seed")
        for key in scale_keys:
            _positive(provenance, key)


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration and validate it before returning it."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded
