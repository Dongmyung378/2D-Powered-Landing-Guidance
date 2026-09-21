"""Package metadata, state vectors and configuration checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import powered_landing_guidance as plg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def test_package_import_and_metadata() -> None:
    assert plg.__version__ == "0.1.0"
    assert plg.STATE_NAMES == ("x", "z", "vx", "vz", "theta", "omega", "mass")
    assert plg.STATE_UNITS == ("m", "m", "m/s", "m/s", "rad", "rad/s", "kg")
    assert plg.ACTION_NAMES == ("throttle", "gimbal_angle")
    assert plg.ACTION_UNITS == ("1", "rad")


def test_state_and_control_vector_order() -> None:
    state = plg.State.from_array([1.0, 100.0, 2.0, -10.0, 0.1, -0.2, 900.0])
    control = plg.Control.from_array([0.75, 0.05])

    np.testing.assert_allclose(state.as_array(), [1.0, 100.0, 2.0, -10.0, 0.1, -0.2, 900.0])
    np.testing.assert_allclose(control.as_array(), [0.75, 0.05])


@pytest.mark.parametrize("throttle", [-0.01, 1.01])
def test_control_rejects_out_of_range_throttle(throttle: float) -> None:
    with pytest.raises(ValueError, match="throttle"):
        plg.Control(throttle=throttle, gimbal_angle=0.0)


def test_default_config_loads_and_uses_python_3127() -> None:
    config = plg.load_config(DEFAULT_CONFIG)

    assert str(config["project"]["python_version"]) == "3.12.7"
    assert config["conventions"]["z_positive"] == "up"
    assert config["landing_success"]["max_abs_x_m"] == 1.0
    assert config["landing_success"]["max_abs_vz_m_s"] == 2.0
    assert config["vertical_velocity_controller"]["gains"] == {
        "kp": 0.08,
        "ki": 0.001,
        "kd": 0.01,
    }
    assert config["horizontal_attitude_controller"]["outer_loop"]["max_tilt_deg"] == 15.0
    assert config["integrated_landing_controller"]["control_interval_s"] == 0.1
    assert config["baseline_tuning"]["search"]["method"] == "grid"
    assert (
        config["baseline_tuning"]["search"]["train_seed"]
        != config["baseline_tuning"]["search"]["validation_seed"]
    )


def test_frozen_pid_baseline_protocol_loads() -> None:
    config = plg.load_config(ROOT / "configs/pid-baseline-v1.yaml")
    protocol = config["baseline_protocol"]

    assert protocol["id"] == "pid-baseline-v1"
    assert protocol["controller"] == "integrated-pid"
    assert protocol["frozen"] is True
    assert protocol["initial_conditions"]["episodes"] == 1000
    assert protocol["acceptance"]["minimum_success_rate"] == 0.70


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("intervals",), 0, "intervals"),
        (("duration_s",), [5.0, 5.0], "duration_s"),
        (("max_abs_thrust_angle_deg",), 0.0, "max_abs_thrust_angle_deg"),
        (("max_abs_thrust_angle_deg",), 60.0, "max_abs_thrust_angle_deg"),
        (("max_abs_tilt_deg",), 0.0, "max_abs_tilt_deg"),
        (("max_abs_tilt_deg",), 91.0, "max_abs_tilt_deg"),
        (("max_abs_tilt_deg",), 4.0, "landing attitude"),
        (("max_abs_angular_rate_deg_s",), 4.0, "angular-rate limit"),
        (("min_propellant_reserve_kg",), 250.0, "reserve"),
        (("target_touchdown_vz_m_s",), 0.0, "target touchdown vz"),
        (("feasibility_tolerance",), 1.0, "feasibility_tolerance"),
        (("objective_weights", "fuel"), 0.0, "fuel"),
    ],
)
def test_config_rejects_invalid_optimal_control_settings(path, value, message) -> None:
    config = deepcopy(plg.load_config(ROOT / "configs/pid-baseline-v1.yaml"))
    target = config["optimal_control"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("mesh_intervals",), [25, 25, 50], "mesh_intervals"),
        (("replay_dt_s",), 0.02, "replay_dt_s"),
        (("final_state_tolerances", "x_m"), 0.0, "x_m"),
        (
            ("objective_profiles", "baseline", "fuel"),
            2.0,
            "baseline weights",
        ),
        (
            ("objective_profiles", "smooth_control", "smoothness"),
            0.001,
            "smoothness weights",
        ),
    ],
)
def test_config_rejects_invalid_optimal_control_study(path, value, message) -> None:
    config = deepcopy(plg.load_config(ROOT / "configs/pid-baseline-v1.yaml"))
    target = config["optimal_control"]["study"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("episodes", 0, "episodes"),
        ("minimum_cases", 101, "minimum_cases"),
        ("runner", "parallel", "runner"),
        ("attempt_timeout_s", 0.0, "attempt_timeout_s"),
        ("max_retries", -1, "max_retries"),
        ("warm_start", "none", "warm_start"),
        ("replay_dt_s", 0.02, "replay_dt_s"),
    ],
)
def test_config_rejects_invalid_teacher_pipeline(key, value, message) -> None:
    config = deepcopy(plg.load_config(ROOT / "configs/pid-baseline-v1.yaml"))
    config["teacher_pipeline"][key] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


def test_config_rejects_initial_mass_below_dry_mass() -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config["initial_state"]["mass_kg"] = config["vehicle"]["dry_mass_kg"] - 1.0

    with pytest.raises(plg.ConfigError, match="initial_state.mass_kg"):
        plg.validate_config(config)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 0, -1])
def test_config_rejects_invalid_positive_parameters(value) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config["simulation"]["dt_s"] = value
    with pytest.raises(plg.ConfigError, match="dt_s"):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("simulation", "ground_z_m", np.nan),
        ("initial_state", "x_m", np.inf),
        ("initial_state", "mass_kg", True),
        ("vehicle", "throttle_min", False),
    ],
)
def test_config_rejects_non_finite_or_boolean_numbers(section, key, value) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config[section][key] = value
    with pytest.raises(plg.ConfigError, match=key):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("simulation", "integrator", "unknown", "integrator"),
        ("initial_state", "z_m", -1.0, "z_m"),
        ("vehicle", "gimbal_limit_deg", 91.0, "gimbal_limit_deg"),
        (
            "conventions",
            "theta_zero",
            "positive_x",
            "theta_zero",
        ),
        (
            "landing_success",
            "evaluate_immediately_before_ground_contact",
            False,
            "evaluate_immediately_before_ground_contact",
        ),
    ],
)
def test_config_rejects_inconsistent_model_contract(section, key, value, message) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config[section][key] = value
    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("profile", "deceleration_m_s2", 0.0, "deceleration_m_s2"),
        ("gains", "kp", -0.01, "kp"),
        ("anti_windup", "integral_limit_m", 0.0, "integral_limit_m"),
        ("evaluation", "vertical_speed_m_s", [-20.0, 1.0], "vertical_speed_m_s"),
        ("gain_sweep", "kd", [], "gain_sweep.kd"),
    ],
)
def test_config_rejects_invalid_vertical_controller_settings(section, key, value, message) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config["vertical_velocity_controller"][section][key] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("outer_loop", "max_tilt_deg", 46.0, "max_tilt_deg"),
        ("inner_loop", "attitude_kp_s2", 0.0, "attitude_kp_s2"),
        ("coupling", "compensate_vertical_thrust", 1, "compensate_vertical_thrust"),
        ("evaluation", "x_m", [-4.0, 20.0], "x_m"),
        ("evaluation", "theta_deg", [-16.0, 5.0], "theta_deg"),
    ],
)
def test_config_rejects_invalid_horizontal_controller_settings(
    section,
    key,
    value,
    message,
) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config["horizontal_attitude_controller"][section][key] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("control_interval_s",), 0.0, "control_interval_s"),
        (("compensate_vertical_thrust",), 1, "compensate_vertical_thrust"),
        (("slew_rates", "throttle_per_s"), 0.0, "throttle_per_s"),
        (
            ("phases", "terminal", "horizontal_outer_loop", "max_tilt_deg"),
            46.0,
            "max_tilt_deg",
        ),
        (("evaluation", "vz_m_s"), [-5.0, 1.0], "vz_m_s"),
    ],
)
def test_config_rejects_invalid_integrated_controller_settings(path, value, message) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    target = config["integrated_landing_controller"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


def test_integrated_controller_rejects_incompatible_control_interval() -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    config["integrated_landing_controller"]["control_interval_s"] = 0.03

    with pytest.raises(ValueError, match="integer multiple"):
        plg.IntegratedLandingController.from_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("search", "method"), "manual", "method"),
        (("search", "train_episodes"), 0, "train_episodes"),
        (("search", "grid", "position_gain_scale"), [], "position_gain_scale"),
        (("search", "random", "velocity_gain_scale"), [0.0, 1.0], "velocity_gain_scale"),
        (("objective", "failure_penalty"), 0.0, "failure_penalty"),
    ],
)
def test_config_rejects_invalid_baseline_tuning_settings(path, value, message) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    target = config["baseline_tuning"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


def test_config_rejects_tuning_data_leakage_and_empty_secondary_objective() -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    search = config["baseline_tuning"]["search"]
    search["validation_seed"] = search["train_seed"]
    with pytest.raises(plg.ConfigError, match="seeds must differ"):
        plg.validate_config(config)

    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    objective = config["baseline_tuning"]["objective"]
    objective["fuel_weight"] = 0.0
    objective["landing_error_weight"] = 0.0
    with pytest.raises(plg.ConfigError, match="must weight"):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("episodes_per_level",), 0, "episodes_per_level"),
        (("collapse_success_rate",), 1.1, "collapse_success_rate"),
        (("constant_wind_m_s",), [1.0, 2.0], "constant_wind_m_s"),
        (("gust", "amplitude_m_s"), [0.0, 0.0], "amplitude_m_s"),
        (("sensor_noise", "scales"), [0.0, -1.0], "scales"),
        (("thrust_scale",), [1.0, 1.0], "thrust_scale"),
        (("engine_lag_s",), [0.1, 0.2], "engine_lag_s"),
    ],
)
def test_config_rejects_invalid_disturbance_settings(path, value, message) -> None:
    config = deepcopy(plg.load_config(DEFAULT_CONFIG))
    target = config["disturbance_evaluation"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("frozen",), False, "frozen"),
        (("controller_sha256",), "not-a-hash", "controller_sha256"),
        (("initial_conditions", "episodes"), 0, "episodes"),
        (("initial_conditions", "sampler"), "changed", "sampler"),
        (("acceptance", "minimum_success_rate"), 1.1, "minimum_success_rate"),
        (
            ("representative_cases", "success_initial_condition_index"),
            1000,
            "case index",
        ),
        (("representative_cases", "failure", "horizontal_wind_m_s"), 0.0, "failure wind"),
    ],
)
def test_config_rejects_invalid_frozen_baseline_protocol(path, value, message) -> None:
    config = deepcopy(plg.load_config(ROOT / "configs/pid-baseline-v1.yaml"))
    target = config["baseline_protocol"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(plg.ConfigError, match=message):
        plg.validate_config(config)


def test_week1_sanity_and_termination_verification() -> None:
    from scripts.verify_week1 import verify_model_sanity, verify_termination_cases

    config = plg.load_config(DEFAULT_CONFIG)
    sanity = verify_model_sanity(config)
    termination = verify_termination_cases(config)

    assert len(sanity) == 5
    assert set(termination) == {"success", "hard_landing", "crash", "fuel_depletion", "timeout"}


def test_random_action_verification_is_reproducible() -> None:
    from scripts.verify_week1 import run_random_action_check

    config = plg.load_config(DEFAULT_CONFIG)
    first = run_random_action_check(config, episodes=3, seed=20260907)
    second = run_random_action_check(config, episodes=3, seed=20260907)

    assert first == second
    assert first.episodes == 3
    assert first.steps > 0
