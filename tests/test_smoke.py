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
