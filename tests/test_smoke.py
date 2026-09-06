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
