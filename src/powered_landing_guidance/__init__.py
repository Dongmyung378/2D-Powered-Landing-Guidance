"""Powered-landing guidance research package."""

from powered_landing_guidance.config import ConfigError, load_config, validate_config
from powered_landing_guidance.controllers import (
    HorizontalAttitudeController,
    SuicideBurnController,
    SuicideBurnEstimate,
    VerticalVelocityPIDController,
    VerticalVelocityProfile,
    classify_ignition_timing,
    estimate_suicide_burn,
    estimate_variable_mass_suicide_burn,
)
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    Control,
    State,
)

__all__ = [
    "ACTION_NAMES",
    "ACTION_UNITS",
    "STATE_NAMES",
    "STATE_UNITS",
    "ConfigError",
    "Control",
    "HorizontalAttitudeController",
    "State",
    "SuicideBurnController",
    "SuicideBurnEstimate",
    "VerticalVelocityPIDController",
    "VerticalVelocityProfile",
    "classify_ignition_timing",
    "estimate_suicide_burn",
    "estimate_variable_mass_suicide_burn",
    "load_config",
    "validate_config",
]

__version__ = "0.1.0"
