"""Powered-landing guidance research package."""

from powered_landing_guidance.config import ConfigError, load_config, validate_config
from powered_landing_guidance.controllers import (
    HorizontalAttitudeController,
    IntegratedLandingController,
    SuicideBurnController,
    SuicideBurnEstimate,
    VerticalVelocityPIDController,
    VerticalVelocityProfile,
    classify_ignition_timing,
    estimate_suicide_burn,
    estimate_variable_mass_suicide_burn,
)
from powered_landing_guidance.evaluation import (
    INITIAL_CONDITION_SAMPLER,
    frozen_baseline_initial_states,
    initial_condition_sha256,
    integrated_controller_sha256,
    sample_integrated_initial_states,
)
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    Control,
    State,
)
from powered_landing_guidance.optimal_control import (
    LandingOptimalControlProblem,
    ObjectiveScales,
    ObjectiveTerms,
)

__all__ = [
    "ACTION_NAMES",
    "ACTION_UNITS",
    "STATE_NAMES",
    "STATE_UNITS",
    "ConfigError",
    "Control",
    "HorizontalAttitudeController",
    "IntegratedLandingController",
    "INITIAL_CONDITION_SAMPLER",
    "LandingOptimalControlProblem",
    "ObjectiveTerms",
    "ObjectiveScales",
    "State",
    "SuicideBurnController",
    "SuicideBurnEstimate",
    "VerticalVelocityPIDController",
    "VerticalVelocityProfile",
    "classify_ignition_timing",
    "estimate_suicide_burn",
    "estimate_variable_mass_suicide_burn",
    "frozen_baseline_initial_states",
    "initial_condition_sha256",
    "integrated_controller_sha256",
    "load_config",
    "sample_integrated_initial_states",
    "validate_config",
]

__version__ = "0.1.0"
