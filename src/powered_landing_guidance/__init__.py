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
    frozen_teacher_initial_states,
    initial_condition_sha256,
    integrated_controller_sha256,
    sample_integrated_initial_states,
    teacher_configuration_sha256,
)
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    Control,
    State,
)
from powered_landing_guidance.offline_dataset import (
    build_dataset_design,
    compute_training_normalization_statistics,
    dataset_configuration_sha256,
    harder_test_analysis,
    initial_condition_id,
    sample_split_initial_states,
    validate_split_integrity,
)
from powered_landing_guidance.offline_dataset_finalization import (
    FinalDatasetSettings,
    analyze_coverage,
    compute_shard_normalization,
    merge_packed_shards,
    sample_hard_training_states,
)
from powered_landing_guidance.offline_dataset_generation import (
    OfflineGenerationSettings,
    build_offline_generation_report,
    generate_easy_teacher_trajectories,
    generate_teacher_trajectories,
    packed_shard_sha256,
    validate_packed_trajectory_shard,
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
    "FinalDatasetSettings",
    "HorizontalAttitudeController",
    "IntegratedLandingController",
    "INITIAL_CONDITION_SAMPLER",
    "LandingOptimalControlProblem",
    "ObjectiveTerms",
    "OfflineGenerationSettings",
    "ObjectiveScales",
    "State",
    "SuicideBurnController",
    "SuicideBurnEstimate",
    "VerticalVelocityPIDController",
    "VerticalVelocityProfile",
    "classify_ignition_timing",
    "build_dataset_design",
    "build_offline_generation_report",
    "analyze_coverage",
    "compute_training_normalization_statistics",
    "compute_shard_normalization",
    "dataset_configuration_sha256",
    "estimate_suicide_burn",
    "estimate_variable_mass_suicide_burn",
    "frozen_baseline_initial_states",
    "frozen_teacher_initial_states",
    "generate_easy_teacher_trajectories",
    "generate_teacher_trajectories",
    "harder_test_analysis",
    "initial_condition_sha256",
    "initial_condition_id",
    "integrated_controller_sha256",
    "load_config",
    "merge_packed_shards",
    "packed_shard_sha256",
    "sample_integrated_initial_states",
    "sample_hard_training_states",
    "sample_split_initial_states",
    "teacher_configuration_sha256",
    "validate_split_integrity",
    "validate_config",
    "validate_packed_trajectory_shard",
]

__version__ = "0.1.0"
