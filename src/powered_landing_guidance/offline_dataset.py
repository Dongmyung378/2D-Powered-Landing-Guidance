"""Day 22 offline-dataset schema, split planning, and normalization contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.evaluation import initial_condition_sha256
from powered_landing_guidance.model import ACTION_NAMES, ACTION_UNITS, STATE_NAMES, STATE_UNITS

DATASET_SPLITS = ("train", "validation", "test")
INITIAL_CONDITION_SAMPLER = "signed_uniform_v1"
DATASET_RANGE_FIELDS = {
    "x_m",
    "min_abs_x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "theta_deg",
    "omega_deg_s",
    "mass_kg",
}
_CONFIG_DIGEST_PREFIX = b"powered-landing-guidance:offline-dataset-config:v1\n"
_INITIAL_CONDITION_ID_PREFIX = b"powered-landing-guidance:initial-condition:v1\n"


def dataset_configuration_sha256(config: dict[str, Any]) -> str:
    """Hash the complete Day 22 dataset contract independently of generated data."""
    payload = config.get("offline_dataset")
    if not isinstance(payload, dict):
        raise ValueError("configuration is missing offline_dataset")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(_CONFIG_DIGEST_PREFIX + encoded).hexdigest()


def initial_condition_id(state: NDArray[np.float64] | Sequence[float]) -> str:
    """Return a split-independent ID for one canonical seven-state initial condition."""
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("initial condition must contain seven finite values")
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(_INITIAL_CONDITION_ID_PREFIX + canonical.tobytes()).hexdigest()


def _split_settings(config: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in DATASET_SPLITS:
        raise ValueError(f"split must be one of {DATASET_SPLITS}")
    return config["offline_dataset"]["splits"][split]


def sample_split_initial_states(
    config: dict[str, Any],
    split: str,
) -> NDArray[np.float64]:
    """Sample the configured train, validation, or harder test initial conditions."""
    settings = _split_settings(config, split)
    if settings["sampler"] != INITIAL_CONDITION_SAMPLER:
        raise ValueError(f"unsupported initial-condition sampler: {settings['sampler']}")
    ranges = settings["ranges"]
    episodes = int(settings["episodes"])
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    x_low, x_high = (float(value) for value in ranges["x_m"])
    min_abs_x = float(ranges["min_abs_x_m"])
    rng = np.random.default_rng(int(settings["seed"]))
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=episodes)
    negative_x = rng.uniform(x_low, target_x - min_abs_x, size=episodes)
    positive_x = rng.uniform(target_x + min_abs_x, x_high, size=episodes)

    states = np.empty((episodes, 7), dtype=np.float64)
    states[:, 0] = np.where(signs < 0.0, negative_x, positive_x)
    states[:, 1] = rng.uniform(*ranges["z_m"], size=episodes)
    states[:, 2] = rng.uniform(*ranges["vx_m_s"], size=episodes)
    states[:, 3] = rng.uniform(*ranges["vz_m_s"], size=episodes)
    states[:, 4] = np.deg2rad(rng.uniform(*ranges["theta_deg"], size=episodes))
    states[:, 5] = np.deg2rad(rng.uniform(*ranges["omega_deg_s"], size=episodes))
    states[:, 6] = rng.uniform(*ranges["mass_kg"], size=episodes)
    states.flags.writeable = False
    return states


def split_initial_condition_ids(states: NDArray[np.float64]) -> NDArray[np.str_]:
    """Vectorize stable initial-condition IDs without creating object arrays."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
        raise ValueError("initial states must have shape (episodes, 7)")
    identifiers = np.asarray([initial_condition_id(state) for state in values], dtype="<U64")
    identifiers.flags.writeable = False
    return identifiers


def validate_split_integrity(
    identifiers: Mapping[str, NDArray[np.str_] | Sequence[str]],
) -> dict[str, Any]:
    """Reject duplicate initial conditions within or across trajectory-level splits."""
    if set(identifiers) != set(DATASET_SPLITS):
        raise ValueError("identifiers must contain train, validation, and test")
    sets: dict[str, set[str]] = {}
    duplicates_within: dict[str, int] = {}
    for split in DATASET_SPLITS:
        values = [str(value) for value in identifiers[split]]
        if not values:
            raise ValueError(f"{split} split cannot be empty")
        sets[split] = set(values)
        duplicates_within[split] = len(values) - len(sets[split])
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    passed = not any(duplicates_within.values()) and not any(overlaps.values())
    if not passed:
        raise ValueError(
            "trajectory split leakage detected: "
            f"duplicates={duplicates_within}, overlaps={overlaps}"
        )
    return {
        "split_unit": "trajectory",
        "duplicates_within_split": duplicates_within,
        "cross_split_initial_condition_overlaps": overlaps,
        "passed": True,
    }


def harder_test_analysis(config: dict[str, Any]) -> dict[str, Any]:
    """Explain the strict range separations that make test harder than train."""
    train = config["offline_dataset"]["splits"]["train"]["ranges"]
    test = config["offline_dataset"]["splits"]["test"]["ranges"]
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    train_max_abs_x = max(abs(float(value) - target_x) for value in train["x_m"])
    checks = {
        "horizontal_offset_is_disjoint_and_larger": (float(test["min_abs_x_m"]) > train_max_abs_x),
        "altitude_is_disjoint_and_higher": float(test["z_m"][0]) > float(train["z_m"][1]),
        "descent_speed_is_disjoint_and_faster": (
            float(test["vz_m_s"][1]) < float(train["vz_m_s"][0])
        ),
        "available_propellant_is_disjoint_and_lower": (
            float(test["mass_kg"][1]) < float(train["mass_kg"][0])
        ),
        "horizontal_velocity_envelope_is_wider": (
            float(test["vx_m_s"][0]) < float(train["vx_m_s"][0])
            and float(test["vx_m_s"][1]) > float(train["vx_m_s"][1])
        ),
        "attitude_envelope_is_wider": (
            float(test["theta_deg"][0]) < float(train["theta_deg"][0])
            and float(test["theta_deg"][1]) > float(train["theta_deg"][1])
        ),
        "angular_rate_envelope_is_wider": (
            float(test["omega_deg_s"][0]) < float(train["omega_deg_s"][0])
            and float(test["omega_deg_s"][1]) > float(train["omega_deg_s"][1])
        ),
    }
    return {
        "train_difficulty": config["offline_dataset"]["splits"]["train"]["difficulty"],
        "test_difficulty": config["offline_dataset"]["splits"]["test"]["difficulty"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def dataset_storage_schema() -> dict[str, Any]:
    """Describe the packed, variable-length trajectory shards used from Day 23 onward."""
    arrays = {
        "trajectory_ids": {"shape": ["trajectory_count"], "dtype": "<U64"},
        "initial_condition_ids": {"shape": ["trajectory_count"], "dtype": "<U64"},
        "source_case_indices": {"shape": ["trajectory_count"], "dtype": "int64"},
        "state_offsets": {"shape": ["trajectory_count + 1"], "dtype": "int64"},
        "action_offsets": {"shape": ["trajectory_count + 1"], "dtype": "int64"},
        "times_s": {"shape": ["sum(state_count)"], "dtype": "float64"},
        "states": {
            "shape": ["sum(state_count)", len(STATE_NAMES)],
            "dtype": "float64",
        },
        "actions": {
            "shape": ["sum(action_count)", len(ACTION_NAMES)],
            "dtype": "float64",
        },
        "durations_s": {"shape": ["trajectory_count"], "dtype": "float64"},
        "solver_success": {"shape": ["trajectory_count"], "dtype": "bool"},
        "stage_a_iterations": {"shape": ["trajectory_count"], "dtype": "int64"},
        "stage_b_iterations": {"shape": ["trajectory_count"], "dtype": "int64"},
        "max_hard_violation": {"shape": ["trajectory_count"], "dtype": "float64"},
        "max_terminal_violation": {
            "shape": ["trajectory_count"],
            "dtype": "float64",
        },
        "replay_passed": {"shape": ["trajectory_count"], "dtype": "bool"},
        "max_final_error_ratio": {
            "shape": ["trajectory_count"],
            "dtype": "float64",
        },
        "max_node_error_ratio": {
            "shape": ["trajectory_count"],
            "dtype": "float64",
        },
        "fuel_used_kg": {"shape": ["trajectory_count"], "dtype": "float64"},
        "attempt_count": {"shape": ["trajectory_count"], "dtype": "int64"},
    }
    return {
        "format": "packed_npz_v1",
        "one_file_per_split": True,
        "numeric_dtype": "float64",
        "training_dtype": "float32",
        "allow_pickle": False,
        "state": {
            "names": list(STATE_NAMES),
            "units": list(STATE_UNITS),
            "shape_per_node": [len(STATE_NAMES)],
        },
        "action": {
            "names": list(ACTION_NAMES),
            "units": list(ACTION_UNITS),
            "shape_per_interval": [len(ACTION_NAMES)],
        },
        "trajectory": {
            "state_action_alignment": (
                "states[state_start:state_end-1] -> actions[action_start:action_end]"
            ),
            "terminal_state_has_action": False,
            "variable_length": True,
            "split_field_location": "shard manifest; one shard contains one split only",
        },
        "initial_condition": {
            "order": list(STATE_NAMES),
            "id": "SHA-256 over canonical little-endian float64 state bytes",
            "cross_split_duplicates_allowed": False,
        },
        "solver_quality": {
            "required_fields": [
                "solver_success",
                "stage_a_iterations",
                "stage_b_iterations",
                "max_hard_violation",
                "max_terminal_violation",
                "replay_passed",
                "max_final_error_ratio",
                "max_node_error_ratio",
                "fuel_used_kg",
                "durations_s",
                "attempt_count",
            ],
            "failed_trajectories_in_training_shards": False,
            "failed_attempts_location": "manifest failure records",
        },
        "arrays": arrays,
    }


def _storage_schema_is_complete(schema: dict[str, Any]) -> bool:
    arrays = schema.get("arrays")
    solver_quality = schema.get("solver_quality")
    if not isinstance(arrays, dict) or not isinstance(solver_quality, dict):
        return False
    required_arrays = {
        "trajectory_ids",
        "initial_condition_ids",
        "source_case_indices",
        "state_offsets",
        "action_offsets",
        "times_s",
        "states",
        "actions",
        "durations_s",
        "solver_success",
        "stage_a_iterations",
        "stage_b_iterations",
        "max_hard_violation",
        "max_terminal_violation",
        "replay_passed",
        "max_final_error_ratio",
        "max_node_error_ratio",
        "fuel_used_kg",
        "attempt_count",
    }
    required_quality = solver_quality.get("required_fields")
    return (
        schema.get("format") == "packed_npz_v1"
        and schema.get("allow_pickle") is False
        and schema.get("state", {}).get("names") == list(STATE_NAMES)
        and schema.get("action", {}).get("names") == list(ACTION_NAMES)
        and schema.get("trajectory", {}).get("terminal_state_has_action") is False
        and schema.get("initial_condition", {}).get("cross_split_duplicates_allowed") is False
        and set(arrays) == required_arrays
        and isinstance(required_quality, list)
        and set(required_quality) <= set(arrays)
        and all(
            isinstance(details, dict)
            and set(details) == {"shape", "dtype"}
            and isinstance(details["shape"], list)
            and bool(details["shape"])
            and isinstance(details["dtype"], str)
            for details in arrays.values()
        )
    )


def normalization_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return the leakage-safe normalization rules and statistics schema."""
    settings = config["offline_dataset"]["normalization"]
    return {
        **settings,
        "fit_rows": "nonterminal state/action pairs from accepted train trajectories only",
        "validation_and_test_in_fit": False,
        "formula": "normalized = (value - mean) / scale",
        "scale": "maximum of population standard deviation and minimum_scale",
        "angle_unit": "rad",
        "accumulation_dtype": "float64",
        "inference": "normalize state, predict normalized action, denormalize, then clip",
        "statistics_fields": ["count", "mean", "standard_deviation", "scale", "min", "max"],
    }


def _feature_statistics(
    values: NDArray[np.float64],
    names: Sequence[str],
    minimum_scale: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(names) or len(array) == 0:
        raise ValueError(f"values must have shape (rows, {len(names)})")
    if not np.all(np.isfinite(array)):
        raise ValueError("normalization values must be finite")
    standard_deviation = np.std(array, axis=0, ddof=0)
    scale = np.maximum(standard_deviation, minimum_scale)
    return {
        "count": len(array),
        "names": list(names),
        "mean": np.mean(array, axis=0).tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "scale": scale.tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def compute_training_normalization_statistics(
    config: dict[str, Any],
    train_states: NDArray[np.float64],
    train_actions: NDArray[np.float64],
) -> dict[str, Any]:
    """Fit state and action statistics from aligned training transitions only."""
    settings = config["offline_dataset"]["normalization"]
    if settings["method"] != "standard_score" or settings["fit_split"] != "train":
        raise ValueError("normalization statistics must use standard_score fitted on train")
    states = np.asarray(train_states, dtype=np.float64)
    actions = np.asarray(train_actions, dtype=np.float64)
    if len(states) != len(actions):
        raise ValueError("train state and action rows must be aligned one-to-one")
    minimum_scale = float(settings["minimum_scale"])
    return {
        "schema_version": 1,
        "fit_split": "train",
        "terminal_states_included": False,
        "state": _feature_statistics(states, STATE_NAMES, minimum_scale),
        "action": _feature_statistics(actions, ACTION_NAMES, minimum_scale),
    }


def build_dataset_design(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Build the complete Day 22 report and deterministic initial-condition plan."""
    validate_config(config)
    split_states: dict[str, NDArray[np.float64]] = {}
    split_ids: dict[str, NDArray[np.str_]] = {}
    arrays: dict[str, NDArray[Any]] = {}
    split_report: dict[str, Any] = {}
    for split in DATASET_SPLITS:
        states = sample_split_initial_states(config, split)
        identifiers = split_initial_condition_ids(states)
        split_states[split] = states
        split_ids[split] = identifiers
        arrays[f"{split}_initial_states"] = states
        arrays[f"{split}_initial_condition_ids"] = identifiers
        settings = config["offline_dataset"]["splits"][split]
        split_report[split] = {
            "difficulty": settings["difficulty"],
            "sampler": settings["sampler"],
            "episodes": settings["episodes"],
            "seed": settings["seed"],
            "ranges": settings["ranges"],
            "initial_state_sha256": initial_condition_sha256(states),
        }

    integrity = validate_split_integrity(split_ids)
    difficulty = harder_test_analysis(config)
    storage_schema = dataset_storage_schema()
    normalization = normalization_contract(config)
    completion_checks = {
        "schema_defines_trajectory_state_action_initial_condition_and_solver_quality": (
            _storage_schema_is_complete(storage_schema)
        ),
        "train_validation_test_ranges_defined": all(
            set(config["offline_dataset"]["splits"][split]["ranges"]) == DATASET_RANGE_FIELDS
            for split in DATASET_SPLITS
        ),
        "trajectory_level_split_without_initial_condition_overlap": integrity["passed"],
        "normalization_fits_train_only": normalization["fit_split"] == "train"
        and normalization["validation_and_test_in_fit"] is False,
        "test_ranges_are_strictly_harder_than_train": difficulty["passed"],
    }
    report = {
        "schema_version": 1,
        "problem": "day22-offline-dataset-schema-and-split-design",
        "dataset_id": config["offline_dataset"]["id"],
        "dataset_configuration_sha256": dataset_configuration_sha256(config),
        "teacher_provenance": {
            "protocol_id": config["teacher_protocol"]["id"],
            "configuration_sha256": config["teacher_protocol"]["configuration_sha256"],
        },
        "storage_schema": storage_schema,
        "normalization": normalization,
        "split_plan": split_report,
        "split_integrity": integrity,
        "test_difficulty": difficulty,
        "completion_gate": {
            "checks": completion_checks,
            "passed": all(completion_checks.values()),
        },
        "planned_trajectories": sum(len(states) for states in split_states.values()),
    }
    for array in arrays.values():
        array.flags.writeable = False
    return report, arrays
