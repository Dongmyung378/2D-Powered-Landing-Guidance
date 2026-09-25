"""Day 24 hard-case sampling, coverage repair, and offline-dataset finalization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.offline_dataset import (
    compute_training_normalization_statistics,
    dataset_configuration_sha256,
    initial_condition_id,
    validate_split_integrity,
)
from powered_landing_guidance.offline_dataset_generation import (
    OfflineGenerationResult,
    OfflineGenerationSettings,
    packed_shard_sha256,
    validate_packed_trajectory_shard,
)

COVERAGE_FEATURES = (
    "abs_x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "theta_deg",
    "omega_deg_s",
    "mass_kg",
)


@dataclass(frozen=True, slots=True)
class FinalDatasetSettings:
    """Validated Day 24 generation, coverage, and completion policy."""

    schema_version: int
    dataset_id: str
    generation: dict[str, Any]
    hard_oversampling: dict[str, Any]
    coverage: dict[str, Any]
    completion: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FinalDatasetSettings:
        validate_config(config)
        settings = config["offline_dataset_finalization"]
        return cls(
            schema_version=int(settings["schema_version"]),
            dataset_id=str(settings["id"]),
            generation=dict(settings["generation"]),
            hard_oversampling=dict(settings["hard_oversampling"]),
            coverage=dict(settings["coverage"]),
            completion=dict(settings["completion"]),
        )

    def solver_settings(self, split: str, minimum: int) -> OfflineGenerationSettings:
        generation = self.generation
        return OfflineGenerationSettings(
            schema_version=1,
            split=split,
            minimum_validated_trajectories=minimum,
            runner=str(generation["runner"]),
            attempt_timeout_s=float(generation["attempt_timeout_s"]),
            max_retries=int(generation["max_retries"]),
            warm_start=str(generation["warm_start"]),
            retry_initial_guess=str(generation["retry_initial_guess"]),
            worker_restart_after_attempts=int(generation["worker_restart_after_attempts"]),
            replay_dt_s=float(generation["replay_dt_s"]),
            progress_interval=int(generation["progress_interval"]),
        )


def _sample_ranges(
    ranges: dict[str, Any],
    count: int,
    rng: np.random.Generator,
    target_x: float,
) -> NDArray[np.float64]:
    x_low, x_high = (float(value) for value in ranges["x_m"])
    min_abs_x = float(ranges["min_abs_x_m"])
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=count)
    negative = rng.uniform(x_low, target_x - min_abs_x, size=count)
    positive = rng.uniform(target_x + min_abs_x, x_high, size=count)
    states = np.empty((count, 7), dtype=np.float64)
    states[:, 0] = np.where(signs < 0.0, negative, positive)
    states[:, 1] = rng.uniform(*ranges["z_m"], size=count)
    states[:, 2] = rng.uniform(*ranges["vx_m_s"], size=count)
    states[:, 3] = rng.uniform(*ranges["vz_m_s"], size=count)
    states[:, 4] = np.deg2rad(rng.uniform(*ranges["theta_deg"], size=count))
    states[:, 5] = np.deg2rad(rng.uniform(*ranges["omega_deg_s"], size=count))
    states[:, 6] = rng.uniform(*ranges["mass_kg"], size=count)
    return states


def difficulty_scores(
    states: NDArray[np.float64],
    ranges: dict[str, Any],
    *,
    target_x: float = 0.0,
) -> NDArray[np.float64]:
    """Score larger offsets, rates, tilt, descent speed, altitude, and fuel scarcity."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("states must have shape (episodes, 7) and be finite")

    def scaled(data: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
        return np.clip((data - low) / (high - low), 0.0, 1.0)

    max_abs_x = max(abs(float(value) - target_x) for value in ranges["x_m"])
    max_abs_vx = max(abs(float(value)) for value in ranges["vx_m_s"])
    max_abs_theta = max(abs(float(value)) for value in ranges["theta_deg"])
    max_abs_omega = max(abs(float(value)) for value in ranges["omega_deg_s"])
    components = np.column_stack(
        (
            scaled(
                np.abs(values[:, 0] - target_x),
                float(ranges["min_abs_x_m"]),
                max_abs_x,
            ),
            scaled(values[:, 1], *map(float, ranges["z_m"])),
            np.clip(np.abs(values[:, 2]) / max_abs_vx, 0.0, 1.0),
            scaled(-values[:, 3], -float(ranges["vz_m_s"][1]), -float(ranges["vz_m_s"][0])),
            np.clip(np.abs(np.rad2deg(values[:, 4])) / max_abs_theta, 0.0, 1.0),
            np.clip(np.abs(np.rad2deg(values[:, 5])) / max_abs_omega, 0.0, 1.0),
            scaled(
                float(ranges["mass_kg"][1]) - values[:, 6],
                0.0,
                float(ranges["mass_kg"][1]) - float(ranges["mass_kg"][0]),
            ),
        )
    )
    scores = np.mean(components, axis=1)
    scores.flags.writeable = False
    return scores


def sample_hard_training_states(
    config: dict[str, Any],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Select a deterministic hard-biased training set from an oversized candidate pool."""
    settings = FinalDatasetSettings.from_config(config).hard_oversampling
    episodes = int(settings["episodes"])
    candidate_count = episodes * int(settings["candidate_multiplier"])
    rng = np.random.default_rng(int(settings["seed"]))
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    candidates = _sample_ranges(settings["ranges"], candidate_count, rng, target_x)
    scores = difficulty_scores(candidates, settings["ranges"], target_x=target_x)
    hardest_count = int(round(episodes * float(settings["hardest_fraction"])))
    hardest_count = min(episodes, max(1, hardest_count))
    order = np.argsort(-scores, kind="stable")
    hardest = order[:hardest_count]
    remaining_count = episodes - hardest_count
    if remaining_count:
        selected_remainder = rng.choice(order[hardest_count:], remaining_count, replace=False)
        selected_indices = np.concatenate((hardest, selected_remainder))
    else:
        selected_indices = hardest
    selected = np.ascontiguousarray(candidates[selected_indices], dtype=np.float64)
    selected_scores = scores[selected_indices]
    selected.flags.writeable = False
    report = {
        "sampler": settings["sampler"],
        "score": settings["score"],
        "seed": int(settings["seed"]),
        "candidate_count": candidate_count,
        "selected_count": episodes,
        "hardest_selected_count": hardest_count,
        "random_remainder_count": remaining_count,
        "candidate_score": _score_summary(scores),
        "selected_score": _score_summary(selected_scores),
        "selected_initial_condition_ids": [initial_condition_id(state) for state in selected],
    }
    return selected, report


def _score_summary(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def effective_train_ranges(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return the union envelope covered by easy and hard-biased training states."""
    easy = config["offline_dataset"]["splits"]["train"]["ranges"]
    hard = config["offline_dataset_finalization"]["hard_oversampling"]["ranges"]
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    return {
        "abs_x_m": (
            min(float(easy["min_abs_x_m"]), float(hard["min_abs_x_m"])),
            max(abs(float(value) - target_x) for value in (*easy["x_m"], *hard["x_m"])),
        ),
        "z_m": (
            min(float(easy["z_m"][0]), float(hard["z_m"][0])),
            max(float(easy["z_m"][1]), float(hard["z_m"][1])),
        ),
        "vx_m_s": (
            min(float(easy["vx_m_s"][0]), float(hard["vx_m_s"][0])),
            max(float(easy["vx_m_s"][1]), float(hard["vx_m_s"][1])),
        ),
        "vz_m_s": (
            min(float(easy["vz_m_s"][0]), float(hard["vz_m_s"][0])),
            max(float(easy["vz_m_s"][1]), float(hard["vz_m_s"][1])),
        ),
        "theta_deg": (
            min(float(easy["theta_deg"][0]), float(hard["theta_deg"][0])),
            max(float(easy["theta_deg"][1]), float(hard["theta_deg"][1])),
        ),
        "omega_deg_s": (
            min(float(easy["omega_deg_s"][0]), float(hard["omega_deg_s"][0])),
            max(float(easy["omega_deg_s"][1]), float(hard["omega_deg_s"][1])),
        ),
        "mass_kg": (
            min(float(easy["mass_kg"][0]), float(hard["mass_kg"][0])),
            max(float(easy["mass_kg"][1]), float(hard["mass_kg"][1])),
        ),
    }


def coverage_feature_matrix(
    states: NDArray[np.float64],
    *,
    target_x: float = 0.0,
) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("states must have shape (episodes, 7) and be finite")
    matrix = values.copy()
    matrix[:, 0] = np.abs(matrix[:, 0] - target_x)
    matrix[:, 4:6] = np.rad2deg(matrix[:, 4:6])
    return matrix


def analyze_coverage(
    config: dict[str, Any],
    states: NDArray[np.float64],
) -> dict[str, Any]:
    """Measure marginal coverage over the final expanded training envelope."""
    settings = FinalDatasetSettings.from_config(config).coverage
    bins = int(settings["bins_per_feature"])
    minimum = int(settings["minimum_count_per_bin"])
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    matrix = coverage_feature_matrix(states, target_x=target_x)
    ranges = effective_train_ranges(config)
    features: dict[str, Any] = {}
    for column, feature in enumerate(COVERAGE_FEATURES):
        low, high = ranges[feature]
        edges = np.linspace(low, high, bins + 1)
        counts, _ = np.histogram(matrix[:, column], bins=edges)
        deficits = np.maximum(minimum - counts, 0)
        features[feature] = {
            "range": [low, high],
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "deficits": deficits.tolist(),
            "minimum_count": int(np.min(counts)),
        }
    return {
        "bins_per_feature": bins,
        "required_count_per_bin": minimum,
        "features": features,
        "total_deficit": sum(sum(item["deficits"]) for item in features.values()),
        "passed": all(item["minimum_count"] >= minimum for item in features.values()),
    }


def _interleaved_deficit_targets(coverage: dict[str, Any]) -> list[tuple[int, int]]:
    deficits = [
        np.asarray(coverage["features"][feature]["deficits"], dtype=np.int64)
        for feature in COVERAGE_FEATURES
    ]
    targets: list[tuple[int, int]] = []
    while any(np.any(values > 0) for values in deficits):
        for feature_index, values in enumerate(deficits):
            if np.any(values > 0):
                bin_index = int(np.argmax(values))
                targets.append((feature_index, bin_index))
                values[bin_index] -= 1
    return targets


def sample_coverage_states(
    config: dict[str, Any],
    existing_states: NDArray[np.float64],
    *,
    round_index: int,
    maximum_count: int,
    excluded_initial_ids: set[str] | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Create states aimed at currently deficient marginal coverage bins."""
    if maximum_count <= 0:
        return np.empty((0, 7), dtype=np.float64), []
    final = FinalDatasetSettings.from_config(config)
    coverage = analyze_coverage(config, existing_states)
    targets = _interleaved_deficit_targets(coverage)[:maximum_count]
    if not targets:
        return np.empty((0, 7), dtype=np.float64), []
    ranges = effective_train_ranges(config)
    rng = np.random.default_rng(int(final.coverage["seed"]) + round_index)
    seen = set() if excluded_initial_ids is None else set(excluded_initial_ids)
    bins = int(final.coverage["bins_per_feature"])
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    states: list[NDArray[np.float64]] = []
    plan: list[dict[str, Any]] = []
    for feature_index, bin_index in targets:
        for _ in range(100):
            feature_values = np.asarray(
                [rng.uniform(*ranges[feature]) for feature in COVERAGE_FEATURES],
                dtype=np.float64,
            )
            low, high = ranges[COVERAGE_FEATURES[feature_index]]
            edges = np.linspace(low, high, bins + 1)
            feature_values[feature_index] = rng.uniform(edges[bin_index], edges[bin_index + 1])
            sign = rng.choice(np.asarray((-1.0, 1.0)))
            state = np.asarray(
                (
                    target_x + sign * feature_values[0],
                    feature_values[1],
                    feature_values[2],
                    feature_values[3],
                    np.deg2rad(feature_values[4]),
                    np.deg2rad(feature_values[5]),
                    feature_values[6],
                ),
                dtype=np.float64,
            )
            identifier = initial_condition_id(state)
            if identifier not in seen:
                seen.add(identifier)
                states.append(state)
                plan.append(
                    {
                        "feature": COVERAGE_FEATURES[feature_index],
                        "bin_index": bin_index,
                        "initial_condition_id": identifier,
                    }
                )
                break
        else:
            raise RuntimeError("could not sample a unique coverage initial condition")
    result = np.asarray(states, dtype=np.float64)
    result.flags.writeable = False
    return result, plan


def initial_states_from_shard(arrays: dict[str, NDArray[Any]]) -> NDArray[np.float64]:
    validation = validate_packed_trajectory_shard(arrays)
    if not validation["passed"]:
        raise ValueError("packed trajectory shard is invalid")
    states = np.asarray(arrays["states"][arrays["state_offsets"][:-1]], dtype=np.float64)
    states.flags.writeable = False
    return states


def merge_packed_shards(
    shards: Sequence[dict[str, NDArray[Any]]],
) -> dict[str, NDArray[Any]]:
    """Merge accepted-only packed shards and rebuild canonical offsets and case indices."""
    if not shards:
        raise ValueError("at least one shard is required")
    for shard in shards:
        if not validate_packed_trajectory_shard(shard)["passed"]:
            raise ValueError("cannot merge an invalid packed trajectory shard")
    trajectory_ids = np.concatenate([shard["trajectory_ids"] for shard in shards])
    initial_ids = np.concatenate([shard["initial_condition_ids"] for shard in shards])
    if len(set(trajectory_ids.tolist())) != len(trajectory_ids):
        raise ValueError("duplicate trajectory ID across source shards")
    if len(set(initial_ids.tolist())) != len(initial_ids):
        raise ValueError("duplicate initial condition across source shards")
    state_counts = np.concatenate([np.diff(shard["state_offsets"]) for shard in shards])
    action_counts = np.concatenate([np.diff(shard["action_offsets"]) for shard in shards])
    trajectory_count = len(trajectory_ids)
    arrays: dict[str, NDArray[Any]] = {
        "trajectory_ids": trajectory_ids,
        "initial_condition_ids": initial_ids,
        "source_case_indices": np.arange(trajectory_count, dtype=np.int64),
        "state_offsets": np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(state_counts, dtype=np.int64))
        ),
        "action_offsets": np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(action_counts, dtype=np.int64))
        ),
        "times_s": np.concatenate([shard["times_s"] for shard in shards]),
        "states": np.concatenate([shard["states"] for shard in shards]),
        "actions": np.concatenate([shard["actions"] for shard in shards]),
    }
    per_trajectory = (
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
    )
    arrays.update(
        {name: np.concatenate([shard[name] for shard in shards]) for name in per_trajectory}
    )
    for values in arrays.values():
        values.flags.writeable = False
    if not validate_packed_trajectory_shard(arrays)["passed"]:
        raise RuntimeError("merged packed trajectory shard failed validation")
    return arrays


def compute_shard_normalization(
    config: dict[str, Any],
    train_arrays: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    """Fit normalization on every nonterminal transition in the final train shard only."""
    if not validate_packed_trajectory_shard(train_arrays)["passed"]:
        raise ValueError("train shard is invalid")
    state_rows = [
        train_arrays["states"][
            int(train_arrays["state_offsets"][index]) : int(
                train_arrays["state_offsets"][index + 1]
            )
            - 1
        ]
        for index in range(len(train_arrays["trajectory_ids"]))
    ]
    states = np.concatenate(state_rows)
    actions = np.asarray(train_arrays["actions"], dtype=np.float64)
    return compute_training_normalization_statistics(config, states, actions)


def final_test_difficulty_analysis(config: dict[str, Any]) -> dict[str, Any]:
    """Verify the frozen test split stays harder than the expanded training envelope."""
    train = effective_train_ranges(config)
    test = config["offline_dataset"]["splits"]["test"]["ranges"]
    checks = {
        "horizontal_offset_is_disjoint_and_larger": float(test["min_abs_x_m"])
        > train["abs_x_m"][1],
        "altitude_is_disjoint_and_higher": float(test["z_m"][0]) > train["z_m"][1],
        "descent_speed_is_disjoint_and_faster": float(test["vz_m_s"][1]) < train["vz_m_s"][0],
        "available_propellant_is_disjoint_and_lower": float(test["mass_kg"][1])
        < train["mass_kg"][0],
        "horizontal_velocity_envelope_is_wider": float(test["vx_m_s"][0]) < train["vx_m_s"][0]
        and float(test["vx_m_s"][1]) > train["vx_m_s"][1],
        "attitude_envelope_is_wider": float(test["theta_deg"][0]) < train["theta_deg"][0]
        and float(test["theta_deg"][1]) > train["theta_deg"][1],
        "angular_rate_envelope_is_wider": float(test["omega_deg_s"][0]) < train["omega_deg_s"][0]
        and float(test["omega_deg_s"][1]) > train["omega_deg_s"][1],
    }
    return {"checks": checks, "passed": all(checks.values())}


def load_verified_source_train(
    config: dict[str, Any],
    source_dir: Path,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Load the Day 23 train shard only after manifest, digest, and schema checks pass."""
    manifest_path = source_dir / "train-manifest.json"
    shard_path = source_dir / "train-trajectories.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(shard_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    checks = {
        "problem": manifest.get("problem") == "day23-large-scale-easy-teacher-generation",
        "split": manifest.get("split") == "train",
        "completion_gate": manifest.get("completion_gate", {}).get("passed") is True,
        "dataset_configuration": manifest.get("dataset_configuration_sha256")
        == dataset_configuration_sha256(config),
        "shard_validation": validate_packed_trajectory_shard(arrays)["passed"],
        "shard_sha256": manifest.get("shard", {}).get("sha256") == packed_shard_sha256(arrays),
    }
    if not all(checks.values()):
        raise ValueError(f"Day 23 source train verification failed: {checks}")
    for values in arrays.values():
        values.flags.writeable = False
    return arrays, {
        "directory": str(source_dir),
        "trajectory_count": len(arrays["trajectory_ids"]),
        "shard_sha256": packed_shard_sha256(arrays),
        "checks": checks,
        "passed": True,
    }


def generation_summary(result: OfflineGenerationResult) -> dict[str, Any]:
    batch = result.solver_batch
    attempts = [attempt for case in batch.cases for attempt in case["attempts"]]
    failures = [case for case in batch.cases if case["status"] != "success"]
    return {
        "planned_cases": len(batch.initial_states),
        "executed_cases": len(batch.cases),
        "solver_and_replay_successes": batch.successful_cases,
        "accepted_trajectories": result.accepted_trajectories,
        "filter_rejections": len(result.rejections),
        "failed_cases": len(failures),
        "retried_cases": sum(case["attempt_count"] > 1 for case in batch.cases),
        "timeout_attempts": sum(attempt["status"] == "timeout" for attempt in attempts),
        "wall_time_s": batch.wall_time_s,
        "failures": failures,
        "filter_rejection_records": list(result.rejections),
    }


def build_final_dataset_manifest(
    config: dict[str, Any],
    *,
    source_train: dict[str, Any],
    hard_selection: dict[str, Any],
    challenge_result: OfflineGenerationResult,
    coverage_results: Sequence[OfflineGenerationResult],
    coverage_rounds: Sequence[dict[str, Any]],
    validation_result: OfflineGenerationResult,
    test_result: OfflineGenerationResult,
    shards: dict[str, dict[str, NDArray[Any]]],
    normalization: dict[str, Any],
    coverage_before: dict[str, Any],
    coverage_after: dict[str, Any],
    progress: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build the final versioned manifest and its Day 24 completion evidence."""
    settings = FinalDatasetSettings.from_config(config)
    identifiers = {split: shard["initial_condition_ids"] for split, shard in shards.items()}
    split_integrity = validate_split_integrity(identifiers)
    difficulty = final_test_difficulty_analysis(config)
    shard_validation = {
        split: validate_packed_trajectory_shard(shard) for split, shard in shards.items()
    }
    shard_hashes = {split: packed_shard_sha256(shard) for split, shard in shards.items()}
    counts = {split: len(shard["trajectory_ids"]) for split, shard in shards.items()}
    completion = settings.completion
    checks = {
        "verified_day23_source_train": source_train["passed"],
        "hard_cases_oversampled": hard_selection["candidate_count"]
        > hard_selection["selected_count"]
        and hard_selection["hardest_selected_count"] > 0,
        "expanded_train_stays_easier_than_test": difficulty["passed"],
        "coverage_target_met": coverage_after["passed"],
        "train_minimum_met": counts["train"] >= int(completion["minimum_train_trajectories"]),
        "validation_minimum_met": counts["validation"]
        >= int(completion["minimum_validation_trajectories"]),
        "test_minimum_met": counts["test"] >= int(completion["minimum_test_trajectories"]),
        "all_shards_valid": all(item["passed"] for item in shard_validation.values()),
        "trajectory_split_integrity": split_integrity["passed"],
        "normalization_fit_on_train_only": normalization["fit_split"] == "train",
    }
    return {
        "schema_version": 1,
        "problem": "day24-final-offline-dataset",
        "status": "complete" if all(checks.values()) else "incomplete",
        "dataset_id": settings.dataset_id,
        "base_dataset_id": config["offline_dataset"]["id"],
        "dataset_configuration_sha256": dataset_configuration_sha256(config),
        "teacher_provenance": {
            "protocol_id": config["teacher_protocol"]["id"],
            "configuration_sha256": config["teacher_protocol"]["configuration_sha256"],
        },
        "settings": asdict(settings),
        "source_train": source_train,
        "hard_oversampling": hard_selection,
        "generation": {
            "challenge_train": generation_summary(challenge_result),
            "coverage_rounds": [generation_summary(result) for result in coverage_results],
            "coverage_plans": list(coverage_rounds),
            "validation": generation_summary(validation_result),
            "test": generation_summary(test_result),
        },
        "coverage": {"before_supplement": coverage_before, "final": coverage_after},
        "test_difficulty": difficulty,
        "split_integrity": split_integrity,
        "normalization": normalization,
        "shards": {
            split: {
                "file": f"{split}-trajectories.npz",
                "trajectory_count": counts[split],
                "sha256": shard_hashes[split],
                "validation": shard_validation[split],
            }
            for split in ("train", "validation", "test")
        },
        "progress": progress,
        "completion_gate": {
            "criterion": (
                "produce leakage-free train, validation, and harder test shards; meet coverage "
                "and minimum-count requirements; and fit normalization on train only"
            ),
            "checks": checks,
            "passed": all(checks.values()),
        },
        "artifacts": {
            "manifest": "dataset-manifest.json",
            "normalization": "normalization.json",
            "distribution_plot": "dataset-distribution.png",
            "train_shard": "train-trajectories.npz",
            "validation_shard": "validation-trajectories.npz",
            "test_shard": "test-trajectories.npz",
        },
    }


def save_distribution_plot(
    config: dict[str, Any],
    shards: dict[str, dict[str, NDArray[Any]]],
    output_path: Path,
) -> None:
    """Plot the initial-condition distributions for all final trajectory splits."""
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    matrices = {
        split: coverage_feature_matrix(initial_states_from_shard(shard), target_x=target_x)
        for split, shard in shards.items()
    }
    labels = (
        "|x| (m)",
        "z (m)",
        "vx (m/s)",
        "vz (m/s)",
        "theta (deg)",
        "omega (deg/s)",
        "mass (kg)",
    )
    colors = {"train": "#1f77b4", "validation": "#2ca02c", "test": "#d62728"}
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.5), constrained_layout=True)
    for column, (feature, label) in enumerate(zip(COVERAGE_FEATURES, labels, strict=True)):
        axis = axes.flat[column]
        combined = np.concatenate([matrix[:, column] for matrix in matrices.values()])
        edges = np.linspace(float(np.min(combined)), float(np.max(combined)), 17)
        for split in ("train", "validation", "test"):
            axis.hist(
                matrices[split][:, column],
                bins=edges,
                histtype="step",
                linewidth=1.8,
                color=colors[split],
                label=split,
            )
        axis.set_title(feature)
        axis.set_xlabel(label)
        axis.set_ylabel("Trajectories")
        axis.grid(alpha=0.2)
    summary_axis = axes.flat[7]
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.95,
        "Final offline dataset\n\n"
        + "\n".join(f"{split}: {len(shard['trajectory_ids'])}" for split, shard in shards.items()),
        va="top",
        fontsize=13,
    )
    axes.flat[0].legend(loc="best")
    figure.suptitle("Day 24 final initial-condition distribution", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
