"""Day 23 verified Teacher generation for the easy offline-dataset split."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.evaluation import initial_condition_sha256
from powered_landing_guidance.offline_dataset import (
    dataset_configuration_sha256,
    initial_condition_id,
    sample_split_initial_states,
    split_initial_condition_ids,
)
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem
from powered_landing_guidance.planar_optimal_control import PlanarLandingResult
from powered_landing_guidance.teacher_pipeline import (
    ProgressCallback,
    TeacherSolveBatchResult,
    solve_teacher_batch,
)

_TRAJECTORY_ID_PREFIX = b"powered-landing-guidance:trajectory:v1\n"
_SHARD_DIGEST_PREFIX = b"powered-landing-guidance:packed-shard:v1\n"


@dataclass(frozen=True, slots=True)
class OfflineGenerationSettings:
    """Validated Day 23 generation and completion policy."""

    schema_version: int
    split: str
    minimum_validated_trajectories: int
    runner: str
    attempt_timeout_s: float
    max_retries: int
    warm_start: str
    retry_initial_guess: str
    worker_restart_after_attempts: int
    replay_dt_s: float
    progress_interval: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OfflineGenerationSettings:
        validate_config(config)
        settings = config["offline_dataset_generation"]
        return cls(
            schema_version=int(settings["schema_version"]),
            split=str(settings["split"]),
            minimum_validated_trajectories=int(settings["minimum_validated_trajectories"]),
            runner=str(settings["runner"]),
            attempt_timeout_s=float(settings["attempt_timeout_s"]),
            max_retries=int(settings["max_retries"]),
            warm_start=str(settings["warm_start"]),
            retry_initial_guess=str(settings["retry_initial_guess"]),
            worker_restart_after_attempts=int(settings["worker_restart_after_attempts"]),
            replay_dt_s=float(settings["replay_dt_s"]),
            progress_interval=int(settings["progress_interval"]),
        )


@dataclass(frozen=True, slots=True)
class OfflineGenerationResult:
    """Solver output, filtering decisions, and packed accepted trajectories."""

    settings: OfflineGenerationSettings
    initial_condition_ids: NDArray[np.str_]
    solver_batch: TeacherSolveBatchResult
    accepted_case_indices: tuple[int, ...]
    trajectory_ids: tuple[str, ...]
    rejections: tuple[dict[str, Any], ...]
    arrays: dict[str, NDArray[Any]]

    @property
    def accepted_trajectories(self) -> int:
        return len(self.accepted_case_indices)

    @property
    def completion_passed(self) -> bool:
        return (
            len(self.solver_batch.cases) == len(self.solver_batch.initial_states)
            and self.accepted_trajectories >= self.settings.minimum_validated_trajectories
        )


def trajectory_id(
    config: dict[str, Any],
    split: str,
    initial_id: str,
) -> str:
    """Create a stable ID from dataset, split, Teacher, and initial-condition identity."""
    if split not in config["offline_dataset"]["splits"]:
        raise ValueError("unknown dataset split")
    if len(initial_id) != 64 or any(
        character not in "0123456789abcdef" for character in initial_id
    ):
        raise ValueError("initial_id must be a lowercase SHA-256 digest")
    components = (
        config["offline_dataset"]["id"],
        split,
        config["teacher_protocol"]["id"],
        config["teacher_protocol"]["configuration_sha256"],
        initial_id,
    )
    payload = "\0".join(components).encode("ascii")
    return hashlib.sha256(_TRAJECTORY_ID_PREFIX + payload).hexdigest()


def teacher_trajectory_rejection_reasons(
    config: dict[str, Any],
    initial_state: NDArray[np.float64],
    initial_id: str,
    case: dict[str, Any],
    result: PlanarLandingResult,
    seen_initial_ids: set[str],
    seen_trajectory_ids: set[str],
    *,
    split: str,
) -> tuple[str, ...]:
    """Return every duplicate, nonfinite, structural, or constraint rejection reason."""
    reasons: set[str] = set()
    expected_intervals = int(config["optimal_control"]["intervals"])
    expected_trajectory_id = trajectory_id(config, split, initial_id)
    if initial_id in seen_initial_ids:
        reasons.add("duplicate_initial_condition")
    if expected_trajectory_id in seen_trajectory_ids:
        reasons.add("duplicate_trajectory")
    try:
        canonical_id = initial_condition_id(initial_state)
    except ValueError:
        canonical_id = ""
        reasons.add("nonfinite_initial_condition")
    if canonical_id != initial_id:
        reasons.add("initial_condition_id_mismatch")

    arrays = (
        result.times_s,
        result.states,
        result.controls,
        result.rollout_times_s,
        result.rollout_states,
    )
    if any(not np.all(np.isfinite(np.asarray(values))) for values in arrays):
        reasons.add("nonfinite_trajectory")
    if not np.isfinite(result.duration_s) or result.duration_s <= 0.0:
        reasons.add("invalid_duration")

    expected_shapes = (
        result.times_s.shape == (expected_intervals + 1,),
        result.states.shape == (expected_intervals + 1, 7),
        result.controls.shape == (expected_intervals, 2),
        result.rollout_times_s.ndim == 1,
        result.rollout_states.ndim == 2 and result.rollout_states.shape[1] == 7,
        len(result.rollout_times_s) == len(result.rollout_states),
    )
    if not all(expected_shapes):
        reasons.add("invalid_trajectory_shape")
    else:
        if not np.allclose(result.states[0], initial_state, rtol=0.0, atol=1.0e-12):
            reasons.add("initial_state_mismatch")
        if (
            result.times_s[0] != 0.0
            or not np.all(np.diff(result.times_s) > 0.0)
            or not np.isclose(result.times_s[-1], result.duration_s, rtol=0.0, atol=1.0e-12)
        ):
            reasons.add("invalid_time_grid")

    replay = case.get("replay_validation")
    replay_checks = replay.get("checks") if isinstance(replay, dict) else None
    feasibility_tolerance = float(config["optimal_control"]["feasibility_tolerance"])
    numeric_quality = (
        result.stage_a.max_hard_violation,
        result.stage_a.max_terminal_violation,
        result.stage_b.max_hard_violation,
        result.stage_b.max_terminal_violation,
        case.get("fuel_used_kg", np.nan),
        replay.get("max_final_error_ratio", np.nan) if isinstance(replay, dict) else np.nan,
        replay.get("max_node_error_ratio", np.nan) if isinstance(replay, dict) else np.nan,
        replay.get("max_terminal_violation", np.nan) if isinstance(replay, dict) else np.nan,
    )
    if not np.all(np.isfinite(np.asarray(numeric_quality, dtype=np.float64))):
        reasons.add("nonfinite_quality_metric")

    p = config["vehicle"]
    controls = np.asarray(result.controls, dtype=np.float64)
    actuator_violation = False
    if controls.ndim == 2 and controls.shape[1] == 2 and np.all(np.isfinite(controls)):
        gimbal_limit = np.deg2rad(float(p["gimbal_limit_deg"]))
        actuator_violation = bool(
            np.any(controls[:, 0] < float(p["throttle_min"]) - 1.0e-12)
            or np.any(controls[:, 0] > float(p["throttle_max"]) + 1.0e-12)
            or np.any(np.abs(controls[:, 1]) > gimbal_limit + 1.0e-12)
        )
    physical_constraint_violation = True
    if (
        result.rollout_states.ndim == 2
        and result.rollout_states.shape[1] == 7
        and len(result.rollout_states) > 0
        and np.all(np.isfinite(result.rollout_states))
    ):
        problem = LandingOptimalControlProblem.from_config(config, initial_state)
        tolerance_config = config["optimal_control"]["study"]["final_state_tolerances"]
        replay_tolerances = np.asarray(
            (
                tolerance_config["x_m"],
                tolerance_config["z_m"],
                tolerance_config["vx_m_s"],
                tolerance_config["vz_m_s"],
                np.deg2rad(tolerance_config["theta_deg"]),
                np.deg2rad(tolerance_config["omega_deg_s"]),
                tolerance_config["mass_kg"],
            ),
            dtype=np.float64,
        )
        rollout = result.rollout_states
        physical_constraint_violation = bool(
            np.max(problem.terminal_violations(rollout[-1])) > feasibility_tolerance
            or np.min(rollout[:, 1]) < problem.ground_z_m - replay_tolerances[1]
            or np.min(rollout[:, 6])
            < problem.parameters.dry_mass_kg
            + problem.min_propellant_reserve_kg
            - replay_tolerances[6]
            or np.max(np.abs(rollout[:, 4])) > problem.max_abs_tilt_rad + replay_tolerances[4]
            or np.max(np.abs(rollout[:, 5]))
            > problem.max_abs_angular_rate_rad_s + replay_tolerances[5]
        )
    if (
        case.get("status") != "success"
        or not isinstance(replay, dict)
        or replay.get("passed") is not True
        or not isinstance(replay_checks, dict)
        or not replay_checks
        or not all(value is True for value in replay_checks.values())
        or result.stage_b.max_hard_violation > feasibility_tolerance
        or result.stage_b.max_terminal_violation > feasibility_tolerance
        or (
            isinstance(replay, dict)
            and float(replay.get("max_terminal_violation", np.inf)) > feasibility_tolerance
        )
        or actuator_violation
        or physical_constraint_violation
    ):
        reasons.add("constraint_violation")
    return tuple(sorted(reasons))


def _empty_packed_arrays() -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {
        "trajectory_ids": np.empty((0,), dtype="<U64"),
        "initial_condition_ids": np.empty((0,), dtype="<U64"),
        "source_case_indices": np.empty((0,), dtype=np.int64),
        "state_offsets": np.zeros((1,), dtype=np.int64),
        "action_offsets": np.zeros((1,), dtype=np.int64),
        "times_s": np.empty((0,), dtype=np.float64),
        "states": np.empty((0, 7), dtype=np.float64),
        "actions": np.empty((0, 2), dtype=np.float64),
        "durations_s": np.empty((0,), dtype=np.float64),
        "solver_success": np.empty((0,), dtype=np.bool_),
        "stage_a_iterations": np.empty((0,), dtype=np.int64),
        "stage_b_iterations": np.empty((0,), dtype=np.int64),
        "max_hard_violation": np.empty((0,), dtype=np.float64),
        "max_terminal_violation": np.empty((0,), dtype=np.float64),
        "replay_passed": np.empty((0,), dtype=np.bool_),
        "max_final_error_ratio": np.empty((0,), dtype=np.float64),
        "max_node_error_ratio": np.empty((0,), dtype=np.float64),
        "fuel_used_kg": np.empty((0,), dtype=np.float64),
        "attempt_count": np.empty((0,), dtype=np.int64),
    }
    for values in arrays.values():
        values.flags.writeable = False
    return arrays


def _pack_trajectories(
    accepted: list[tuple[int, str, str, dict[str, Any], PlanarLandingResult]],
) -> dict[str, NDArray[Any]]:
    if not accepted:
        return _empty_packed_arrays()
    state_counts = np.asarray([len(item[4].states) for item in accepted], dtype=np.int64)
    action_counts = np.asarray([len(item[4].controls) for item in accepted], dtype=np.int64)
    state_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(state_counts)))
    action_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(action_counts)))
    arrays: dict[str, NDArray[Any]] = {
        "trajectory_ids": np.asarray([item[2] for item in accepted], dtype="<U64"),
        "initial_condition_ids": np.asarray([item[1] for item in accepted], dtype="<U64"),
        "source_case_indices": np.asarray([item[0] for item in accepted], dtype=np.int64),
        "state_offsets": state_offsets,
        "action_offsets": action_offsets,
        "times_s": np.concatenate([item[4].times_s for item in accepted]),
        "states": np.concatenate([item[4].states for item in accepted]),
        "actions": np.concatenate([item[4].controls for item in accepted]),
        "durations_s": np.asarray([item[4].duration_s for item in accepted]),
        "solver_success": np.ones(len(accepted), dtype=np.bool_),
        "stage_a_iterations": np.asarray(
            [item[4].stage_a.iterations for item in accepted], dtype=np.int64
        ),
        "stage_b_iterations": np.asarray(
            [item[4].stage_b.iterations for item in accepted], dtype=np.int64
        ),
        "max_hard_violation": np.asarray([item[4].stage_b.max_hard_violation for item in accepted]),
        "max_terminal_violation": np.asarray(
            [
                max(
                    item[4].stage_b.max_terminal_violation,
                    float(item[3]["replay_validation"]["max_terminal_violation"]),
                )
                for item in accepted
            ]
        ),
        "replay_passed": np.ones(len(accepted), dtype=np.bool_),
        "max_final_error_ratio": np.asarray(
            [item[3]["replay_validation"]["max_final_error_ratio"] for item in accepted]
        ),
        "max_node_error_ratio": np.asarray(
            [item[3]["replay_validation"]["max_node_error_ratio"] for item in accepted]
        ),
        "fuel_used_kg": np.asarray([item[3]["fuel_used_kg"] for item in accepted]),
        "attempt_count": np.asarray(
            [item[3]["attempt_count"] for item in accepted], dtype=np.int64
        ),
    }
    for values in arrays.values():
        values.flags.writeable = False
    return arrays


def validate_packed_trajectory_shard(arrays: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """Validate packed offsets, identities, finite values, and accepted-only quality flags."""
    empty_arrays = _empty_packed_arrays()
    expected = set(empty_arrays)
    checks: dict[str, bool] = {"fields": set(arrays) == expected}
    if not checks["fields"]:
        return {"checks": checks, "passed": False}
    checks["dtypes"] = all(
        arrays[name].dtype == values.dtype for name, values in empty_arrays.items()
    )
    trajectory_count = len(arrays["trajectory_ids"])
    per_trajectory = (
        "initial_condition_ids",
        "source_case_indices",
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
    checks["per_trajectory_shapes"] = all(
        arrays[name].shape == (trajectory_count,) for name in per_trajectory
    )
    checks["state_shape"] = arrays["states"].ndim == 2 and arrays["states"].shape[1] == 7
    checks["action_shape"] = arrays["actions"].ndim == 2 and arrays["actions"].shape[1] == 2
    checks["offset_shapes"] = arrays["state_offsets"].shape == (trajectory_count + 1,) and arrays[
        "action_offsets"
    ].shape == (trajectory_count + 1,)
    offsets_valid = checks["offset_shapes"]
    if offsets_valid:
        state_offsets = arrays["state_offsets"]
        action_offsets = arrays["action_offsets"]
        offsets_valid = bool(
            state_offsets[0] == 0
            and action_offsets[0] == 0
            and state_offsets[-1] == len(arrays["states"])
            and state_offsets[-1] == len(arrays["times_s"])
            and action_offsets[-1] == len(arrays["actions"])
            and np.all(np.diff(state_offsets) > 0)
            and np.all(np.diff(action_offsets) > 0)
            and np.all(np.diff(state_offsets) == np.diff(action_offsets) + 1)
        )
    checks["offset_values"] = offsets_valid
    float_arrays = (
        "times_s",
        "states",
        "actions",
        "durations_s",
        "max_hard_violation",
        "max_terminal_violation",
        "max_final_error_ratio",
        "max_node_error_ratio",
        "fuel_used_kg",
    )
    checks["finite_values"] = all(np.all(np.isfinite(arrays[name])) for name in float_arrays)
    checks["unique_initial_conditions"] = len(set(arrays["initial_condition_ids"].tolist())) == (
        trajectory_count
    )
    checks["unique_trajectories"] = len(set(arrays["trajectory_ids"].tolist())) == trajectory_count
    checks["accepted_only"] = bool(
        np.all(arrays["solver_success"]) and np.all(arrays["replay_passed"])
    )
    checks["positive_attempt_counts"] = bool(np.all(arrays["attempt_count"] > 0))
    time_grids_valid = offsets_valid and checks["finite_values"]
    identities_valid = offsets_valid
    if offsets_valid:
        for index in range(trajectory_count):
            start = int(arrays["state_offsets"][index])
            end = int(arrays["state_offsets"][index + 1])
            times = arrays["times_s"][start:end]
            if (
                times[0] != 0.0
                or not np.all(np.diff(times) > 0.0)
                or not np.isclose(times[-1], arrays["durations_s"][index], rtol=0.0, atol=1.0e-12)
            ):
                time_grids_valid = False
            try:
                reconstructed_id = initial_condition_id(arrays["states"][start])
            except ValueError:
                reconstructed_id = ""
            if reconstructed_id != arrays["initial_condition_ids"][index]:
                identities_valid = False
    checks["time_grids"] = time_grids_valid
    checks["initial_condition_identity"] = identities_valid
    return {"checks": checks, "passed": all(checks.values())}


def packed_shard_sha256(arrays: dict[str, NDArray[Any]]) -> str:
    """Hash a shard canonically without depending on ZIP metadata or file timestamps."""
    digest = hashlib.sha256(_SHARD_DIGEST_PREFIX)
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        canonical_dtype = values.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(values, dtype=canonical_dtype)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(canonical.shape, dtype="<u8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def generate_easy_teacher_trajectories(
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> OfflineGenerationResult:
    """Generate, filter, and pack the configured Day 23 easy training split."""
    settings = OfflineGenerationSettings.from_config(config)
    initial_states = sample_split_initial_states(config, settings.split)
    initial_ids = split_initial_condition_ids(initial_states)
    batch = solve_teacher_batch(
        config,
        initial_states,
        attempt_timeout_s=settings.attempt_timeout_s,
        max_retries=settings.max_retries,
        retry_initial_guess=settings.retry_initial_guess,
        worker_restart_after_attempts=settings.worker_restart_after_attempts,
        replay_dt_s=settings.replay_dt_s,
        progress=progress,
    )
    solution_by_case = dict(zip(batch.successful_case_indices, batch.solutions, strict=True))
    accepted: list[tuple[int, str, str, dict[str, Any], PlanarLandingResult]] = []
    rejections: list[dict[str, Any]] = []
    seen_initial_ids: set[str] = set()
    seen_trajectory_ids: set[str] = set()
    for case in batch.cases:
        case_index = int(case["case_index"])
        result = solution_by_case.get(case_index)
        if result is None:
            continue
        initial_id_value = str(initial_ids[case_index])
        trajectory_id_value = trajectory_id(config, settings.split, initial_id_value)
        reasons = teacher_trajectory_rejection_reasons(
            config,
            batch.initial_states[case_index],
            initial_id_value,
            case,
            result,
            seen_initial_ids,
            seen_trajectory_ids,
            split=settings.split,
        )
        if reasons:
            rejections.append({"case_index": case_index, "reasons": list(reasons)})
            continue
        accepted.append((case_index, initial_id_value, trajectory_id_value, case, result))
        seen_initial_ids.add(initial_id_value)
        seen_trajectory_ids.add(trajectory_id_value)
    arrays = _pack_trajectories(accepted)
    return OfflineGenerationResult(
        settings=settings,
        initial_condition_ids=initial_ids,
        solver_batch=batch,
        accepted_case_indices=tuple(item[0] for item in accepted),
        trajectory_ids=tuple(item[2] for item in accepted),
        rejections=tuple(rejections),
        arrays=arrays,
    )


def build_offline_generation_report(
    config: dict[str, Any],
    result: OfflineGenerationResult,
    progress_measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the Day 23 manifest with solver, filter, timing, and completion evidence."""
    batch = result.solver_batch
    failures = [case for case in batch.cases if case["status"] != "success"]
    attempts = [attempt for case in batch.cases for attempt in case["attempts"]]
    shard_validation = validate_packed_trajectory_shard(result.arrays)
    rejection_counts: dict[str, int] = {}
    for rejection in result.rejections:
        for reason in rejection["reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    completion_checks = {
        "all_planned_initial_conditions_executed": len(batch.cases) == len(batch.initial_states),
        "minimum_validated_trajectories": (
            result.accepted_trajectories >= result.settings.minimum_validated_trajectories
        ),
        "packed_shard_validation": shard_validation["passed"],
    }
    return {
        "schema_version": 1,
        "problem": "day23-large-scale-easy-teacher-generation",
        "status": "complete" if all(completion_checks.values()) else "incomplete",
        "dataset_id": config["offline_dataset"]["id"],
        "split": result.settings.split,
        "dataset_configuration_sha256": dataset_configuration_sha256(config),
        "teacher_provenance": {
            "protocol_id": config["teacher_protocol"]["id"],
            "configuration_sha256": config["teacher_protocol"]["configuration_sha256"],
        },
        "settings": asdict(result.settings),
        "initial_conditions": {
            "episodes": len(batch.initial_states),
            "seed": config["offline_dataset"]["splits"][result.settings.split]["seed"],
            "sha256": initial_condition_sha256(batch.initial_states),
        },
        "summary": {
            "planned_cases": len(batch.initial_states),
            "executed_cases": len(batch.cases),
            "solver_successes": batch.successful_cases,
            "solver_failures": len(failures),
            "filter_rejections": len(result.rejections),
            "accepted_trajectories": result.accepted_trajectories,
            "acceptance_rate": result.accepted_trajectories / len(batch.initial_states),
            "total_attempts": len(attempts),
            "retried_cases": sum(case["attempt_count"] > 1 for case in batch.cases),
            "timeout_attempts": sum(attempt["status"] == "timeout" for attempt in attempts),
            "wall_time_s": batch.wall_time_s,
            "mean_wall_time_per_case_s": batch.wall_time_s / len(batch.cases),
        },
        "filters": {
            "rules": [
                "duplicate initial-condition or trajectory ID",
                "NaN or infinite state, action, time, duration, or quality metric",
                "malformed state-action trajectory or time grid",
                "solver, replay, terminal, state, or actuator constraint violation",
            ],
            "rejection_counts": rejection_counts,
            "rejections": list(result.rejections),
        },
        "progress": {
            "measurement": "elapsed wall time and rolling whole-run ETA",
            "interval_cases": result.settings.progress_interval,
            "samples": progress_measurements,
        },
        "shard": {
            "file": f"{result.settings.split}-trajectories.npz",
            "sha256": packed_shard_sha256(result.arrays),
            "validation": shard_validation,
            "arrays": {
                name: {"shape": list(values.shape), "dtype": str(values.dtype)}
                for name, values in result.arrays.items()
            },
        },
        "completion_gate": {
            "criterion": (
                "execute the complete easy split and retain at least "
                f"{result.settings.minimum_validated_trajectories} verified trajectories"
            ),
            "checks": completion_checks,
            "passed": all(completion_checks.values()),
        },
        "cases": list(batch.cases),
        "solver_failures": failures,
    }
