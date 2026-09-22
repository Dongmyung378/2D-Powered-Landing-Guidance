"""Day 21 frozen-teacher validation and comparison against the nominal PID baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.controllers import IntegratedLandingController
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    applied_thrust_n,
    simulate_planar,
)
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.evaluation import (
    frozen_teacher_initial_states,
    initial_condition_sha256,
    teacher_configuration_sha256,
)
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    State,
)
from powered_landing_guidance.optimal_control import (
    TERMINAL_RESIDUAL_NAMES,
    LandingOptimalControlProblem,
)

DATASET_KEYS = {
    "case_indices",
    "initial_states",
    "times_s",
    "states_x_z_vx_vz_theta_omega_mass",
    "controls_throttle_gimbal_rad",
    "durations_s",
}
REPLAY_CHECK_NAMES = (
    "final_state_agreement",
    "terminal_constraints",
    "altitude",
    "propellant_reserve",
    "tilt",
    "angular_rate",
    "throttle",
    "gimbal",
)


@dataclass(frozen=True, slots=True)
class TeacherBundle:
    """Validated Day 20 report and numeric trajectory archive."""

    report: dict[str, Any]
    case_indices: NDArray[np.int64]
    initial_states: NDArray[np.float64]
    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    durations_s: NDArray[np.float64]


def _finite_array(array: NDArray[Any], name: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"teacher dataset {name} must contain finite numeric values")


def _expected_dataset_shapes(intervals: int, successes: int) -> dict[str, tuple[int, ...]]:
    return {
        "case_indices": (successes,),
        "initial_states": (successes, 7),
        "times_s": (successes, intervals + 1),
        "states_x_z_vx_vz_theta_omega_mass": (successes, intervals + 1, 7),
        "controls_throttle_gimbal_rad": (successes, intervals, 2),
        "durations_s": (successes,),
    }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_bundle(
    config: dict[str, Any],
    report_path: str | Path,
    dataset_path: str | Path,
) -> TeacherBundle:
    """Load the Day 20 outputs and reject schema, set, or numeric inconsistencies."""
    expected_initial_states = frozen_teacher_initial_states(config)
    expected_hash = initial_condition_sha256(expected_initial_states)
    source_artifacts = config["teacher_protocol"]["source_artifacts"]
    if _file_sha256(report_path) != source_artifacts["report_sha256"]:
        raise ValueError("teacher source report does not match its frozen SHA-256")
    if _file_sha256(dataset_path) != source_artifacts["dataset_sha256"]:
        raise ValueError("teacher source dataset does not match its frozen SHA-256")
    with Path(report_path).open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("teacher report must use schema_version 1")
    if report.get("problem") != "day20-multi-initial-condition-teacher-pipeline":
        raise ValueError("teacher report problem identifier is invalid")

    protocol = config["teacher_protocol"]
    conditions = protocol["nominal_test_set"]
    batch = report.get("batch")
    expected_batch = {
        "sampler": conditions["sampler"],
        "episodes": conditions["episodes"],
        "seed": conditions["seed"],
        "sha256": expected_hash,
    }
    if batch != expected_batch:
        raise ValueError("teacher report batch does not match the frozen nominal test set")

    cases = report.get("cases")
    failures = report.get("failures")
    summary = report.get("summary")
    if not isinstance(cases, list) or len(cases) != len(expected_initial_states):
        raise ValueError("teacher report must contain one record per nominal case")
    if not isinstance(failures, list) or not isinstance(summary, dict):
        raise ValueError("teacher report is missing summary or failure records")
    case_indices = [case.get("case_index") for case in cases if isinstance(case, dict)]
    if case_indices != list(range(len(expected_initial_states))):
        raise ValueError("teacher report cases must be complete and ordered")
    successful_indices = [case["case_index"] for case in cases if case.get("status") == "success"]
    failed_indices = [case["case_index"] for case in cases if case.get("status") != "success"]
    reported_failed_indices = [
        case.get("case_index") for case in failures if isinstance(case, dict)
    ]
    if reported_failed_indices != failed_indices:
        raise ValueError("teacher report failures do not account for every failed case")
    if (
        summary.get("executed_cases") != len(cases)
        or summary.get("successful_cases") != len(successful_indices)
        or summary.get("failed_cases") != len(failed_indices)
    ):
        raise ValueError("teacher report summary counts are inconsistent")

    with np.load(dataset_path, allow_pickle=False) as archive:
        if set(archive.files) != DATASET_KEYS:
            raise ValueError("teacher dataset fields do not match the supported schema")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    intervals = int(config["optimal_control"]["intervals"])
    shapes = _expected_dataset_shapes(intervals, len(successful_indices))
    for name, expected_shape in shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"teacher dataset {name} must have shape {expected_shape}, got {arrays[name].shape}"
            )
        _finite_array(arrays[name], name)
    if arrays["case_indices"].dtype.kind not in "iu":
        raise ValueError("teacher dataset case_indices must be integers")
    archived_indices = arrays["case_indices"].astype(np.int64, copy=False)
    if archived_indices.tolist() != successful_indices:
        raise ValueError("teacher dataset case indices do not match successful report cases")
    if not np.array_equal(arrays["initial_states"], expected_initial_states[archived_indices]):
        raise ValueError("teacher dataset initial states do not match the frozen nominal set")
    if not np.array_equal(
        arrays["states_x_z_vx_vz_theta_omega_mass"][:, 0], arrays["initial_states"]
    ):
        raise ValueError("teacher trajectories do not begin at their declared initial states")
    if not np.all(arrays["times_s"][:, 0] == 0.0):
        raise ValueError("teacher trajectory times must start at zero")
    if not np.all(np.diff(arrays["times_s"], axis=1) > 0.0):
        raise ValueError("teacher trajectory times must increase strictly")
    if not np.allclose(arrays["times_s"][:, -1], arrays["durations_s"], rtol=0.0, atol=1e-12):
        raise ValueError("teacher trajectory durations do not match their final time nodes")

    dataset_metadata = report.get("dataset")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("teacher report is missing dataset metadata")
    if dataset_metadata.get("successful_case_indices") != successful_indices:
        raise ValueError("teacher report dataset indices are inconsistent")
    metadata_arrays = dataset_metadata.get("arrays")
    if not isinstance(metadata_arrays, dict) or set(metadata_arrays) != DATASET_KEYS:
        raise ValueError("teacher report dataset array metadata is invalid")
    for name, values in arrays.items():
        expected_metadata = {"shape": list(values.shape), "dtype": str(values.dtype)}
        if metadata_arrays.get(name) != expected_metadata:
            raise ValueError(f"teacher report metadata for {name} is inconsistent")

    parameters = PlanarDynamicsParameters.from_config(config)
    controls = arrays["controls_throttle_gimbal_rad"]
    tolerance = 1e-12
    if np.any(controls[:, :, 0] < parameters.throttle_min - tolerance) or np.any(
        controls[:, :, 0] > parameters.throttle_max + tolerance
    ):
        raise ValueError("teacher dataset throttle exceeds the frozen actuator bounds")
    if np.any(np.abs(controls[:, :, 1]) > parameters.gimbal_limit_rad + tolerance):
        raise ValueError("teacher dataset gimbal exceeds the frozen actuator bounds")

    for array in arrays.values():
        array.flags.writeable = False
    return TeacherBundle(
        report=report,
        case_indices=archived_indices,
        initial_states=arrays["initial_states"],
        times_s=arrays["times_s"],
        states=arrays["states_x_z_vx_vz_theta_omega_mass"],
        controls=controls,
        durations_s=arrays["durations_s"],
    )


def _replay_tolerances(config: dict[str, Any]) -> NDArray[np.float64]:
    values = config["optimal_control"]["study"]["final_state_tolerances"]
    return np.asarray(
        (
            values["x_m"],
            values["z_m"],
            values["vx_m_s"],
            values["vz_m_s"],
            np.deg2rad(values["theta_deg"]),
            np.deg2rad(values["omega_deg_s"]),
            values["mass_kg"],
        ),
        dtype=np.float64,
    )


def _simulate_control_sequence(
    problem: LandingOptimalControlProblem,
    controls: NDArray[np.float64],
    duration_s: float,
    replay_dt_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    interval_s = duration_s / len(controls)
    current = problem.initial_state.as_array()
    times = [0.0]
    states = [current.copy()]
    node_states = [current.copy()]
    for interval_index, control in enumerate(controls):
        local_times, local_states = simulate_planar(
            State.from_array(current),
            control,
            problem.parameters,
            duration_s=interval_s,
            dt_s=replay_dt_s,
        )
        times.extend((interval_index * interval_s + local_times[1:]).tolist())
        states.extend(local_states[1:].tolist())
        current = local_states[-1]
        node_states.append(current.copy())
    return np.asarray(times), np.asarray(states), np.asarray(node_states)


def revalidate_teacher_bundle(
    config: dict[str, Any],
    bundle: TeacherBundle,
) -> dict[str, Any]:
    """Replay every accepted control sequence and recompute all acceptance checks."""
    replay_dt_s = float(config["teacher_pipeline"]["replay_dt_s"])
    tolerances = _replay_tolerances(config)
    cases: list[dict[str, Any]] = []
    worst_final_ratio = 0.0
    worst_node_ratio = 0.0
    worst_terminal_violation = 0.0
    check_failures: Counter[str] = Counter()
    minimum_margins = {
        "altitude_m": np.inf,
        "propellant_reserve_kg": np.inf,
        "tilt_deg": np.inf,
        "angular_rate_deg_s": np.inf,
        "throttle_lower": np.inf,
        "throttle_upper": np.inf,
        "gimbal_deg": np.inf,
    }

    for row, case_index in enumerate(bundle.case_indices):
        problem = LandingOptimalControlProblem.from_config(config, bundle.initial_states[row])
        _, replay_states, node_states = _simulate_control_sequence(
            problem,
            bundle.controls[row],
            float(bundle.durations_s[row]),
            replay_dt_s,
        )
        final_error = np.abs(replay_states[-1] - bundle.states[row, -1])
        node_error = np.max(np.abs(node_states - bundle.states[row]), axis=0)
        terminal_violations = problem.terminal_violations(replay_states[-1])
        controls = bundle.controls[row]
        p = problem.parameters
        checks = {
            "final_state_agreement": bool(np.all(final_error <= tolerances)),
            "terminal_constraints": bool(
                np.max(terminal_violations) <= problem.feasibility_tolerance
            ),
            "altitude": bool(np.min(replay_states[:, 1]) >= problem.ground_z_m - tolerances[1]),
            "propellant_reserve": bool(
                np.min(replay_states[:, 6])
                >= p.dry_mass_kg + problem.min_propellant_reserve_kg - tolerances[6]
            ),
            "tilt": bool(
                np.max(np.abs(replay_states[:, 4])) <= problem.max_abs_tilt_rad + tolerances[4]
            ),
            "angular_rate": bool(
                np.max(np.abs(replay_states[:, 5]))
                <= problem.max_abs_angular_rate_rad_s + tolerances[5]
            ),
            "throttle": bool(
                np.min(controls[:, 0]) >= p.throttle_min - 1e-12
                and np.max(controls[:, 0]) <= p.throttle_max + 1e-12
            ),
            "gimbal": bool(np.max(np.abs(controls[:, 1])) <= p.gimbal_limit_rad + 1e-12),
        }
        for name, passed in checks.items():
            if not passed:
                check_failures[name] += 1
        final_ratio = float(np.max(final_error / tolerances))
        node_ratio = float(np.max(node_error / tolerances))
        max_terminal_violation = float(np.max(terminal_violations))
        worst_final_ratio = max(worst_final_ratio, final_ratio)
        worst_node_ratio = max(worst_node_ratio, node_ratio)
        worst_terminal_violation = max(worst_terminal_violation, max_terminal_violation)
        minimum_margins["altitude_m"] = min(
            minimum_margins["altitude_m"],
            float(np.min(replay_states[:, 1] - problem.ground_z_m)),
        )
        minimum_margins["propellant_reserve_kg"] = min(
            minimum_margins["propellant_reserve_kg"],
            float(np.min(replay_states[:, 6] - p.dry_mass_kg - problem.min_propellant_reserve_kg)),
        )
        minimum_margins["tilt_deg"] = min(
            minimum_margins["tilt_deg"],
            float(np.rad2deg(problem.max_abs_tilt_rad - np.max(np.abs(replay_states[:, 4])))),
        )
        minimum_margins["angular_rate_deg_s"] = min(
            minimum_margins["angular_rate_deg_s"],
            float(
                np.rad2deg(problem.max_abs_angular_rate_rad_s - np.max(np.abs(replay_states[:, 5])))
            ),
        )
        minimum_margins["throttle_lower"] = min(
            minimum_margins["throttle_lower"],
            float(np.min(controls[:, 0] - p.throttle_min)),
        )
        minimum_margins["throttle_upper"] = min(
            minimum_margins["throttle_upper"],
            float(np.min(p.throttle_max - controls[:, 0])),
        )
        minimum_margins["gimbal_deg"] = min(
            minimum_margins["gimbal_deg"],
            float(np.rad2deg(p.gimbal_limit_rad - np.max(np.abs(controls[:, 1])))),
        )
        cases.append(
            {
                "case_index": int(case_index),
                "passed": all(checks.values()),
                "checks": checks,
                "max_final_error_ratio": final_ratio,
                "max_node_error_ratio": node_ratio,
                "max_terminal_violation": max_terminal_violation,
                "terminal_violations": dict(
                    zip(TERMINAL_RESIDUAL_NAMES, terminal_violations.tolist(), strict=True)
                ),
            }
        )

    passed_cases = sum(case["passed"] for case in cases)
    return {
        "replay_dt_s": replay_dt_s,
        "accepted_cases": len(cases),
        "passed_cases": passed_cases,
        "all_passed": passed_cases == len(cases),
        "check_failure_counts": {name: check_failures[name] for name in REPLAY_CHECK_NAMES},
        "worst_final_error_ratio": worst_final_ratio,
        "worst_node_error_ratio": worst_node_ratio,
        "worst_terminal_violation": worst_terminal_violation,
        "minimum_path_margins": minimum_margins,
        "cases": cases,
    }


def _terminal_violations(config: dict[str, Any], state: NDArray[np.float64]) -> dict[str, float]:
    landing = config["landing_success"]
    target_x = float(config["integrated_landing_controller"]["target_x_m"])
    ground = float(config["simulation"]["ground_z_m"])
    theta = float(np.arctan2(np.sin(state[4]), np.cos(state[4])))
    return {
        "x_m": max(0.0, abs(float(state[0]) - target_x) - float(landing["max_abs_x_m"])),
        "z_m": abs(float(state[1]) - ground),
        "vx_m_s": max(0.0, abs(float(state[2])) - float(landing["max_abs_vx_m_s"])),
        "vz_m_s": max(
            0.0,
            -float(landing["max_abs_vz_m_s"]) - float(state[3]),
            float(state[3]),
        ),
        "theta_deg": max(
            0.0,
            abs(float(np.rad2deg(theta))) - float(landing["max_abs_theta_deg"]),
        ),
        "omega_deg_s": max(
            0.0,
            abs(float(np.rad2deg(state[5]))) - float(landing["max_abs_omega_deg_s"]),
        ),
        "mass_kg": max(0.0, float(config["vehicle"]["dry_mass_kg"]) - float(state[6])),
    }


def evaluate_pid_on_states(
    config: dict[str, Any],
    initial_states: NDArray[np.float64],
) -> dict[str, Any]:
    """Evaluate the integrated PID on an explicit nominal set with per-case diagnostics."""
    states = np.asarray(initial_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    if not np.all(np.isfinite(states)):
        raise ValueError("initial_states must contain finite values")
    if not np.isclose(
        float(config["environment"]["horizontal_wind_m_s"]), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Day 21 nominal PID evaluation requires zero horizontal wind")

    env = RocketLandingEnv(config)
    cases: list[dict[str, Any]] = []
    controller_compute_s = 0.0
    evaluation_started = perf_counter()
    try:
        for case_index, initial_state in enumerate(states):
            controller = IntegratedLandingController.from_config(config)
            state, info = env.reset(options={"initial_state": initial_state})
            while True:
                command_started = perf_counter()
                action = controller.command(state, time_s=float(info["time_s"]))
                controller_compute_s += perf_counter() - command_started
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            violations = _terminal_violations(config, np.asarray(state, dtype=np.float64))
            cases.append(
                {
                    "case_index": case_index,
                    "outcome": str(info["outcome"]),
                    "success": info["outcome"] == "success",
                    "final_state": dict(zip(STATE_NAMES, state.tolist(), strict=True)),
                    "fuel_used_kg": float(info["fuel_used_kg"]),
                    "flight_time_s": float(info["time_s"]),
                    "control_updates": controller.update_count,
                    "terminal_violations": violations,
                }
            )
    finally:
        env.close()
    wall_time_s = perf_counter() - evaluation_started

    outcomes = Counter(case["outcome"] for case in cases)
    successful = [case for case in cases if case["success"]]
    violation_names = tuple(cases[0]["terminal_violations"])
    violation_counts = {
        name: sum(case["terminal_violations"][name] > 0.0 for case in cases)
        for name in violation_names
    }
    maximum_violations = {
        name: max(case["terminal_violations"][name] for case in cases) for name in violation_names
    }
    total_updates = sum(case["control_updates"] for case in cases)
    return {
        "episodes": len(cases),
        "successes": len(successful),
        "success_rate": len(successful) / len(cases),
        "outcome_counts": dict(sorted(outcomes.items())),
        "mean_fuel_used_kg_all_cases": float(np.mean([case["fuel_used_kg"] for case in cases])),
        "mean_fuel_used_kg_successes": (
            float(np.mean([case["fuel_used_kg"] for case in successful])) if successful else None
        ),
        "mean_flight_time_s": float(np.mean([case["flight_time_s"] for case in cases])),
        "wall_time_s": wall_time_s,
        "controller_compute_time_s": controller_compute_s,
        "mean_controller_compute_ms_per_update": (
            1000.0 * controller_compute_s / total_updates if total_updates else 0.0
        ),
        "terminal_constraint_analysis": {
            "cases_with_any_violation": sum(
                any(value > 0.0 for value in case["terminal_violations"].values()) for case in cases
            ),
            "violation_counts": violation_counts,
            "maximum_physical_violations": maximum_violations,
        },
        "cases": cases,
    }


def select_representative_case(bundle: TeacherBundle) -> dict[str, Any]:
    """Select the accepted trajectory closest to the median teacher fuel use."""
    if len(bundle.case_indices) == 0:
        raise ValueError("cannot select a representative case from an empty teacher dataset")
    cases_by_index = {case["case_index"]: case for case in bundle.report["cases"]}
    fuel = np.asarray(
        [float(cases_by_index[int(index)]["fuel_used_kg"]) for index in bundle.case_indices]
    )
    median_fuel = float(np.median(fuel))
    order = np.lexsort((bundle.case_indices, np.abs(fuel - median_fuel)))
    row = int(order[0])
    return {
        "selection": "median_fuel",
        "dataset_row": row,
        "case_index": int(bundle.case_indices[row]),
        "median_fuel_used_kg": median_fuel,
        "selected_fuel_used_kg": float(fuel[row]),
    }


def build_teacher_episode(
    config: dict[str, Any],
    bundle: TeacherBundle,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Build a standard episode log from an independently replayed teacher trajectory."""
    row = int(selection["dataset_row"])
    case_index = int(selection["case_index"])
    if row < 0 or row >= len(bundle.case_indices) or int(bundle.case_indices[row]) != case_index:
        raise ValueError("representative selection does not match the teacher dataset")
    problem = LandingOptimalControlProblem.from_config(config, bundle.initial_states[row])
    replay_dt_s = float(config["teacher_pipeline"]["replay_dt_s"])
    interval_s = float(bundle.durations_s[row]) / len(bundle.controls[row])
    current = bundle.initial_states[row].copy()
    parameters = problem.parameters
    initial_mass = float(current[6])
    steps: list[dict[str, Any]] = []
    absolute_time = 0.0
    for interval_index, control in enumerate(bundle.controls[row]):
        local_times, local_states = simulate_planar(
            State.from_array(current),
            control,
            parameters,
            duration_s=interval_s,
            dt_s=replay_dt_s,
        )
        previous = current
        for local_index in range(1, len(local_times)):
            state = local_states[local_index]
            time_s = interval_index * interval_s + float(local_times[local_index])
            steps.append(
                {
                    "index": len(steps),
                    "state": state.tolist(),
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "outcome": "running",
                    "is_success": False,
                    "time_s": time_s,
                    "fuel_used_kg": initial_mass - float(state[6]),
                    "commanded_action": control.tolist(),
                    "clipped_action": control.tolist(),
                    "thrust_start_n": applied_thrust_n(
                        State.from_array(previous), control, parameters
                    ),
                    "thrust_end_n": applied_thrust_n(State.from_array(state), control, parameters),
                }
            )
            previous = state
            absolute_time = time_s
        current = local_states[-1]
    if not steps:
        raise ValueError("representative teacher trajectory contains no replay steps")
    replay_states = np.asarray(
        (bundle.initial_states[row].tolist(), *(step["state"] for step in steps)),
        dtype=np.float64,
    )
    tolerances = _replay_tolerances(config)
    if np.any(np.abs(current - bundle.states[row, -1]) > tolerances):
        raise ValueError("representative replay does not agree with the saved final state")
    if np.max(problem.terminal_violations(current)) > problem.feasibility_tolerance:
        raise ValueError("representative replay does not satisfy the landing constraints")
    if np.min(replay_states[:, 1]) < problem.ground_z_m - tolerances[1]:
        raise ValueError("representative replay crosses below the allowed ground tolerance")
    if np.min(replay_states[:, 6]) < (
        parameters.dry_mass_kg + problem.min_propellant_reserve_kg - tolerances[6]
    ):
        raise ValueError("representative replay violates the propellant reserve")
    if np.max(np.abs(replay_states[:, 4])) > problem.max_abs_tilt_rad + tolerances[4]:
        raise ValueError("representative replay violates the body-tilt limit")
    if np.max(np.abs(replay_states[:, 5])) > problem.max_abs_angular_rate_rad_s + tolerances[5]:
        raise ValueError("representative replay violates the angular-rate limit")
    steps[-1].update(reward=1.0, terminated=True, outcome="success", is_success=True)
    episode_config = deepcopy(config)
    episode_config["simulation"]["dt_s"] = replay_dt_s
    episode_config["simulation"]["max_time_s"] = max(
        float(episode_config["simulation"]["max_time_s"]), absolute_time + replay_dt_s
    )
    return {
        "schema_version": 1,
        "source": "day21-frozen-planar-teacher-replay",
        "teacher_protocol_id": config["teacher_protocol"]["id"],
        "case_index": case_index,
        "config": episode_config,
        "seed_argument": None,
        "state_names": list(STATE_NAMES),
        "state_units": list(STATE_UNITS),
        "action_names": list(ACTION_NAMES),
        "action_units": list(ACTION_UNITS),
        "initial_state": bundle.initial_states[row].tolist(),
        "steps": steps,
        "outcome": "success",
    }


def build_teacher_validation_report(
    config: dict[str, Any],
    bundle: TeacherBundle,
    replay: dict[str, Any],
    pid: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the Day 21 comparison, failure accounting, and Week 3 gate."""
    source = bundle.report
    source_cases = source["cases"]
    successful_teacher = [case for case in source_cases if case["status"] == "success"]
    failed_teacher = [case for case in source_cases if case["status"] != "success"]
    attempts = [attempt for case in source_cases for attempt in case["attempts"]]
    accepted_attempts = [
        attempt
        for case in successful_teacher
        for attempt in case["attempts"]
        if attempt["status"] == "accepted"
    ]
    teacher_fuel = {case["case_index"]: float(case["fuel_used_kg"]) for case in successful_teacher}
    paired = [
        (teacher_fuel[case["case_index"]], float(case["fuel_used_kg"]))
        for case in pid["cases"]
        if case["success"] and case["case_index"] in teacher_fuel
    ]
    paired_teacher_fuel = [values[0] for values in paired]
    paired_pid_fuel = [values[1] for values in paired]
    stored_replay_passed = all(
        case.get("replay_validation", {}).get("passed") is True for case in successful_teacher
    )
    failure_indices = [case["case_index"] for case in failed_teacher]
    reported_failure_indices = [case["case_index"] for case in source["failures"]]
    failures_accounted = failure_indices == reported_failure_indices
    protocol = config["teacher_protocol"]
    minimum_success_rate = float(protocol["acceptance"]["minimum_solver_success_rate"])
    solver_success_rate = len(successful_teacher) / len(source_cases)
    gates = {
        "frozen_configuration_verified": (
            teacher_configuration_sha256(config) == protocol["configuration_sha256"]
        ),
        "nominal_test_set_verified": (
            initial_condition_sha256(frozen_teacher_initial_states(config))
            == protocol["nominal_test_set"]["sha256"]
        ),
        "minimum_solver_success_rate": solver_success_rate >= minimum_success_rate,
        "stored_replay_checks": stored_replay_passed,
        "independent_simulator_replays": bool(replay["all_passed"]),
        "failed_cases_reported_separately": failures_accounted,
    }
    stage_b_hard_violation = max(
        (float(case["stage_b"]["max_hard_violation"]) for case in successful_teacher),
        default=0.0,
    )
    source_wall_time_s = float(source["summary"]["wall_time_s"])
    total_attempt_time_s = float(sum(float(attempt["elapsed_s"]) for attempt in attempts))
    total_accepted_attempt_time_s = float(
        sum(float(attempt["elapsed_s"]) for attempt in accepted_attempts)
    )
    report = {
        "schema_version": 1,
        "problem": "day21-teacher-validation-and-freeze",
        "protocol": {
            "id": protocol["id"],
            "teacher": protocol["teacher"],
            "frozen": protocol["frozen"],
            "configuration_sha256": protocol["configuration_sha256"],
            "nominal_test_set": protocol["nominal_test_set"],
            "source_artifacts": protocol["source_artifacts"],
        },
        "teacher": {
            "executed_cases": len(source_cases),
            "successful_cases": len(successful_teacher),
            "failed_cases": len(failed_teacher),
            "success_rate": solver_success_rate,
            "outcome": source.get("status"),
            "mean_fuel_used_kg": float(
                np.mean([case["fuel_used_kg"] for case in successful_teacher])
            ),
            "fuel_used_range_kg": [
                float(min(case["fuel_used_kg"] for case in successful_teacher)),
                float(max(case["fuel_used_kg"] for case in successful_teacher)),
            ],
            "source_pipeline_wall_time_s": source_wall_time_s,
            "total_attempt_time_s": total_attempt_time_s,
            "total_accepted_attempt_time_s": total_accepted_attempt_time_s,
            "mean_accepted_attempt_time_s": (
                total_accepted_attempt_time_s / len(accepted_attempts)
                if accepted_attempts
                else None
            ),
            "maximum_stage_b_hard_violation": stage_b_hard_violation,
            "independent_replay": replay,
            "failed_optimization_cases": source["failures"],
        },
        "pid": pid,
        "comparison": {
            "same_initial_condition_cases": len(source_cases) == pid["episodes"],
            "teacher_minus_pid_success_rate": solver_success_rate - float(pid["success_rate"]),
            "paired_success_cases": len(paired),
            "mean_teacher_fuel_on_pid_successes_kg": (
                float(np.mean(paired_teacher_fuel)) if paired else None
            ),
            "mean_pid_fuel_on_successes_kg": (float(np.mean(paired_pid_fuel)) if paired else None),
            "mean_paired_fuel_delta_teacher_minus_pid_kg": (
                float(np.mean(np.asarray(paired_teacher_fuel) - np.asarray(paired_pid_fuel)))
                if paired
                else None
            ),
            "timing_note": (
                "Teacher time is offline solver-process time; PID controller time measures only "
                "online command computation. They are reported separately, not as equivalent "
                "clocks."
            ),
        },
        "representative_case": selection,
        "week3_acceptance": {
            "minimum_solver_success_rate": minimum_success_rate,
            "checks": gates,
            "passed": all(gates.values()),
        },
    }
    return report
