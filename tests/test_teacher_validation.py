"""Day 21 frozen protocol, bundle integrity, selection, and PID diagnostics."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import (
    frozen_teacher_initial_states,
    initial_condition_sha256,
    load_config,
    sample_integrated_initial_states,
    teacher_configuration_sha256,
)
from powered_landing_guidance.teacher_validation import (
    TeacherBundle,
    evaluate_pid_on_states,
    load_teacher_bundle,
    select_representative_case,
)
from scripts.validate_teacher import artifact_paths, ensure_outputs_available

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def _small_config() -> dict:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 2
    config["teacher_pipeline"].update(episodes=2, minimum_cases=2, seed=17)
    initial_states = sample_integrated_initial_states(config, episodes=2, seed=17)
    config["teacher_protocol"]["nominal_test_set"].update(
        episodes=2,
        seed=17,
        sha256=initial_condition_sha256(initial_states),
    )
    config["teacher_protocol"]["configuration_sha256"] = teacher_configuration_sha256(config)
    return config


def _write_small_bundle(config: dict, tmp_path: Path) -> tuple[Path, Path]:
    initial_states = frozen_teacher_initial_states(config)
    case_indices = np.arange(2, dtype=np.int64)
    times = np.tile(np.asarray((0.0, 2.5, 5.0)), (2, 1))
    states = np.repeat(initial_states[:, None, :], 3, axis=1)
    controls = np.zeros((2, 2, 2), dtype=np.float64)
    durations = np.full(2, 5.0)
    arrays = {
        "case_indices": case_indices,
        "initial_states": initial_states,
        "times_s": times,
        "states_x_z_vx_vz_theta_omega_mass": states,
        "controls_throttle_gimbal_rad": controls,
        "durations_s": durations,
    }
    cases = [
        {"case_index": index, "status": "success", "fuel_used_kg": 20.0 + index}
        for index in range(2)
    ]
    report = {
        "schema_version": 1,
        "problem": "day20-multi-initial-condition-teacher-pipeline",
        "status": "complete",
        "batch": config["teacher_protocol"]["nominal_test_set"],
        "summary": {"executed_cases": 2, "successful_cases": 2, "failed_cases": 0},
        "cases": cases,
        "failures": [],
        "dataset": {
            "file": "teacher-trajectories.npz",
            "successful_case_indices": [0, 1],
            "arrays": {
                name: {"shape": list(values.shape), "dtype": str(values.dtype)}
                for name, values in arrays.items()
            },
        },
    }
    report_path = tmp_path / "teacher-pipeline.json"
    dataset_path = tmp_path / "teacher-trajectories.npz"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(dataset_path, **arrays)
    config["teacher_protocol"]["source_artifacts"] = {
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    return report_path, dataset_path


def test_frozen_teacher_protocol_recreates_the_day20_batch() -> None:
    states = frozen_teacher_initial_states(CONFIG)
    protocol = CONFIG["teacher_protocol"]

    assert protocol["frozen"] is True
    assert states.shape == (100, 7)
    assert initial_condition_sha256(states) == protocol["nominal_test_set"]["sha256"]
    assert teacher_configuration_sha256(CONFIG) == protocol["configuration_sha256"]

    changed = deepcopy(CONFIG)
    changed["optimal_control"]["objective_weights"]["smoothness"] *= 2
    with pytest.raises(ValueError, match="settings have changed"):
        frozen_teacher_initial_states(changed)


def test_bundle_loader_checks_schema_metadata_and_frozen_states(tmp_path: Path) -> None:
    config = _small_config()
    report_path, dataset_path = _write_small_bundle(config, tmp_path)
    bundle = load_teacher_bundle(config, report_path, dataset_path)

    assert bundle.case_indices.tolist() == [0, 1]
    assert bundle.controls.shape == (2, 2, 2)
    assert bundle.states.flags.writeable is False

    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["initial_states"][0, 0] += 1.0
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="source dataset"):
        load_teacher_bundle(config, report_path, tampered)


def test_bundle_loader_requires_failed_cases_to_be_reported(tmp_path: Path) -> None:
    config = _small_config()
    report_path, dataset_path = _write_small_bundle(config, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cases"][1]["status"] = "failed"
    report["summary"].update(successful_cases=1, failed_cases=1)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config["teacher_protocol"]["source_artifacts"]["report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="account for every failed case"):
        load_teacher_bundle(config, report_path, dataset_path)


def test_representative_selection_uses_median_fuel_then_lowest_index() -> None:
    report = {
        "cases": [
            {"case_index": 7, "fuel_used_kg": 10.0},
            {"case_index": 9, "fuel_used_kg": 20.0},
        ]
    }
    bundle = TeacherBundle(
        report=report,
        case_indices=np.asarray((7, 9), dtype=np.int64),
        initial_states=np.zeros((2, 7)),
        times_s=np.zeros((2, 2)),
        states=np.zeros((2, 2, 7)),
        controls=np.zeros((2, 1, 2)),
        durations_s=np.ones(2),
    )

    selection = select_representative_case(bundle)

    assert selection["median_fuel_used_kg"] == 15.0
    assert selection["case_index"] == 7


def test_pid_evaluation_records_case_level_constraint_violations() -> None:
    config = deepcopy(CONFIG)
    initial_states = np.asarray(((0.0, 0.001, 0.0, -0.1, 0.0, 0.0, 900.0),))

    result = evaluate_pid_on_states(config, initial_states)

    assert result["episodes"] == result["successes"] == 1
    assert result["outcome_counts"] == {"success": 1}
    assert result["terminal_constraint_analysis"]["cases_with_any_violation"] == 0
    assert result["controller_compute_time_s"] >= 0.0


def test_day21_command_has_fixed_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    destinations = artifact_paths(tmp_path)

    assert set(destinations) == {"report", "representative_log", "representative_gif"}
    ensure_outputs_available(destinations)
    destinations["representative_gif"].write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_outputs_available(destinations)
