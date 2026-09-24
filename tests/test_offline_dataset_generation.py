"""Day 23 Teacher generation, filtering, packing, timing, and CLI checks."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("casadi")

from powered_landing_guidance import load_config, validate_config  # noqa: E402
from powered_landing_guidance.offline_dataset import (  # noqa: E402
    initial_condition_id,
    sample_split_initial_states,
)
from powered_landing_guidance.offline_dataset_generation import (  # noqa: E402
    OfflineGenerationSettings,
    packed_shard_sha256,
    teacher_trajectory_rejection_reasons,
    trajectory_id,
    validate_packed_trajectory_shard,
)
from powered_landing_guidance.planar_optimal_control import (  # noqa: E402
    PlanarLandingResult,
    PlanarSolverStage,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def _solver_stage() -> PlanarSolverStage:
    return PlanarSolverStage(
        status="Solve_Succeeded",
        iterations=1,
        objective=0.0,
        max_hard_violation=0.0,
        max_terminal_violation=0.0,
        hard_violations={},
        terminal_violations={},
        max_rk4_defects={},
        worst_constraint="none",
        worst_violation=0.0,
    )


def _valid_result_and_case() -> tuple[np.ndarray, PlanarLandingResult, dict[str, object]]:
    initial_state = sample_split_initial_states(CONFIG, "train")[0].copy()
    intervals = int(CONFIG["optimal_control"]["intervals"])
    duration_s = 5.0
    final_state = np.asarray(
        (
            CONFIG["integrated_landing_controller"]["target_x_m"],
            CONFIG["simulation"]["ground_z_m"],
            0.0,
            -1.0,
            0.0,
            0.0,
            initial_state[6] - 20.0,
        ),
        dtype=np.float64,
    )
    blend = np.linspace(0.0, 1.0, intervals + 1)[:, None]
    states = initial_state + blend * (final_state - initial_state)
    times = np.linspace(0.0, duration_s, intervals + 1)
    controls = np.zeros((intervals, 2), dtype=np.float64)
    stage = _solver_stage()
    result = PlanarLandingResult(
        times_s=times,
        states=states,
        controls=controls,
        duration_s=duration_s,
        guess_source="pid",
        source_outcome="landed",
        stage_a=stage,
        stage_b=stage,
        rollout_times_s=times.copy(),
        rollout_states=states.copy(),
    )
    case: dict[str, object] = {
        "case_index": 0,
        "status": "success",
        "attempt_count": 1,
        "fuel_used_kg": 20.0,
        "replay_validation": {
            "passed": True,
            "checks": {
                "final_state_agreement": True,
                "terminal_constraints": True,
                "altitude": True,
                "propellant_reserve": True,
                "tilt": True,
                "angular_rate": True,
                "gimbal": True,
            },
            "max_final_error_ratio": 0.0,
            "max_node_error_ratio": 0.0,
            "max_terminal_violation": 0.0,
        },
    }
    return initial_state, result, case


def _run_cli(config: dict, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    config_path = tmp_path / "day23.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "day23"
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_offline_dataset.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return completed, output_dir


def test_generation_policy_and_trajectory_identity_are_frozen() -> None:
    settings = OfflineGenerationSettings.from_config(CONFIG)
    initial_state = sample_split_initial_states(CONFIG, "train")[0]
    initial_id_value = initial_condition_id(initial_state)

    assert settings.split == "train"
    assert settings.minimum_validated_trajectories == 500
    assert settings.attempt_timeout_s == pytest.approx(30.0)
    assert settings.max_retries == 1
    assert settings.warm_start == "previous_success"
    assert settings.retry_initial_guess == "pid"
    assert settings.progress_interval == 10
    first = trajectory_id(CONFIG, "train", initial_id_value)
    assert first == trajectory_id(CONFIG, "train", initial_id_value)
    assert len(first) == 64
    assert first != trajectory_id(CONFIG, "validation", initial_id_value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"minimum_validated_trajectories": 801}, "minimum cannot exceed"),
        ({"split": "test"}, "split must be 'train'"),
        ({"replay_dt_s": 0.02}, "replay_dt_s"),
        ({"progress_interval": 801}, "progress_interval"),
    ],
)
def test_config_rejects_invalid_generation_policy(mutation, message) -> None:
    config = deepcopy(CONFIG)
    config["offline_dataset_generation"].update(mutation)

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_filter_rejects_duplicate_nonfinite_and_constraint_violations() -> None:
    initial_state, result, case = _valid_result_and_case()
    initial_id_value = initial_condition_id(initial_state)
    trajectory_id_value = trajectory_id(CONFIG, "train", initial_id_value)

    accepted = teacher_trajectory_rejection_reasons(
        CONFIG,
        initial_state,
        initial_id_value,
        case,
        result,
        set(),
        set(),
        split="train",
    )
    assert accepted == ()

    duplicate = teacher_trajectory_rejection_reasons(
        CONFIG,
        initial_state,
        initial_id_value,
        case,
        result,
        {initial_id_value},
        {trajectory_id_value},
        split="train",
    )
    assert "duplicate_initial_condition" in duplicate
    assert "duplicate_trajectory" in duplicate

    states_with_nan = result.states.copy()
    states_with_nan[1, 0] = np.nan
    nonfinite = teacher_trajectory_rejection_reasons(
        CONFIG,
        initial_state,
        initial_id_value,
        case,
        replace(result, states=states_with_nan),
        set(),
        set(),
        split="train",
    )
    assert "nonfinite_trajectory" in nonfinite

    invalid_rollout = result.rollout_states.copy()
    invalid_rollout[-1, 0] = 10.0
    constraint = teacher_trajectory_rejection_reasons(
        CONFIG,
        initial_state,
        initial_id_value,
        case,
        replace(result, rollout_states=invalid_rollout),
        set(),
        set(),
        split="train",
    )
    assert "constraint_violation" in constraint


def test_cli_generates_valid_packed_shard_with_progress_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 30
    config["offline_dataset"]["splits"]["train"]["episodes"] = 3
    config["offline_dataset_generation"].update(
        {
            "minimum_validated_trajectories": 2,
            "worker_restart_after_attempts": 2,
            "progress_interval": 1,
        }
    )
    first, output_dir = _run_cli(config, tmp_path)

    assert first.returncode == 0, first.stderr
    report = json.loads((output_dir / "train-manifest.json").read_text(encoding="utf-8"))
    assert report["problem"] == "day23-large-scale-easy-teacher-generation"
    assert report["summary"]["executed_cases"] == 3
    assert report["summary"]["accepted_trajectories"] >= 2
    assert report["completion_gate"]["passed"] is True
    assert len(report["progress"]["samples"]) == 3
    assert report["progress"]["samples"][-1]["estimated_remaining_s"] == pytest.approx(0.0)
    with np.load(output_dir / "train-trajectories.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validation = validate_packed_trajectory_shard(arrays)
    assert validation["passed"] is True
    assert packed_shard_sha256(arrays) == report["shard"]["sha256"]

    duplicate = {name: values.copy() for name, values in arrays.items()}
    duplicate["initial_condition_ids"][1] = duplicate["initial_condition_ids"][0]
    assert (
        validate_packed_trajectory_shard(duplicate)["checks"]["unique_initial_conditions"] is False
    )
    nonfinite = {name: values.copy() for name, values in arrays.items()}
    nonfinite["states"][0, 0] = np.nan
    assert validate_packed_trajectory_shard(nonfinite)["checks"]["finite_values"] is False

    second, _ = _run_cli(config, tmp_path)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
