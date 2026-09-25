"""Day 24 hard sampling, coverage repair, final shards, and CLI checks."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("casadi")

from powered_landing_guidance import load_config, validate_config  # noqa: E402
from powered_landing_guidance.offline_dataset import (  # noqa: E402
    dataset_configuration_sha256,
    initial_condition_id,
)
from powered_landing_guidance.offline_dataset_finalization import (  # noqa: E402
    FinalDatasetSettings,
    analyze_coverage,
    compute_shard_normalization,
    difficulty_scores,
    final_test_difficulty_analysis,
    load_verified_source_train,
    merge_packed_shards,
    sample_coverage_states,
    sample_hard_training_states,
)
from powered_landing_guidance.offline_dataset_generation import (  # noqa: E402
    packed_shard_sha256,
    trajectory_id,
    validate_packed_trajectory_shard,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def _synthetic_shard(
    config: dict,
    split: str,
    initial_states: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(initial_states)
    state_blocks = []
    time_blocks = []
    for state in initial_states:
        final = np.asarray((0.0, 0.0, 0.0, -0.8, 0.0, 0.0, state[6] - 20.0))
        state_blocks.append(np.vstack((state, (state + final) / 2.0, final)))
        time_blocks.append(np.asarray((0.0, 1.0, 2.0)))
    initial_ids = np.asarray(
        [initial_condition_id(state) for state in initial_states], dtype="<U64"
    )
    arrays = {
        "trajectory_ids": np.asarray(
            [trajectory_id(config, split, identifier) for identifier in initial_ids], dtype="<U64"
        ),
        "initial_condition_ids": initial_ids,
        "source_case_indices": np.arange(count, dtype=np.int64),
        "state_offsets": np.arange(0, 3 * count + 1, 3, dtype=np.int64),
        "action_offsets": np.arange(0, 2 * count + 1, 2, dtype=np.int64),
        "times_s": np.concatenate(time_blocks),
        "states": np.concatenate(state_blocks),
        "actions": np.zeros((2 * count, 2), dtype=np.float64),
        "durations_s": np.full(count, 2.0, dtype=np.float64),
        "solver_success": np.ones(count, dtype=np.bool_),
        "stage_a_iterations": np.ones(count, dtype=np.int64),
        "stage_b_iterations": np.ones(count, dtype=np.int64),
        "max_hard_violation": np.zeros(count, dtype=np.float64),
        "max_terminal_violation": np.zeros(count, dtype=np.float64),
        "replay_passed": np.ones(count, dtype=np.bool_),
        "max_final_error_ratio": np.zeros(count, dtype=np.float64),
        "max_node_error_ratio": np.zeros(count, dtype=np.float64),
        "fuel_used_kg": np.full(count, 20.0, dtype=np.float64),
        "attempt_count": np.ones(count, dtype=np.int64),
    }
    assert validate_packed_trajectory_shard(arrays)["passed"] is True
    return arrays


def _source_states() -> np.ndarray:
    return np.asarray(
        (
            (-8.0, 75.0, -1.0, -18.0, np.deg2rad(-2.0), np.deg2rad(-1.0), 980.0),
            (5.0, 90.0, 0.0, -20.0, 0.0, 0.0, 990.0),
            (9.0, 102.0, 1.0, -16.0, np.deg2rad(2.0), np.deg2rad(1.0), 998.0),
        ),
        dtype=np.float64,
    )


def test_hard_oversampling_is_reproducible_and_test_remains_harder() -> None:
    first, first_report = sample_hard_training_states(CONFIG)
    second, second_report = sample_hard_training_states(CONFIG)
    settings = FinalDatasetSettings.from_config(CONFIG)
    ranges = settings.hard_oversampling["ranges"]

    np.testing.assert_array_equal(first, second)
    assert first_report == second_report
    assert first.shape == (160, 7)
    assert len(set(first_report["selected_initial_condition_ids"])) == 160
    assert first_report["candidate_count"] == 640
    assert first_report["hardest_selected_count"] == 120
    assert first_report["selected_score"]["mean"] > first_report["candidate_score"]["mean"]
    assert np.all(np.abs(first[:, 0]) >= ranges["min_abs_x_m"])
    assert np.all(np.abs(np.rad2deg(first[:, 4])) <= ranges["theta_deg"][1])
    assert final_test_difficulty_analysis(CONFIG)["passed"] is True


def test_difficulty_score_increases_for_jointly_harder_state() -> None:
    ranges = CONFIG["offline_dataset_finalization"]["hard_oversampling"]["ranges"]
    easy = np.asarray((8.0, 90.0, 0.0, -20.0, 0.0, 0.0, 980.0))
    hard = np.asarray((11.0, 108.0, 3.0, -23.5, np.deg2rad(7.0), np.deg2rad(3.0), 962.0))

    scores = difficulty_scores(np.vstack((easy, hard)), ranges)

    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(1.0)


def test_coverage_sampler_targets_deficient_bins_deterministically() -> None:
    repeated = np.repeat(_source_states()[1:2], 40, axis=0)
    coverage = analyze_coverage(CONFIG, repeated)

    first, first_plan = sample_coverage_states(CONFIG, repeated, round_index=0, maximum_count=20)
    second, second_plan = sample_coverage_states(CONFIG, repeated, round_index=0, maximum_count=20)

    assert coverage["passed"] is False
    assert coverage["total_deficit"] > 0
    assert first.shape == (20, 7)
    np.testing.assert_array_equal(first, second)
    assert first_plan == second_plan
    assert len({entry["initial_condition_id"] for entry in first_plan}) == 20
    assert {entry["feature"] for entry in first_plan} <= {
        "abs_x_m",
        "z_m",
        "vx_m_s",
        "vz_m_s",
        "theta_deg",
        "omega_deg_s",
        "mass_kg",
    }


def test_shard_merge_and_normalization_preserve_trajectory_alignment() -> None:
    states = _source_states()
    first = _synthetic_shard(CONFIG, "train", states[:2])
    second = _synthetic_shard(CONFIG, "train", states[2:])

    merged = merge_packed_shards((first, second))
    normalization = compute_shard_normalization(CONFIG, merged)

    assert validate_packed_trajectory_shard(merged)["passed"] is True
    assert len(merged["trajectory_ids"]) == 3
    assert normalization["fit_split"] == "train"
    assert normalization["state"]["count"] == 6
    assert normalization["action"]["count"] == 6

    duplicate = _synthetic_shard(CONFIG, "train", states[:1])
    with pytest.raises(ValueError, match="duplicate"):
        merge_packed_shards((first, duplicate))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("hard_oversampling", "candidate_multiplier"), 1, "candidate_multiplier"),
        (("hard_oversampling", "ranges", "theta_deg"), [-2.0, 2.0], "expand train"),
        (("coverage", "maximum_rounds"), 0, "positive integer"),
        (("completion", "minimum_test_trajectories"), 201, "minimum test"),
    ],
)
def test_config_rejects_invalid_finalization_policy(path, replacement, message) -> None:
    config = deepcopy(CONFIG)
    target = config["offline_dataset_finalization"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_source_train_verification_and_small_cli(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["optimal_control"]["intervals"] = 30
    config["offline_dataset"]["splits"]["validation"]["episodes"] = 2
    config["offline_dataset"]["splits"]["test"]["episodes"] = 2
    final = config["offline_dataset_finalization"]
    final["hard_oversampling"].update(
        {"episodes": 2, "candidate_multiplier": 2, "hardest_fraction": 0.5}
    )
    final["coverage"].update(
        {"bins_per_feature": 1, "minimum_count_per_bin": 1, "maximum_additional_cases": 2}
    )
    final["completion"].update(
        {
            "minimum_train_trajectories": 3,
            "minimum_validation_trajectories": 1,
            "minimum_test_trajectories": 1,
        }
    )
    validate_config(config)
    source_dir = tmp_path / "day23"
    source_dir.mkdir()
    source_arrays = _synthetic_shard(config, "train", _source_states())
    np.savez_compressed(source_dir / "train-trajectories.npz", **source_arrays)
    source_manifest = {
        "problem": "day23-large-scale-easy-teacher-generation",
        "split": "train",
        "dataset_configuration_sha256": dataset_configuration_sha256(config),
        "completion_gate": {"passed": True},
        "shard": {"sha256": packed_shard_sha256(source_arrays)},
    }
    (source_dir / "train-manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    loaded, source_report = load_verified_source_train(config, source_dir)
    assert source_report["passed"] is True
    assert packed_shard_sha256(loaded) == packed_shard_sha256(source_arrays)

    config_path = tmp_path / "day24.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "day24"
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_offline_dataset.py"),
        "--config",
        str(config_path),
        "--source-train-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
    ]
    first_run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert first_run.returncode == 0, first_run.stderr
    manifest = json.loads((output_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["completion_gate"]["passed"] is True
    assert manifest["split_integrity"]["passed"] is True
    assert manifest["normalization"]["fit_split"] == "train"
    assert (output_dir / "dataset-distribution.png").stat().st_size > 0
    for split in ("train", "validation", "test"):
        with np.load(output_dir / f"{split}-trajectories.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        assert validate_packed_trajectory_shard(arrays)["passed"] is True

    second_run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second_run.returncode != 0
    assert "refusing to overwrite" in second_run.stderr
