"""Day 25 supervised learning and frozen-dataset checks."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import load_config
from powered_landing_guidance.behavior_cloning import (
    BCDataset,
    BCPolicy,
    TrainingSettings,
    action_error,
    load_bc_dataset,
    load_checkpoint,
    overfit_check,
    save_checkpoint,
    train_bc,
)
from powered_landing_guidance.offline_dataset import (
    compute_training_normalization_statistics,
    dataset_configuration_sha256,
    initial_condition_id,
)
from powered_landing_guidance.offline_dataset_generation import (
    _empty_packed_arrays,
    packed_shard_sha256,
    trajectory_id,
    validate_packed_trajectory_shard,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")


def _shard(split: str, x: float) -> dict[str, np.ndarray]:
    arrays = _empty_packed_arrays()
    initial = np.asarray((x, 100.0, 0.0, -20.0, 0.0, 0.0, 980.0))
    initial_id = initial_condition_id(initial)
    arrays.update(
        {
            "trajectory_ids": np.asarray([trajectory_id(CONFIG, split, initial_id)], dtype="<U64"),
            "initial_condition_ids": np.asarray([initial_id], dtype="<U64"),
            "source_case_indices": np.asarray([0], dtype=np.int64),
            "state_offsets": np.asarray([0, 3], dtype=np.int64),
            "action_offsets": np.asarray([0, 2], dtype=np.int64),
            "times_s": np.asarray([0.0, 1.0, 2.0]),
            "states": np.vstack(
                (initial, initial + [0, -10, 0, 1, 0, 0, -5], initial + [0, -20, 0, 2, 0, 0, -10])
            ),
            "actions": np.asarray(((0.3, -0.02), (0.6, 0.01))),
            "durations_s": np.asarray([2.0]),
            "solver_success": np.asarray([True]),
            "stage_a_iterations": np.asarray([1], dtype=np.int64),
            "stage_b_iterations": np.asarray([1], dtype=np.int64),
            "max_hard_violation": np.asarray([0.0]),
            "max_terminal_violation": np.asarray([0.0]),
            "replay_passed": np.asarray([True]),
            "max_final_error_ratio": np.asarray([0.0]),
            "max_node_error_ratio": np.asarray([0.0]),
            "fuel_used_kg": np.asarray([10.0]),
            "attempt_count": np.asarray([1], dtype=np.int64),
        }
    )
    assert validate_packed_trajectory_shard(arrays)["passed"]
    return arrays


def _dataset() -> BCDataset:
    rng = np.random.default_rng(3)
    train_x = rng.normal(size=(128, 7)).astype(np.float32)
    val_x = rng.normal(size=(32, 7)).astype(np.float32)
    matrix = rng.normal(size=(7, 2)).astype(np.float32) * 0.2
    train_y = train_x @ matrix
    val_y = val_x @ matrix
    normalization = {"action": {"scale": [0.25, 0.05]}}
    return BCDataset(train_x, train_y, val_x, val_y, normalization, "fixture", "0" * 64, {}, {})


def test_loader_verifies_provenance_and_excludes_terminal_state(tmp_path: Path) -> None:
    train = _shard("train", 3.0)
    validation = _shard("validation", 4.0)
    statistics = compute_training_normalization_statistics(
        CONFIG, train["states"][:2], train["actions"]
    )
    manifest = {
        "problem": "day24-final-offline-dataset",
        "dataset_id": CONFIG["offline_dataset_finalization"]["id"],
        "dataset_configuration_sha256": dataset_configuration_sha256(CONFIG),
        "teacher_provenance": {
            "configuration_sha256": CONFIG["teacher_protocol"]["configuration_sha256"]
        },
        "completion_gate": {"passed": True},
        "split_integrity": {"passed": True},
        "normalization": statistics,
        "shards": {
            split: {
                "file": f"{split}-trajectories.npz",
                "trajectory_count": 1,
                "sha256": packed_shard_sha256(shard),
            }
            for split, shard in (("train", train), ("validation", validation))
        },
    }
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "normalization.json").write_text(json.dumps(statistics), encoding="utf-8")
    np.savez_compressed(tmp_path / "train-trajectories.npz", **train)
    np.savez_compressed(tmp_path / "validation-trajectories.npz", **validation)

    loaded = load_bc_dataset(tmp_path, CONFIG)
    assert loaded.train_states.shape == (2, 7)
    assert loaded.validation_actions.shape == (2, 2)
    np.testing.assert_allclose(loaded.train_states.mean(axis=0), 0.0, atol=1e-6)
    assert loaded.normalization["state"]["count"] == 2

    bad_manifest = deepcopy(manifest)
    bad_manifest["shards"]["validation"]["sha256"] = "0" * 64
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bc_dataset(tmp_path, CONFIG)

    bad_normalization = deepcopy(statistics)
    bad_normalization["state"]["mean"][0] += 1.0
    manifest["normalization"] = bad_normalization
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "normalization.json").write_text(json.dumps(bad_normalization), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization values"):
        load_bc_dataset(tmp_path, CONFIG)


def test_policy_gradient_matches_finite_difference() -> None:
    policy = BCPolicy((5,), 7)
    policy.weights[0] = np.abs(policy.weights[0])
    policy.biases[0].fill(0.3)
    states = np.full((3, 7), 0.5, dtype=np.float32)
    targets = np.full((3, 2), -0.2, dtype=np.float32)
    _, gradients, _ = policy.gradients(states, targets)
    epsilon = 1e-3
    original = float(policy.weights[0][0, 0])
    policy.weights[0][0, 0] = original + epsilon
    plus = policy.gradients(states, targets)[0]
    policy.weights[0][0, 0] = original - epsilon
    minus = policy.gradients(states, targets)[0]
    policy.weights[0][0, 0] = original
    assert gradients[0][0, 0] == pytest.approx((plus - minus) / (2 * epsilon), rel=2e-3)


def test_subset_overfit_full_training_checkpoint_and_clipping(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = TrainingSettings(
        hidden_sizes=(16,),
        batch_size=32,
        max_epochs=70,
        patience=8,
        learning_rate=0.01,
        overfit_examples=8,
        overfit_steps=500,
        overfit_max_mse=0.0025,
    )
    assert overfit_check(dataset, settings)["passed"]
    policy, result = train_bc(dataset, settings)
    assert result["best_validation"]["normalized_mse"] < 0.01
    assert result["best_epoch"] <= result["epochs_run"]
    assert result["best_validation"]["normalized_mse"] == pytest.approx(
        min(row["validation"]["normalized_mse"] for row in result["history"])
    )
    assert (
        action_error(
            policy, dataset.validation_states, dataset.validation_actions, dataset.normalization
        )
        == result["best_validation"]
    )

    metadata = {"training_settings": settings.as_dict(), "normalization": dataset.normalization}
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, policy, metadata)
    restored, saved = load_checkpoint(path)
    assert saved == metadata
    np.testing.assert_array_equal(
        restored.predict_normalized(dataset.validation_states),
        policy.predict_normalized(dataset.validation_states),
    )
    normalization = {
        "state": {"mean": [0] * 7, "scale": [1] * 7},
        "action": {"mean": [0, 0], "scale": [1, 1]},
    }
    action = restored.predict_action(np.zeros(7), normalization, (0.0, 1.0), 0.1)
    assert 0.0 <= action[0] <= 1.0
    assert abs(action[1]) <= 0.1


def test_early_stopping_uses_validation_only() -> None:
    dataset = _dataset()
    contradictory = BCDataset(
        dataset.train_states,
        np.ones_like(dataset.train_actions),
        dataset.validation_states,
        -np.ones_like(dataset.validation_actions),
        dataset.normalization,
        dataset.dataset_id,
        dataset.dataset_configuration_sha256,
        {},
        {},
    )
    settings = TrainingSettings(
        hidden_sizes=(8,), batch_size=32, max_epochs=30, patience=3, learning_rate=0.01
    )
    _, result = train_bc(contradictory, settings)
    assert result["stopped_early"] is True
    assert result["epochs_run"] == result["best_epoch"] + settings.patience
