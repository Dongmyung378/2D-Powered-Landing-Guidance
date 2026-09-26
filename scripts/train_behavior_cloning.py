"""Train the first Day 25 Behavior Cloning policy on the frozen offline dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.behavior_cloning import (  # noqa: E402
    TrainingSettings,
    load_bc_dataset,
    load_checkpoint,
    overfit_check,
    save_checkpoint,
    train_bc,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/pid-baseline-v1.yaml")
    parser.add_argument("--training-config", type=Path, default=root / "configs/bc-day25.yaml")
    parser.add_argument("--dataset-dir", type=Path, default=root / "artifacts/day24-final-dataset")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/day25-bc")
    args = parser.parse_args()
    checkpoint_path = args.output_dir / "bc-checkpoint.npz"
    report_path = args.output_dir / "training-report.json"
    if checkpoint_path.exists() or report_path.exists():
        parser.error("output already exists; choose another --output-dir")

    config = load_config(args.config)
    settings = TrainingSettings.from_dict(
        yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    )
    started = perf_counter()
    dataset = load_bc_dataset(args.dataset_dir, config)
    print(
        f"Loaded {len(dataset.train_states)} train and "
        f"{len(dataset.validation_states)} validation state-action pairs",
        flush=True,
    )
    overfit = overfit_check(dataset, settings)
    print(
        f"Small-subset overfit: {overfit['passed']} at step {overfit['steps']}, "
        f"MSE={overfit['metrics']['normalized_mse']:.6f}",
        flush=True,
    )
    if not overfit["passed"]:
        raise SystemExit("small-subset overfit gate failed")

    def progress(epoch: int, train_mse: float, validation_mse: float, best_epoch: int) -> None:
        print(
            f"epoch {epoch:03d}: train MSE={train_mse:.6f} "
            f"validation MSE={validation_mse:.6f} best={best_epoch}",
            flush=True,
        )

    policy, training = train_bc(dataset, settings, progress=progress)
    metadata = {
        "schema_version": 1,
        "policy": "planar-bc-mlp-v1",
        "dataset_id": dataset.dataset_id,
        "dataset_configuration_sha256": dataset.dataset_configuration_sha256,
        "shard_sha256": dataset.shard_sha256,
        "normalization": dataset.normalization,
        "training_settings": settings.as_dict(),
        "best_epoch": training["best_epoch"],
        "throttle_bounds": [config["vehicle"]["throttle_min"], config["vehicle"]["throttle_max"]],
        "gimbal_limit_rad": float(np.deg2rad(config["vehicle"]["gimbal_limit_deg"])),
    }
    report = {
        "schema_version": 1,
        "problem": "day25-behavior-cloning",
        "status": "complete",
        "dataset_id": dataset.dataset_id,
        "dataset_configuration_sha256": dataset.dataset_configuration_sha256,
        "shard_sha256": dataset.shard_sha256,
        "trajectory_counts": dataset.trajectory_counts,
        "sample_counts": {
            "train": len(dataset.train_states),
            "validation": len(dataset.validation_states),
        },
        "settings": settings.as_dict(),
        "overfit_check": overfit,
        "training": training,
        "elapsed_s": perf_counter() - started,
        "completion_gate": {
            "small_subset_overfit": overfit["passed"],
            "full_dataset_first_training": bool(training["best_epoch"]),
        },
    }
    report["completion_gate"]["passed"] = all(report["completion_gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, policy, metadata)
    loaded, saved_metadata = load_checkpoint(checkpoint_path)
    if saved_metadata != metadata or any(
        not np.array_equal(first, second)
        for first, second in zip(
            policy.weights + policy.biases, loaded.weights + loaded.biases, strict=True
        )
    ):
        raise RuntimeError("saved BC checkpoint failed round-trip validation")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        f"best epoch={training['best_epoch']}; train MSE="
        f"{training['best_train']['normalized_mse']:.6f}; validation MSE="
        f"{training['best_validation']['normalized_mse']:.6f}",
        flush=True,
    )
    print(f"saved: {checkpoint_path}\nsaved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
