"""Build and verify the Day 24 final offline dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.offline_dataset import (
    initial_condition_id,
    sample_split_initial_states,
)
from powered_landing_guidance.offline_dataset_finalization import (
    FinalDatasetSettings,
    analyze_coverage,
    build_final_dataset_manifest,
    compute_shard_normalization,
    initial_states_from_shard,
    load_verified_source_train,
    merge_packed_shards,
    sample_coverage_states,
    sample_hard_training_states,
    save_distribution_plot,
)
from powered_landing_guidance.offline_dataset_generation import (
    OfflineGenerationResult,
    generate_teacher_trajectories,
)

_OUTPUT_NAMES = {
    "manifest": "dataset-manifest.json",
    "normalization": "normalization.json",
    "distribution": "dataset-distribution.png",
    "train": "train-trajectories.npz",
    "validation": "validation-trajectories.npz",
    "test": "test-trajectories.npz",
}


class ProgressMeter:
    """Record elapsed time and ETA for one named solver batch."""

    def __init__(self, label: str, interval: int) -> None:
        self.label = label
        self.interval = interval
        self.started = perf_counter()
        self.successes = 0
        self.failures = 0
        self.samples: list[dict[str, Any]] = []

    def __call__(self, completed: int, total: int, case: dict[str, Any]) -> None:
        if case["status"] == "success":
            self.successes += 1
        else:
            self.failures += 1
        if completed != 1 and completed % self.interval != 0 and completed != total:
            return
        elapsed = perf_counter() - self.started
        throughput = completed / elapsed if elapsed > 0.0 else 0.0
        eta = (total - completed) / throughput if throughput > 0.0 else None
        sample = {
            "completed": completed,
            "total": total,
            "solver_successes": self.successes,
            "solver_failures": self.failures,
            "elapsed_s": elapsed,
            "throughput_cases_per_s": throughput,
            "estimated_remaining_s": eta,
        }
        self.samples.append(sample)
        eta_text = "unknown" if eta is None else f"{eta:.1f}s"
        print(
            f"[{self.label} {completed:04d}/{total:04d}] success={self.successes} "
            f"failed={self.failures} elapsed={elapsed:.1f}s eta={eta_text}",
            flush=True,
        )


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in _OUTPUT_NAMES.items()}


def ensure_outputs_available(destinations: dict[str, Path]) -> None:
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")


def _run_generation(
    config: dict[str, Any],
    final: FinalDatasetSettings,
    split: str,
    states: np.ndarray,
    minimum: int,
    label: str,
) -> tuple[OfflineGenerationResult, list[dict[str, Any]]]:
    progress = ProgressMeter(label, int(final.generation["progress_interval"]))
    result = generate_teacher_trajectories(
        config,
        states,
        final.solver_settings(split, minimum),
        progress=progress,
    )
    return result, progress.samples


def _write_outputs(
    destinations: dict[str, Path],
    config: dict[str, Any],
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    shards: dict[str, dict[str, np.ndarray]],
) -> None:
    output_dir = destinations["manifest"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary: dict[str, Path] = {}
    try:
        for split in ("train", "validation", "test"):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{split}-trajectories-",
                suffix=".npz.tmp",
                delete=False,
            ) as stream:
                temporary[split] = Path(stream.name)
                np.savez_compressed(stream, **shards[split])
        for name, payload in (("manifest", manifest), ("normalization", normalization)):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=f".{name}-",
                suffix=".json.tmp",
                delete=False,
            ) as stream:
                temporary[name] = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=".dataset-distribution-",
            suffix=".png",
            delete=False,
        ) as stream:
            temporary["distribution"] = Path(stream.name)
        save_distribution_plot(config, shards, temporary["distribution"])
        for name in _OUTPUT_NAMES:
            os.replace(temporary.pop(name), destinations[name])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/pid-baseline-v1.yaml",
    )
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=root / "artifacts/day23-easy-dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/day24-final-dataset",
    )
    args = parser.parse_args()
    destinations = artifact_paths(args.output_dir)
    try:
        ensure_outputs_available(destinations)
    except FileExistsError as error:
        parser.error(str(error))

    config = load_config(args.config)
    final = FinalDatasetSettings.from_config(config)
    source_train_arrays, source_train_report = load_verified_source_train(
        config, args.source_train_dir
    )

    hard_states, hard_selection = sample_hard_training_states(config)
    challenge_result, challenge_progress = _run_generation(
        config,
        final,
        "train",
        hard_states,
        1,
        "challenge",
    )
    train_sources = [source_train_arrays, challenge_result.arrays]
    current_train = merge_packed_shards(train_sources)
    coverage_before = analyze_coverage(config, initial_states_from_shard(current_train))

    coverage_results: list[OfflineGenerationResult] = []
    coverage_rounds: list[dict[str, Any]] = []
    progress: dict[str, list[dict[str, Any]]] = {"challenge_train": challenge_progress}
    maximum_cases = int(final.coverage["maximum_additional_cases"])
    maximum_rounds = int(final.coverage["maximum_rounds"])
    generated_cases = 0
    for round_index in range(maximum_rounds):
        current_states = initial_states_from_shard(current_train)
        current_coverage = analyze_coverage(config, current_states)
        if current_coverage["passed"]:
            break
        remaining = maximum_cases - generated_cases
        excluded = {initial_condition_id(state) for state in current_states}
        candidates, plan = sample_coverage_states(
            config,
            current_states,
            round_index=round_index,
            maximum_count=remaining,
            excluded_initial_ids=excluded,
        )
        if not len(candidates):
            break
        result, samples = _run_generation(
            config,
            final,
            "train",
            candidates,
            1,
            f"coverage-{round_index + 1}",
        )
        generated_cases += len(candidates)
        coverage_results.append(result)
        coverage_rounds.append(
            {
                "round": round_index + 1,
                "coverage_before": current_coverage,
                "planned_cases": len(candidates),
                "target_plan": plan,
                "accepted_trajectories": result.accepted_trajectories,
            }
        )
        progress[f"coverage_round_{round_index + 1}"] = samples
        train_sources.append(result.arrays)
        current_train = merge_packed_shards(train_sources)

    validation_states = sample_split_initial_states(config, "validation")
    validation_result, validation_progress = _run_generation(
        config,
        final,
        "validation",
        validation_states,
        int(final.completion["minimum_validation_trajectories"]),
        "validation",
    )
    progress["validation"] = validation_progress

    test_states = sample_split_initial_states(config, "test")
    test_result, test_progress = _run_generation(
        config,
        final,
        "test",
        test_states,
        int(final.completion["minimum_test_trajectories"]),
        "hard-test",
    )
    progress["test"] = test_progress

    shards = {
        "train": current_train,
        "validation": validation_result.arrays,
        "test": test_result.arrays,
    }
    coverage_after = analyze_coverage(config, initial_states_from_shard(current_train))
    normalization = compute_shard_normalization(config, current_train)
    manifest = build_final_dataset_manifest(
        config,
        source_train=source_train_report,
        hard_selection=hard_selection,
        challenge_result=challenge_result,
        coverage_results=coverage_results,
        coverage_rounds=coverage_rounds,
        validation_result=validation_result,
        test_result=test_result,
        shards=shards,
        normalization=normalization,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        progress=progress,
    )
    _write_outputs(destinations, config, manifest, normalization, shards)

    counts = {split: len(shard["trajectory_ids"]) for split, shard in shards.items()}
    print(
        f"final trajectories train={counts['train']} validation={counts['validation']} "
        f"test={counts['test']} coverage_min="
        f"{min(item['minimum_count'] for item in coverage_after['features'].values())}",
        flush=True,
    )
    print(f"completion gate={manifest['completion_gate']['passed']}", flush=True)
    for path in destinations.values():
        print(f"saved: {path}", flush=True)
    if not manifest["completion_gate"]["passed"]:
        raise SystemExit("Day 24 final offline-dataset completion gate did not pass")


if __name__ == "__main__":
    main()
