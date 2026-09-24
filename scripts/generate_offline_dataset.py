"""Generate and verify the Day 23 easy-split Teacher trajectories."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.offline_dataset_generation import (
    OfflineGenerationSettings,
    build_offline_generation_report,
    generate_easy_teacher_trajectories,
)

_OUTPUT_NAMES = {
    "manifest": "train-manifest.json",
    "shard": "train-trajectories.npz",
}


class ProgressMeter:
    """Measure and print elapsed time, throughput, and whole-run ETA."""

    def __init__(self, interval: int) -> None:
        self.interval = interval
        self.started = perf_counter()
        self.successes = 0
        self.failures = 0
        self.samples: list[dict[str, object]] = []

    def __call__(self, completed: int, total: int, case: dict[str, object]) -> None:
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
            f"[{completed:04d}/{total:04d}] success={self.successes} "
            f"failed={self.failures} elapsed={elapsed:.1f}s eta={eta_text}",
            flush=True,
        )


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the complete Day 23 output set."""
    return {name: output_dir / filename for name, filename in _OUTPUT_NAMES.items()}


def ensure_outputs_available(destinations: dict[str, Path]) -> None:
    """Refuse to replace either a manifest or trajectory shard."""
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")


def _write_outputs(
    destinations: dict[str, Path],
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir = destinations["manifest"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_temp: Path | None = None
    shard_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=".train-trajectories-",
            suffix=".npz.tmp",
            delete=False,
        ) as stream:
            shard_temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".train-manifest-",
            suffix=".json.tmp",
            delete=False,
        ) as stream:
            manifest_temp = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(shard_temp, destinations["shard"])
        shard_temp = None
        os.replace(manifest_temp, destinations["manifest"])
        manifest_temp = None
    finally:
        for temporary in (manifest_temp, shard_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/pid-baseline-v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/day23-easy-dataset",
    )
    args = parser.parse_args()
    destinations = artifact_paths(args.output_dir)
    try:
        ensure_outputs_available(destinations)
    except FileExistsError as error:
        parser.error(str(error))

    config = load_config(args.config)
    settings = OfflineGenerationSettings.from_config(config)
    progress = ProgressMeter(settings.progress_interval)
    result = generate_easy_teacher_trajectories(config, progress=progress)
    report = build_offline_generation_report(config, result, progress.samples)
    report["artifacts"] = {
        "manifest": destinations["manifest"].name,
        "trajectory_shard": destinations["shard"].name,
    }
    _write_outputs(destinations, report, result.arrays)

    summary = report["summary"]
    print(
        f"executed={summary['executed_cases']} solver_success={summary['solver_successes']} "
        f"accepted={summary['accepted_trajectories']} "
        f"rejected={summary['filter_rejections']} retries={summary['retried_cases']} "
        f"timeouts={summary['timeout_attempts']}",
        flush=True,
    )
    print(f"shard SHA-256={report['shard']['sha256']}", flush=True)
    print(f"completion gate={report['completion_gate']['passed']}", flush=True)
    for path in destinations.values():
        print(f"saved: {path}", flush=True)
    if not report["completion_gate"]["passed"]:
        raise SystemExit("Day 23 offline-dataset generation completion gate did not pass")


if __name__ == "__main__":
    main()
