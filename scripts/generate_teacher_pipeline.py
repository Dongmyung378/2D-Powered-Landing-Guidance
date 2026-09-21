"""Run the Day 20 multi-initial-condition teacher solver pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.teacher_pipeline import (
    build_teacher_pipeline_report,
    run_teacher_pipeline,
    teacher_trajectory_arrays,
)


def _progress(completed: int, total: int, case: dict[str, object]) -> None:
    attempts = int(case["attempt_count"])
    print(
        f"[{completed:03d}/{total:03d}] case={case['case_index']} "
        f"status={case['status']} attempts={attempts}",
        flush=True,
    )


def _write_outputs(
    output_dir: Path,
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    report_path = output_dir / "teacher-pipeline.json"
    dataset_path = output_dir / "teacher-trajectories.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_temp: Path | None = None
    dataset_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".teacher-pipeline-",
            suffix=".json.tmp",
            delete=False,
        ) as stream:
            report_temp = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=".teacher-trajectories-",
            suffix=".npz.tmp",
            delete=False,
        ) as stream:
            dataset_temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(dataset_temp, dataset_path)
        dataset_temp = None
        os.replace(report_temp, report_path)
        report_temp = None
    finally:
        for temporary in (report_temp, dataset_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return report_path, dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pid-baseline-v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/day20-teacher-pipeline",
    )
    args = parser.parse_args()
    destinations = (
        args.output_dir / "teacher-pipeline.json",
        args.output_dir / "teacher-trajectories.npz",
    )
    for destination in destinations:
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    result = run_teacher_pipeline(load_config(args.config), progress=_progress)
    report = build_teacher_pipeline_report(result)
    arrays = teacher_trajectory_arrays(result)
    report["dataset"] = {
        "file": "teacher-trajectories.npz",
        "successful_case_indices": list(result.successful_case_indices),
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in arrays.items()
        },
    }
    report_path, dataset_path = _write_outputs(args.output_dir, report, arrays)
    summary = report["summary"]
    print(
        f"completed={summary['executed_cases']} success={summary['successful_cases']} "
        f"failed={summary['failed_cases']} retries={summary['retried_cases']} "
        f"timeouts={summary['timeout_attempts']}",
        flush=True,
    )
    print(f"completion gate={report['completion_gate']['passed']}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"dataset={dataset_path}", flush=True)
    if not result.completion_passed:
        raise SystemExit("teacher pipeline did not execute the configured minimum case count")


if __name__ == "__main__":
    main()
