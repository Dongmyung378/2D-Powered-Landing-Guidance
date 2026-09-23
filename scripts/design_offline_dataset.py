"""Create the Day 22 dataset schema and deterministic split plan."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.offline_dataset import build_dataset_design

_ARTIFACT_NAMES = {
    "report": "dataset-design.json",
    "initial_conditions": "initial-condition-plan.npz",
}


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the complete Day 22 generated output set."""
    return {name: output_dir / filename for name, filename in _ARTIFACT_NAMES.items()}


def ensure_outputs_available(destinations: dict[str, Path]) -> None:
    """Reject the run before planning when either final output already exists."""
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")


def _write_outputs(
    destinations: dict[str, Path],
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir = destinations["report"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_temp: Path | None = None
    initial_conditions_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".dataset-design-",
            suffix=".json.tmp",
            delete=False,
        ) as stream:
            report_temp = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=".initial-condition-plan-",
            suffix=".npz.tmp",
            delete=False,
        ) as stream:
            initial_conditions_temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(initial_conditions_temp, destinations["initial_conditions"])
        initial_conditions_temp = None
        os.replace(report_temp, destinations["report"])
        report_temp = None
    finally:
        for temporary in (report_temp, initial_conditions_temp):
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
        default=root / "artifacts/day22-dataset-design",
    )
    args = parser.parse_args()
    destinations = artifact_paths(args.output_dir)
    try:
        ensure_outputs_available(destinations)
    except FileExistsError as error:
        parser.error(str(error))

    report, arrays = build_dataset_design(load_config(args.config))
    report["artifacts"] = {
        "initial_condition_plan": destinations["initial_conditions"].name,
        "dataset_card": "docs/dataset-card.md",
        "dataset_card_korean": "docs/dataset-card.ko.md",
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in arrays.items()
        },
    }
    _write_outputs(destinations, report, arrays)

    print(f"dataset={report['dataset_id']}", flush=True)
    print(f"configuration SHA-256={report['dataset_configuration_sha256']}", flush=True)
    for split, details in report["split_plan"].items():
        print(
            f"{split}: episodes={details['episodes']} seed={details['seed']} "
            f"SHA-256={details['initial_state_sha256']}",
            flush=True,
        )
    print(f"split integrity={report['split_integrity']['passed']}", flush=True)
    print(f"harder test={report['test_difficulty']['passed']}", flush=True)
    print(f"completion gate={report['completion_gate']['passed']}", flush=True)
    for path in destinations.values():
        print(f"saved: {path}", flush=True)
    if not report["completion_gate"]["passed"]:
        raise SystemExit("Day 22 dataset-design completion gate did not pass")


if __name__ == "__main__":
    main()
