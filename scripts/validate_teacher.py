"""Validate and freeze the Day 21 teacher against the same nominal PID test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from powered_landing_guidance import frozen_teacher_initial_states, load_config
from powered_landing_guidance.teacher_validation import (
    build_teacher_episode,
    build_teacher_validation_report,
    evaluate_pid_on_states,
    load_teacher_bundle,
    revalidate_teacher_bundle,
    select_representative_case,
)
from powered_landing_guidance.visualization import save_animation, save_episode_data

_ARTIFACT_NAMES = {
    "report": "teacher-validation.json",
    "representative_log": "representative-teacher.json",
    "representative_gif": "representative-teacher.gif",
}


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the complete Day 21 output set."""
    return {name: output_dir / filename for name, filename in _ARTIFACT_NAMES.items()}


def ensure_outputs_available(destinations: dict[str, Path]) -> None:
    """Refuse the run before evaluation when any final output already exists."""
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/pid-baseline-v1.yaml",
    )
    parser.add_argument(
        "--teacher-report",
        type=Path,
        default=root / "artifacts/day20-teacher-pipeline/teacher-pipeline.json",
    )
    parser.add_argument(
        "--teacher-dataset",
        type=Path,
        default=root / "artifacts/day20-teacher-pipeline/teacher-trajectories.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/day21-teacher-validation",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()

    destinations = artifact_paths(args.output_dir)
    try:
        ensure_outputs_available(destinations)
    except FileExistsError as error:
        parser.error(str(error))
    if args.fps <= 0 or args.max_frames <= 0:
        parser.error("--fps and --max-frames must be positive")

    config = load_config(args.config)
    print("loading and checking the frozen Day 20 bundle", flush=True)
    bundle = load_teacher_bundle(config, args.teacher_report, args.teacher_dataset)
    print(f"replaying {len(bundle.case_indices)} accepted teacher trajectories", flush=True)
    replay = revalidate_teacher_bundle(config, bundle)
    print(f"evaluating PID on {len(bundle.report['cases'])} identical initial states", flush=True)
    pid = evaluate_pid_on_states(config, frozen_teacher_initial_states(config))

    selection = select_representative_case(bundle)
    episode = build_teacher_episode(config, bundle, selection)
    report = build_teacher_validation_report(config, bundle, replay, pid, selection)
    report["artifacts"] = {
        "representative_log": destinations["representative_log"].name,
        "representative_animation": destinations["representative_gif"].name,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_episode_data(episode, destinations["representative_log"])
    save_animation(
        episode,
        destinations["representative_gif"],
        fps=args.fps,
        max_frames=args.max_frames,
    )
    with destinations["report"].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)

    teacher = report["teacher"]
    print(
        f"teacher success={teacher['successful_cases']}/{teacher['executed_cases']} "
        f"({teacher['success_rate']:.1%})",
        flush=True,
    )
    print(
        f"independent replay={replay['passed_cases']}/{replay['accepted_cases']} "
        f"passed={replay['all_passed']}",
        flush=True,
    )
    print(
        f"PID success={pid['successes']}/{pid['episodes']} ({pid['success_rate']:.1%})",
        flush=True,
    )
    print(f"Week 3 gate={report['week3_acceptance']['passed']}", flush=True)
    for path in destinations.values():
        print(f"saved: {path}", flush=True)
    if not report["week3_acceptance"]["passed"]:
        raise SystemExit("Day 21 Week 3 acceptance gate did not pass")


if __name__ == "__main__":
    main()
