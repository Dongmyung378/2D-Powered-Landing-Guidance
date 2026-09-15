"""Evaluate the frozen Week 2 PID baseline and render representative GIFs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance import (
    IntegratedLandingController,
    frozen_baseline_initial_states,
    initial_condition_sha256,
    integrated_controller_sha256,
    load_config,
)
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.visualization import save_animation, save_episode_data

if __package__:
    from scripts.evaluate_integrated_control import (
        IntegratedControlResult,
        evaluate_integrated_control,
    )
else:
    from evaluate_integrated_control import IntegratedControlResult, evaluate_integrated_control


_ARTIFACT_NAMES = {
    "report": "week2-baseline.json",
    "success_log": "week2-success.json",
    "success_gif": "week2-success.gif",
    "failure_log": "week2-failure.json",
    "failure_gif": "week2-failure.gif",
}


def baseline_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return every output path produced by the frozen baseline command."""
    return {label: output_dir / name for label, name in _ARTIFACT_NAMES.items()}


def ensure_artifact_paths_available(destinations: dict[str, Path]) -> None:
    """Reject a run before simulation if any destination already exists."""
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")


def run_pid_episode(
    config: dict[str, Any],
    initial_state: NDArray[np.float64],
    *,
    horizontal_wind_m_s: float = 0.0,
) -> dict[str, Any]:
    """Run one integrated-PID episode and return its complete replay log."""
    episode_config = deepcopy(config)
    episode_config["environment"]["horizontal_wind_m_s"] = float(horizontal_wind_m_s)
    env = RocketLandingEnv(episode_config)
    controller = IntegratedLandingController.from_config(episode_config)
    try:
        state, info = env.reset(options={"initial_state": initial_state})
        while True:
            action = controller.command(state, time_s=float(info["time_s"]))
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return env.episode_log
    finally:
        env.close()


def evaluate_frozen_baseline(
    config: dict[str, Any],
) -> tuple[IntegratedControlResult, NDArray[np.float64]]:
    """Verify both hashes and evaluate the immutable nominal benchmark set."""
    initial_states = frozen_baseline_initial_states(config)
    result = evaluate_integrated_control(config, initial_states)
    minimum = float(config["baseline_protocol"]["acceptance"]["minimum_success_rate"])
    if result.success_rate < minimum:
        raise RuntimeError(
            f"frozen baseline failed its acceptance gate: {result.success_rate:.1%} < {minimum:.1%}"
        )
    return result, initial_states


def representative_episodes(
    config: dict[str, Any],
    initial_states: NDArray[np.float64],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce the configured nominal success and stressed failure cases."""
    cases = config["baseline_protocol"]["representative_cases"]
    success_index = int(cases["success_initial_condition_index"])
    failure = cases["failure"]
    failure_index = int(failure["initial_condition_index"])
    wind = float(failure["horizontal_wind_m_s"])

    success_episode = run_pid_episode(config, initial_states[success_index])
    if success_episode["outcome"] != "success":
        raise RuntimeError("configured representative success case did not land safely")

    failure_episode = run_pid_episode(
        config,
        initial_states[failure_index],
        horizontal_wind_m_s=wind,
    )
    if failure_episode["outcome"] == "success":
        raise RuntimeError("configured representative stress case did not fail")
    return success_episode, failure_episode


def _case_summary(
    episode: dict[str, Any],
    *,
    initial_condition_index: int,
    horizontal_wind_m_s: float,
) -> dict[str, Any]:
    final_step = episode["steps"][-1]
    return {
        "initial_condition_index": initial_condition_index,
        "horizontal_wind_m_s": horizontal_wind_m_s,
        "outcome": episode["outcome"],
        "initial_state": episode["initial_state"],
        "final_state": final_step["state"],
        "flight_time_s": final_step["time_s"],
        "fuel_used_kg": final_step["fuel_used_kg"],
    }


def save_baseline_artifacts(
    config: dict[str, Any],
    result: IntegratedControlResult,
    initial_states: NDArray[np.float64],
    output_dir: Path,
    *,
    fps: int,
    max_frames: int,
) -> tuple[Path, ...]:
    """Write the benchmark report, replay logs, and representative GIFs."""
    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")
    destinations = baseline_artifact_paths(output_dir)
    ensure_artifact_paths_available(destinations)

    success_episode, failure_episode = representative_episodes(config, initial_states)
    cases = config["baseline_protocol"]["representative_cases"]
    failure_settings = cases["failure"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_episode_data(success_episode, destinations["success_log"])
    save_animation(
        success_episode,
        destinations["success_gif"],
        fps=fps,
        max_frames=max_frames,
    )
    save_episode_data(failure_episode, destinations["failure_log"])
    save_animation(
        failure_episode,
        destinations["failure_gif"],
        fps=fps,
        max_frames=max_frames,
    )

    protocol = config["baseline_protocol"]
    minimum = float(protocol["acceptance"]["minimum_success_rate"])
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "controller": protocol["controller"],
        "frozen": protocol["frozen"],
        "controller_sha256": integrated_controller_sha256(config),
        "initial_conditions": {
            **protocol["initial_conditions"],
            "verified_sha256": initial_condition_sha256(initial_states),
        },
        "acceptance": {
            "minimum_success_rate": minimum,
            "passed": result.success_rate >= minimum,
        },
        "nominal_result": asdict(result),
        "representative_cases": {
            "success": _case_summary(
                success_episode,
                initial_condition_index=int(cases["success_initial_condition_index"]),
                horizontal_wind_m_s=0.0,
            ),
            "failure": _case_summary(
                failure_episode,
                initial_condition_index=int(failure_settings["initial_condition_index"]),
                horizontal_wind_m_s=float(failure_settings["horizontal_wind_m_s"]),
            ),
        },
    }
    with destinations["report"].open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return tuple(destinations.values())


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
        default=root / "artifacts/week2-baseline",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()

    try:
        ensure_artifact_paths_available(baseline_artifact_paths(args.output_dir))
    except FileExistsError as error:
        parser.error(str(error))
    config = load_config(args.config)
    result, initial_states = evaluate_frozen_baseline(config)
    paths = save_baseline_artifacts(
        config,
        result,
        initial_states,
        args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"protocol: {config['baseline_protocol']['id']}")
    print(f"initial-condition SHA-256: {initial_condition_sha256(initial_states)}")
    print(f"success: {result.successes}/{result.episodes} ({result.success_rate:.1%})")
    print(f"outcomes: {result.outcome_counts}")
    print(f"mean fuel used: {result.mean_fuel_used_kg:.4f} kg")
    for path in paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
