"""Evaluate the integrated landing controller on medium-difficulty initial states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance import IntegratedLandingController, load_config
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.evaluation import sample_integrated_initial_states


@dataclass(frozen=True, slots=True)
class IntegratedControlResult:
    episodes: int
    successes: int
    success_rate: float
    outcome_counts: dict[str, int]
    mean_abs_touchdown_x_m: float
    max_abs_touchdown_x_m: float
    mean_abs_touchdown_vx_m_s: float
    max_abs_touchdown_vx_m_s: float
    mean_abs_touchdown_vz_m_s: float
    p95_abs_touchdown_vz_m_s: float
    max_abs_touchdown_vz_m_s: float
    mean_abs_touchdown_theta_deg: float
    max_abs_touchdown_theta_deg: float
    mean_abs_touchdown_omega_deg_s: float
    max_abs_touchdown_omega_deg_s: float
    mean_fuel_used_kg: float
    mean_flight_time_s: float
    mean_control_updates: float
    approach_updates: int
    terminal_updates: int
    slew_limited_updates: int
    max_throttle_slew_per_s: float
    max_gimbal_slew_deg_s: float


def evaluate_integrated_control(
    config: dict,
    initial_states: NDArray[np.float64],
) -> IntegratedControlResult:
    """Run each state to termination and summarize touchdown and rate-limit behavior."""
    if initial_states.ndim != 2 or initial_states.shape[1] != 7 or len(initial_states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    if not np.isclose(
        float(config["environment"]["horizontal_wind_m_s"]),
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("integrated-controller evaluation requires zero horizontal wind")

    outcomes: list[str] = []
    touchdown_states: list[NDArray[np.float64]] = []
    fuel_used: list[float] = []
    flight_times: list[float] = []
    control_updates: list[int] = []
    phase_updates: Counter[str] = Counter()
    slew_limited_updates = 0
    max_throttle_slew = 0.0
    max_gimbal_slew = 0.0
    env = RocketLandingEnv(config)
    try:
        for initial_state in initial_states:
            controller = IntegratedLandingController.from_config(config)
            state, info = env.reset(options={"initial_state": initial_state})
            previous_update_action: NDArray[np.float64] | None = None
            previous_update_time: float | None = None
            while True:
                command_time = float(info["time_s"])
                action = controller.command(state, time_s=command_time)
                if controller.updated_this_step:
                    phase_updates[str(controller.phase)] += 1
                    slew_limited_updates += int(
                        controller.throttle_slew_limited or controller.gimbal_slew_limited
                    )
                    if previous_update_action is not None and previous_update_time is not None:
                        elapsed = command_time - previous_update_time
                        if elapsed > 0.0:
                            max_throttle_slew = max(
                                max_throttle_slew,
                                abs(float(action[0] - previous_update_action[0])) / elapsed,
                            )
                            max_gimbal_slew = max(
                                max_gimbal_slew,
                                abs(float(action[1] - previous_update_action[1])) / elapsed,
                            )
                    previous_update_action = action.copy()
                    previous_update_time = command_time
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            outcomes.append(str(info["outcome"]))
            touchdown_states.append(np.asarray(state, dtype=np.float64))
            fuel_used.append(float(info["fuel_used_kg"]))
            flight_times.append(float(info["time_s"]))
            control_updates.append(controller.update_count)
    finally:
        env.close()

    states = np.asarray(touchdown_states)
    absolute = np.abs(states)
    counts = dict(sorted(Counter(outcomes).items()))
    successes = counts.get("success", 0)
    episodes = len(initial_states)
    return IntegratedControlResult(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        outcome_counts=counts,
        mean_abs_touchdown_x_m=float(np.mean(absolute[:, 0])),
        max_abs_touchdown_x_m=float(np.max(absolute[:, 0])),
        mean_abs_touchdown_vx_m_s=float(np.mean(absolute[:, 2])),
        max_abs_touchdown_vx_m_s=float(np.max(absolute[:, 2])),
        mean_abs_touchdown_vz_m_s=float(np.mean(absolute[:, 3])),
        p95_abs_touchdown_vz_m_s=float(np.percentile(absolute[:, 3], 95)),
        max_abs_touchdown_vz_m_s=float(np.max(absolute[:, 3])),
        mean_abs_touchdown_theta_deg=float(np.rad2deg(np.mean(absolute[:, 4]))),
        max_abs_touchdown_theta_deg=float(np.rad2deg(np.max(absolute[:, 4]))),
        mean_abs_touchdown_omega_deg_s=float(np.rad2deg(np.mean(absolute[:, 5]))),
        max_abs_touchdown_omega_deg_s=float(np.rad2deg(np.max(absolute[:, 5]))),
        mean_fuel_used_kg=float(np.mean(fuel_used)),
        mean_flight_time_s=float(np.mean(flight_times)),
        mean_control_updates=float(np.mean(control_updates)),
        approach_updates=phase_updates["approach"],
        terminal_updates=phase_updates["terminal"],
        slew_limited_updates=slew_limited_updates,
        max_throttle_slew_per_s=float(max_throttle_slew),
        max_gimbal_slew_deg_s=float(np.rad2deg(max_gimbal_slew)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, help="save the result to a new JSON file")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output file already exists: {args.output}")

    config = load_config(args.config)
    initial_states = sample_integrated_initial_states(config, args.episodes, args.seed)
    result = evaluate_integrated_control(config, initial_states)
    print(f"success: {result.successes}/{result.episodes} ({result.success_rate:.1%})")
    print(f"outcomes: {result.outcome_counts}")
    print(f"mean touchdown |x|: {result.mean_abs_touchdown_x_m:.4f} m")
    print(f"max touchdown |x|: {result.max_abs_touchdown_x_m:.4f} m")
    print(f"mean touchdown |vx|: {result.mean_abs_touchdown_vx_m_s:.4f} m/s")
    print(f"max touchdown |vx|: {result.max_abs_touchdown_vx_m_s:.4f} m/s")
    print(f"mean touchdown |vz|: {result.mean_abs_touchdown_vz_m_s:.4f} m/s")
    print(f"p95 touchdown |vz|: {result.p95_abs_touchdown_vz_m_s:.4f} m/s")
    print(f"max touchdown |vz|: {result.max_abs_touchdown_vz_m_s:.4f} m/s")
    print(f"mean touchdown |theta|: {result.mean_abs_touchdown_theta_deg:.4f} deg")
    print(f"max touchdown |theta|: {result.max_abs_touchdown_theta_deg:.4f} deg")
    print(f"mean touchdown |omega|: {result.mean_abs_touchdown_omega_deg_s:.4f} deg/s")
    print(f"max touchdown |omega|: {result.max_abs_touchdown_omega_deg_s:.4f} deg/s")
    print(f"mean fuel used: {result.mean_fuel_used_kg:.4f} kg")
    print(f"mean flight time: {result.mean_flight_time_s:.4f} s")
    print(f"mean control updates: {result.mean_control_updates:.2f}")
    print(f"phase updates: approach={result.approach_updates}, terminal={result.terminal_updates}")
    print(f"slew-limited updates: {result.slew_limited_updates}")
    print(f"max throttle slew: {result.max_throttle_slew_per_s:.4f} /s")
    print(f"max gimbal slew: {result.max_gimbal_slew_deg_s:.4f} deg/s")

    if args.output is not None:
        payload = {
            "config": str(args.config),
            "seed": args.seed,
            "evaluation": config["integrated_landing_controller"]["evaluation"],
            "result": asdict(result),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
