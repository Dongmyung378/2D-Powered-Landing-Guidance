"""Compare vertical-velocity PID gains on identical initial conditions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance import load_config
from powered_landing_guidance.controllers import VerticalVelocityPIDController
from powered_landing_guidance.envs import RocketLandingEnv


@dataclass(frozen=True, slots=True)
class GainSweepResult:
    kp: float
    ki: float
    kd: float
    episodes: int
    successes: int
    success_rate: float
    mean_touchdown_speed_m_s: float
    mean_target_error_m_s: float
    p95_touchdown_speed_m_s: float
    mean_fuel_used_kg: float


def sample_vertical_initial_states(config: dict, episodes: int, seed: int) -> NDArray[np.float64]:
    """Sample reproducible vertical-only states from the configured evaluation range."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    ranges = config["vertical_velocity_controller"]["evaluation"]
    rng = np.random.default_rng(seed)
    states = np.zeros((episodes, 7), dtype=np.float64)
    states[:, 1] = rng.uniform(*ranges["altitude_m"], size=episodes)
    states[:, 3] = rng.uniform(*ranges["vertical_speed_m_s"], size=episodes)
    states[:, 6] = rng.uniform(*ranges["mass_kg"], size=episodes)
    return states


def evaluate_gains(
    config: dict,
    initial_states: NDArray[np.float64],
    *,
    kp: float,
    ki: float,
    kd: float,
) -> GainSweepResult:
    """Evaluate one gain set over a fixed batch of vertical initial states."""
    if initial_states.ndim != 2 or initial_states.shape[1] != 7 or len(initial_states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    outcomes: list[str] = []
    touchdown_speeds: list[float] = []
    fuel_used: list[float] = []
    env = RocketLandingEnv(config)
    try:
        for initial_state in initial_states:
            controller = VerticalVelocityPIDController.from_config(
                config,
                kp=kp,
                ki=ki,
                kd=kd,
            )
            state, _ = env.reset(options={"initial_state": initial_state})
            while True:
                state, _, terminated, truncated, info = env.step(controller.command(state))
                if terminated or truncated:
                    break
            outcomes.append(info["outcome"])
            touchdown_speeds.append(abs(float(state[3])))
            fuel_used.append(float(info["fuel_used_kg"]))
    finally:
        env.close()

    episodes = len(initial_states)
    successes = outcomes.count("success")
    touchdown_target = config["vertical_velocity_controller"]["profile"]["touchdown_speed_m_s"]
    return GainSweepResult(
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        mean_touchdown_speed_m_s=float(np.mean(touchdown_speeds)),
        mean_target_error_m_s=float(
            np.mean(np.abs(np.asarray(touchdown_speeds) - touchdown_target))
        ),
        p95_touchdown_speed_m_s=float(np.percentile(touchdown_speeds, 95)),
        mean_fuel_used_kg=float(np.mean(fuel_used)),
    )


def sweep_gains(
    config: dict,
    initial_states: NDArray[np.float64],
    kp_values: list[float],
    ki_values: list[float],
    kd_values: list[float],
) -> list[GainSweepResult]:
    """Return gain results ordered by success, target error, then fuel use."""
    if not kp_values or not ki_values or not kd_values:
        raise ValueError("each gain list must contain at least one value")
    results = [
        evaluate_gains(config, initial_states, kp=kp, ki=ki, kd=kd)
        for kp, ki, kd in product(kp_values, ki_values, kd_values)
    ]
    return sorted(
        results,
        key=lambda result: (
            -result.success_rate,
            result.mean_target_error_m_s,
            result.mean_fuel_used_kg,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--kp", type=float, nargs="+")
    parser.add_argument("--ki", type=float, nargs="+")
    parser.add_argument("--kd", type=float, nargs="+")
    parser.add_argument("--output", type=Path, help="save results to a new JSON file")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output file already exists: {args.output}")

    config = load_config(args.config)
    sweep = config["vertical_velocity_controller"]["gain_sweep"]
    kp_values = sweep["kp"] if args.kp is None else args.kp
    ki_values = sweep["ki"] if args.ki is None else args.ki
    kd_values = sweep["kd"] if args.kd is None else args.kd
    states = sample_vertical_initial_states(config, args.episodes, args.seed)
    results = sweep_gains(config, states, kp_values, ki_values, kd_values)

    print("kp       ki       kd       success   mean |vz|  target err   p95 |vz|  mean fuel")
    for result in results:
        print(
            f"{result.kp:<8.4f} {result.ki:<8.4f} {result.kd:<8.4f} "
            f"{result.successes:>3}/{result.episodes:<3} "
            f"{result.mean_touchdown_speed_m_s:>9.4f} "
            f"{result.mean_target_error_m_s:>11.4f} "
            f"{result.p95_touchdown_speed_m_s:>10.4f} "
            f"{result.mean_fuel_used_kg:>10.4f}"
        )

    if args.output is not None:
        payload = {
            "config": str(args.config),
            "seed": args.seed,
            "episodes": args.episodes,
            "evaluation_range": config["vertical_velocity_controller"]["evaluation"],
            "results": [asdict(result) for result in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
