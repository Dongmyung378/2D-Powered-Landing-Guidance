"""Evaluate horizontal-position and attitude control in wind-free hover."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance import HorizontalAttitudeController, load_config
from powered_landing_guidance.envs import RocketLandingEnv


@dataclass(frozen=True, slots=True)
class HorizontalControlResult:
    episodes: int
    completed_episodes: int
    converged_episodes: int
    convergence_rate: float
    mean_initial_abs_error_m: float
    mean_final_abs_error_m: float
    mean_error_reduction_m: float
    max_target_tilt_deg: float
    max_gimbal_deg: float
    max_abs_altitude_error_m: float


def sample_horizontal_initial_states(
    config: dict,
    episodes: int,
    seed: int,
) -> NDArray[np.float64]:
    """Sample reproducible states with a nonzero horizontal position error."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    settings = config["horizontal_attitude_controller"]
    ranges = settings["evaluation"]
    target_x = float(settings["target_x_m"])
    low_x, high_x = (float(value) for value in ranges["x_m"])
    min_abs_x = float(ranges["min_abs_x_m"])
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=episodes)
    negative_x = rng.uniform(low_x, target_x - min_abs_x, size=episodes)
    positive_x = rng.uniform(target_x + min_abs_x, high_x, size=episodes)

    states = np.zeros((episodes, 7), dtype=np.float64)
    states[:, 0] = np.where(signs < 0.0, negative_x, positive_x)
    states[:, 1] = float(ranges["altitude_m"])
    states[:, 2] = rng.uniform(*ranges["vx_m_s"], size=episodes)
    states[:, 4] = np.deg2rad(rng.uniform(*ranges["theta_deg"], size=episodes))
    states[:, 5] = np.deg2rad(rng.uniform(*ranges["omega_deg_s"], size=episodes))
    states[:, 6] = rng.uniform(*ranges["mass_kg"], size=episodes)
    return states


def evaluate_horizontal_control(
    config: dict,
    initial_states: NDArray[np.float64],
) -> HorizontalControlResult:
    """Run fixed-duration hover trials and measure motion toward the pad."""
    if initial_states.ndim != 2 or initial_states.shape[1] != 7 or len(initial_states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    wind = float(config["environment"]["horizontal_wind_m_s"])
    if not np.isclose(wind, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("horizontal-control evaluation requires zero horizontal wind")

    settings = config["horizontal_attitude_controller"]
    evaluation = settings["evaluation"]
    target_x = float(settings["target_x_m"])
    altitude = float(evaluation["altitude_m"])
    duration = float(evaluation["duration_s"])
    dt = float(config["simulation"]["dt_s"])
    if duration >= float(config["simulation"]["max_time_s"]):
        raise ValueError("evaluation duration must be shorter than simulation max_time_s")
    step_count = round(duration / dt)
    if not np.isclose(step_count * dt, duration, rtol=0.0, atol=1e-12):
        raise ValueError("evaluation duration must be an integer multiple of simulation dt")

    initial_errors: list[float] = []
    final_errors: list[float] = []
    completed = 0
    max_target_tilt = 0.0
    max_gimbal = 0.0
    max_altitude_error = 0.0
    env = RocketLandingEnv(config)
    try:
        for initial_state in initial_states:
            controller = HorizontalAttitudeController.from_config(config)
            state, _ = env.reset(options={"initial_state": initial_state})
            initial_errors.append(abs(float(state[0]) - target_x))
            episode_completed = True
            for _ in range(step_count):
                hover_throttle = (
                    float(state[6])
                    * controller.parameters.gravity_m_s2
                    / controller.parameters.max_thrust_n
                )
                action = controller.command(state, base_throttle=hover_throttle)
                state, _, terminated, truncated, _ = env.step(action)
                max_target_tilt = max(
                    max_target_tilt,
                    abs(float(controller.target_attitude_rad)),
                )
                max_gimbal = max(max_gimbal, abs(float(action[1])))
                max_altitude_error = max(max_altitude_error, abs(float(state[1]) - altitude))
                if terminated or truncated:
                    episode_completed = False
                    break
            completed += int(episode_completed)
            final_errors.append(abs(float(state[0]) - target_x))
    finally:
        env.close()

    initial_array = np.asarray(initial_errors)
    final_array = np.asarray(final_errors)
    converged = int(np.count_nonzero(final_array < initial_array))
    episodes = len(initial_states)
    return HorizontalControlResult(
        episodes=episodes,
        completed_episodes=completed,
        converged_episodes=converged,
        convergence_rate=converged / episodes,
        mean_initial_abs_error_m=float(np.mean(initial_array)),
        mean_final_abs_error_m=float(np.mean(final_array)),
        mean_error_reduction_m=float(np.mean(initial_array - final_array)),
        max_target_tilt_deg=float(np.rad2deg(max_target_tilt)),
        max_gimbal_deg=float(np.rad2deg(max_gimbal)),
        max_abs_altitude_error_m=float(max_altitude_error),
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
    states = sample_horizontal_initial_states(config, args.episodes, args.seed)
    result = evaluate_horizontal_control(config, states)
    print(f"completed: {result.completed_episodes}/{result.episodes}")
    print(f"converged: {result.converged_episodes}/{result.episodes}")
    print(f"convergence rate: {result.convergence_rate:.1%}")
    print(f"mean initial |x error|: {result.mean_initial_abs_error_m:.4f} m")
    print(f"mean final |x error|: {result.mean_final_abs_error_m:.4f} m")
    print(f"mean error reduction: {result.mean_error_reduction_m:.4f} m")
    print(f"max target tilt: {result.max_target_tilt_deg:.4f} deg")
    print(f"max gimbal: {result.max_gimbal_deg:.4f} deg")
    print(f"max altitude deviation: {result.max_abs_altitude_error_m:.4f} m")

    if args.output is not None:
        payload = {
            "config": str(args.config),
            "seed": args.seed,
            "evaluation": config["horizontal_attitude_controller"]["evaluation"],
            "result": asdict(result),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
