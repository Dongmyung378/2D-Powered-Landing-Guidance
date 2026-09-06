"""Run a constant-command episode or replay a saved recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.visualization import (
    load_episode_data,
    save_animation,
    save_episode_data,
    save_time_series,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", type=Path, help="Load JSON/NPZ without running physics")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--nominal", action="store_true", help="Use YAML initial_state")
    initial.add_argument(
        "--initial-state",
        type=float,
        nargs=7,
        metavar=("X", "Z", "VX", "VZ", "THETA", "OMEGA", "MASS"),
        help="Initial state in SI units; angles in radians",
    )
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument("--gimbal-deg", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Save the episode as JSON or NPZ")
    parser.add_argument("--plot", type=Path, help="Save state and control histories as PNG")
    parser.add_argument("--animation", type=Path, help="Save the recorded trajectory as GIF or MP4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    outputs = [path for path in (args.output, args.plot, args.animation) if path]
    output_paths = [str(path.resolve()).casefold() for path in outputs]
    if len(output_paths) != len(set(output_paths)):
        parser.error("output paths must be different")
    existing = [path for path in outputs if path.exists()]
    if existing:
        parser.error(f"output already exists: {existing[0]}")
    if args.replay and (args.nominal or args.initial_state is not None):
        parser.error("--replay cannot be combined with initial-state options")
    if args.replay and (args.throttle != 0.0 or args.gimbal_deg != 0.0):
        parser.error("--replay cannot be combined with control options")

    if args.replay:
        episode = load_episode_data(args.replay)
        if episode["steps"]:
            state = np.asarray(episode["steps"][-1]["state"], dtype=np.float64)
            info = episode["steps"][-1]
        else:
            state = np.asarray(episode["initial_state"], dtype=np.float64)
            info = {"outcome": "running", "time_s": 0.0, "fuel_used_kg": 0.0}
    else:
        env = RocketLandingEnv(load_config(args.config))
        options = {"randomize": not args.nominal}
        if args.initial_state is not None:
            options["initial_state"] = args.initial_state
        try:
            env.reset(seed=args.seed, options=options)
            action = [args.throttle, np.deg2rad(args.gimbal_deg)]
            while True:
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            episode = env.episode_log
        finally:
            env.close()

    print(f"{info['outcome']}: t={info['time_s']:.6f} s, fuel={info['fuel_used_kg']:.6f} kg")
    print(f"State [x, z, vx, vz, theta, omega, mass]: {state.tolist()}")
    if args.output:
        print(f"Saved: {save_episode_data(episode, args.output)}")
    if args.plot:
        print(f"Saved: {save_time_series(episode, args.plot)}")
    if args.animation:
        print(f"Saved: {save_animation(episode, args.animation, fps=args.fps)}")


if __name__ == "__main__":
    main()
