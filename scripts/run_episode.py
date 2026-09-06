"""Run a constant-command episode; this is not an automatic landing controller."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--output", type=Path, help="Save JSON without overwriting an existing file"
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")

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
        print(f"{info['outcome']}: t={info['time_s']:.6f} s, fuel={info['fuel_used_kg']:.6f} kg")
        print(f"State [x, z, vx, vz, theta, omega, mass]: {state.tolist()}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            env.save_episode(args.output)
            print(f"Saved: {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
