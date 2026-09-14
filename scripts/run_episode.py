"""Run a landing episode or replay a saved record."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.controllers import (
    IntegratedLandingController,
    SuicideBurnController,
    VerticalVelocityPIDController,
)
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
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible initialization")
    parser.add_argument("--replay", type=Path, help="replay a JSON or NPZ record without physics")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument(
        "--nominal", action="store_true", help="use the nominal YAML initial state"
    )
    initial.add_argument(
        "--initial-state",
        type=float,
        nargs=7,
        metavar=("X", "Z", "VX", "VZ", "THETA", "OMEGA", "MASS"),
        help="initial SI state with angles in radians",
    )
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument("--gimbal-deg", type=float, default=0.0)
    parser.add_argument(
        "--controller",
        choices=("integrated-pid", "suicide-burn", "velocity-pid"),
        help="baseline controller that computes commands from state",
    )
    parser.add_argument("--output", type=Path, help="save the episode as JSON or NPZ")
    parser.add_argument("--plot", type=Path, help="save state and control histories as PNG")
    parser.add_argument("--animation", type=Path, help="save the trajectory as GIF or MP4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    outputs = [path for path in (args.output, args.plot, args.animation) if path]
    output_paths = [str(path.resolve()).casefold() for path in outputs]
    if len(output_paths) != len(set(output_paths)):
        parser.error("output paths must be distinct")
    existing = [path for path in outputs if path.exists()]
    if existing:
        parser.error(f"output file already exists: {existing[0]}")
    if args.replay and (args.nominal or args.initial_state is not None):
        parser.error("--replay cannot be combined with initial-state options")
    if args.replay and (
        args.throttle != 0.0 or args.gimbal_deg != 0.0 or args.controller is not None
    ):
        parser.error("--replay cannot be combined with control options")
    if args.controller and (args.throttle != 0.0 or args.gimbal_deg != 0.0):
        parser.error("--controller cannot be combined with fixed control commands")

    controller = None
    if args.replay:
        episode = load_episode_data(args.replay)
        if episode["steps"]:
            state = np.asarray(episode["steps"][-1]["state"], dtype=np.float64)
            info = episode["steps"][-1]
        else:
            state = np.asarray(episode["initial_state"], dtype=np.float64)
            info = {"outcome": "running", "time_s": 0.0, "fuel_used_kg": 0.0}
    else:
        config = load_config(args.config)
        env = RocketLandingEnv(config)
        options = {"randomize": not args.nominal}
        if args.initial_state is not None:
            options["initial_state"] = args.initial_state
        try:
            state, info = env.reset(seed=args.seed, options=options)
            if args.controller == "suicide-burn":
                controller = SuicideBurnController.from_config(config)
            elif args.controller == "velocity-pid":
                controller = VerticalVelocityPIDController.from_config(config)
            elif args.controller == "integrated-pid":
                controller = IntegratedLandingController.from_config(config)
            fixed_action = [args.throttle, np.deg2rad(args.gimbal_deg)]
            while True:
                if isinstance(controller, IntegratedLandingController):
                    action = controller.command(state, time_s=float(info["time_s"]))
                else:
                    action = controller.command(state) if controller is not None else fixed_action
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            episode = env.episode_log
        finally:
            env.close()

    print(
        f"outcome={info['outcome']}, time={info['time_s']:.6f} s, "
        f"fuel used={info['fuel_used_kg']:.6f} kg"
    )
    print(f"state [x, z, vx, vz, theta, omega, mass]: {state.tolist()}")
    if isinstance(controller, SuicideBurnController):
        if controller.actual_ignition_height_m is None:
            print("Suicide-burn ignition: none")
        else:
            print(
                f"Suicide-burn ignition altitude={controller.actual_ignition_height_m:.6f} m, "
                f"classification={controller.ignition_timing}"
            )
    elif isinstance(controller, VerticalVelocityPIDController):
        print(
            f"target vertical speed={controller.target_vertical_speed_m_s:.6f} m/s, "
            f"final throttle={controller.throttle:.6f}, saturated={controller.saturated}"
        )
    elif isinstance(controller, IntegratedLandingController):
        print(
            f"final phase={controller.phase}, control updates={controller.update_count}, "
            f"held action={controller.held_action.tolist()}"
        )
    if args.output:
        print(f"saved: {save_episode_data(episode, args.output)}")
    if args.plot:
        print(f"saved: {save_time_series(episode, args.plot)}")
    if args.animation:
        print(f"saved: {save_animation(episode, args.animation, fps=args.fps)}")


if __name__ == "__main__":
    main()
