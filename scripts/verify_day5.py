"""Exercise all Day 5 outcomes and seeded trajectories; optionally save JSON logs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for five episode JSON files")
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    check_env(RocketLandingEnv(config), skip_render_check=True)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = {
        "success": ([0, 0.001, 0, -0.1, 0, 0, 900], [0, 0]),
        "hard_landing": ([0, 0.001, 0, -3, 0, 0, 900], [0, 0]),
        "crash": ([5, 0.001, 0, -1, 0, 0, 900], [0, 0]),
        "fuel_depletion": ([0, 100, 0, 0, 0, 0, 750.1], [1, 0]),
        "timeout": ([0, 100, 0, 0, 0, 0, 900], [0, 0]),
    }
    for expected, (initial, action) in cases.items():
        short = deepcopy(config)
        short["simulation"]["max_time_s"] = 0.055
        env = RocketLandingEnv(short)
        env.reset(seed=5, options={"initial_state": initial})
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if info["outcome"] != expected:
            raise AssertionError(f"{expected}: {info}")
        print(f"{expected}: PASS (t={info['time_s']:.9f} s)")
        if args.output_dir:
            env.save_episode(args.output_dir / f"{expected}.json")
    a, b = RocketLandingEnv(config), RocketLandingEnv(config)
    np.testing.assert_array_equal(a.reset(seed=20260906)[0], b.reset(seed=20260906)[0])
    actions = np.random.default_rng(5).uniform([0, -0.2], [1, 0.2], size=(20, 2))
    for action in actions:
        np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])
    if a.episode_log != b.episode_log:
        raise AssertionError("Seeded logs differ")
    print("Seeded trajectory and log: PASS")
    print("Day 5 verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
