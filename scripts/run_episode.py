"""고정 제어 명령으로 에피소드를 실행하거나 저장된 기록을 재생한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from powered_landing_guidance import load_config
from powered_landing_guidance.controllers import SuicideBurnController
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
    parser.add_argument("--seed", type=int, default=42, help="초기조건을 재현할 seed")
    parser.add_argument("--replay", type=Path, help="물리 계산 없이 JSON 또는 NPZ 기록 재생")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--nominal", action="store_true", help="YAML의 명목 초기조건 사용")
    initial.add_argument(
        "--initial-state",
        type=float,
        nargs=7,
        metavar=("X", "Z", "VX", "VZ", "THETA", "OMEGA", "MASS"),
        help="SI 단위 초기 상태이며 각도는 radian",
    )
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument("--gimbal-deg", type=float, default=0.0)
    parser.add_argument(
        "--controller",
        choices=("suicide-burn",),
        help="상태에 따라 제어 명령을 계산하는 기준 제어기",
    )
    parser.add_argument("--output", type=Path, help="에피소드를 JSON 또는 NPZ로 저장")
    parser.add_argument("--plot", type=Path, help="상태와 제어 기록을 PNG로 저장")
    parser.add_argument("--animation", type=Path, help="기록된 궤적을 GIF 또는 MP4로 저장")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    outputs = [path for path in (args.output, args.plot, args.animation) if path]
    output_paths = [str(path.resolve()).casefold() for path in outputs]
    if len(output_paths) != len(set(output_paths)):
        parser.error("출력 파일 경로는 서로 달라야 합니다")
    existing = [path for path in outputs if path.exists()]
    if existing:
        parser.error(f"출력 파일이 이미 존재합니다: {existing[0]}")
    if args.replay and (args.nominal or args.initial_state is not None):
        parser.error("--replay는 초기조건 옵션과 함께 사용할 수 없습니다")
    if args.replay and (
        args.throttle != 0.0 or args.gimbal_deg != 0.0 or args.controller is not None
    ):
        parser.error("--replay는 제어 명령 옵션과 함께 사용할 수 없습니다")
    if args.controller and (args.throttle != 0.0 or args.gimbal_deg != 0.0):
        parser.error("--controller는 고정 제어 명령과 함께 사용할 수 없습니다")

    burn_controller = None
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
            state, _ = env.reset(seed=args.seed, options=options)
            if args.controller == "suicide-burn":
                burn_controller = SuicideBurnController.from_config(config)
            fixed_action = [args.throttle, np.deg2rad(args.gimbal_deg)]
            while True:
                action = (
                    burn_controller.command(state) if burn_controller is not None else fixed_action
                )
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            episode = env.episode_log
        finally:
            env.close()

    print(
        f"종료 결과={info['outcome']}, 시간={info['time_s']:.6f} s, "
        f"사용 연료={info['fuel_used_kg']:.6f} kg"
    )
    print(f"상태 [x, z, vx, vz, theta, omega, mass]: {state.tolist()}")
    if burn_controller is not None:
        if burn_controller.actual_ignition_height_m is None:
            print("Suicide-burn 점화 없음")
        else:
            print(
                f"Suicide-burn 점화 고도={burn_controller.actual_ignition_height_m:.6f} m, "
                f"분류={burn_controller.ignition_timing}"
            )
    if args.output:
        print(f"저장 완료: {save_episode_data(episode, args.output)}")
    if args.plot:
        print(f"저장 완료: {save_time_series(episode, args.plot)}")
    if args.animation:
        print(f"저장 완료: {save_animation(episode, args.animation, fps=args.fps)}")


if __name__ == "__main__":
    main()
