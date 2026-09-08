"""1주차 시뮬레이터의 좌표계, 종료 조건과 수치 안정성을 검증한다."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance import load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    angular_acceleration_rad_s2,
    propellant_mass_flow_rate_kg_s,
    state_derivative,
    thrust_vector,
)
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    State,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
EXPECTED_OUTCOMES = {"success", "hard_landing", "crash", "fuel_depletion", "timeout"}


class VerificationError(RuntimeError):
    """검증 조건을 만족하지 못했을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class RandomRunSummary:
    """무작위 행동 검증 결과의 재현 가능한 요약값."""

    episodes: int
    steps: int
    outcomes: dict[str, int]
    min_mass_kg: float
    max_time_s: float
    max_abs_state: tuple[float, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_model_sanity(config: dict) -> tuple[str, ...]:
    """좌표계, 부호, 단위와 연료 유량 규약을 독립적으로 확인한다."""
    parameters = PlanarDynamicsParameters.from_config(config)
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    gimbal = parameters.gimbal_limit_rad

    _require(
        STATE_NAMES == ("x", "z", "vx", "vz", "theta", "omega", "mass")
        and STATE_UNITS == ("m", "m", "m/s", "m/s", "rad", "rad/s", "kg"),
        "상태 벡터의 순서 또는 단위가 명세와 다릅니다",
    )
    _require(
        ACTION_NAMES == ("throttle", "gimbal_angle") and ACTION_UNITS == ("1", "rad"),
        "제어 벡터의 순서 또는 단위가 명세와 다릅니다",
    )

    free_fall = state_derivative(0.0, state.as_array(), [0.0, 0.0], parameters)
    _require(
        free_fall[0] == 0.0
        and free_fall[2] == 0.0
        and np.isclose(free_fall[3], -parameters.gravity_m_s2),
        "무추력 상태에서 +z 위쪽 좌표계와 중력 부호가 일치하지 않습니다",
    )

    upright = thrust_vector(state, [1.0, 0.0], parameters)
    positive = thrust_vector(state, [1.0, gimbal], parameters)
    negative = thrust_vector(state, [1.0, -gimbal], parameters)
    _require(
        np.isclose(upright[0], 0.0)
        and upright[1] > 0.0
        and positive[0] > 0.0
        and negative[0] < 0.0,
        "짐벌과 수평 추력의 부호가 명세와 일치하지 않습니다",
    )
    _require(
        angular_acceleration_rad_s2(state, [1.0, gimbal], parameters) < 0.0
        and angular_acceleration_rad_s2(state, [1.0, -gimbal], parameters) > 0.0,
        "짐벌과 회전 토크의 부호가 명세와 일치하지 않습니다",
    )

    expected_flow = parameters.max_thrust_n / (
        parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    )
    actual_flow = propellant_mass_flow_rate_kg_s(state, [1.0, 0.0], parameters)
    _require(
        np.isclose(actual_flow, expected_flow, rtol=0.0, atol=1e-12),
        "비추력으로 계산한 연료 유량이 명세와 일치하지 않습니다",
    )
    return ("상태와 제어 단위", "중력과 +z 부호", "짐벌과 추력 부호", "회전 토크 부호", "연료 유량")


def verify_termination_cases(config: dict) -> dict[str, str]:
    """모든 종료 결과와 terminated/truncated 의미를 확인한다."""
    ground = float(config["simulation"]["ground_z_m"])
    dry_mass = float(config["vehicle"]["dry_mass_kg"])
    cases = {
        "success": ([0.0, ground, 0.0, 0.0, 0.0, 0.0, dry_mass + 100.0], [0.0, 0.0]),
        "hard_landing": (
            [0.0, ground, 0.0, -3.0, 0.0, 0.0, dry_mass + 100.0],
            [0.0, 0.0],
        ),
        "crash": ([2.0, ground, 0.0, 0.0, 0.0, 0.0, dry_mass + 100.0], [0.0, 0.0]),
        "fuel_depletion": ([0.0, ground + 10.0, 0.0, 0.0, 0.0, 0.0, dry_mass], [1.0, 0.0]),
    }
    observed: dict[str, str] = {}
    for expected, (initial_state, action) in cases.items():
        env = RocketLandingEnv(config)
        try:
            env.reset(options={"initial_state": initial_state})
            _, _, terminated, truncated, info = env.step(action)
        finally:
            env.close()
        _require(info["outcome"] == expected, f"{expected} 종료 상황을 재현하지 못했습니다")
        _require(terminated and not truncated, f"{expected}의 종료 플래그가 잘못되었습니다")
        observed[expected] = "terminated"

    timeout_config = deepcopy(config)
    timeout_config["simulation"]["max_time_s"] = 2.0 * float(config["simulation"]["dt_s"])
    env = RocketLandingEnv(timeout_config)
    try:
        env.reset(
            options={"initial_state": [0.0, ground + 100.0, 0.0, 0.0, 0.0, 0.0, dry_mass + 100.0]}
        )
        while True:
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            if terminated or truncated:
                break
    finally:
        env.close()
    _require(info["outcome"] == "timeout", "timeout 종료 상황을 재현하지 못했습니다")
    _require(truncated and not terminated, "timeout의 종료 플래그가 잘못되었습니다")
    observed["timeout"] = "truncated"
    _require(set(observed) == EXPECTED_OUTCOMES, "종료 상황 검증 항목이 누락되었습니다")
    return observed


def _check_random_step(
    env: RocketLandingEnv,
    state: NDArray[np.float64],
    previous_state: NDArray[np.float64],
    previous_time: float,
    initial_mass: float,
    info: dict,
) -> None:
    tolerance = 1e-10
    _require(np.all(np.isfinite(state)), "상태에 NaN 또는 Inf가 발생했습니다")
    _require(state[1] >= env.ground - tolerance, "지면 아래 상태가 반환되었습니다")
    _require(
        state[6] >= env.parameters.dry_mass_kg - tolerance,
        "질량이 건조 질량 아래로 내려갔습니다",
    )
    _require(state[6] <= previous_state[6] + tolerance, "질량이 증가했습니다")
    _require(
        previous_time < info["time_s"] <= env.max_time + tolerance, "시간이 증가하지 않았습니다"
    )
    _require(np.isfinite(info["fuel_used_kg"]), "사용 연료에 NaN 또는 Inf가 발생했습니다")
    expected_fuel = initial_mass - state[6]
    _require(
        np.isclose(info["fuel_used_kg"], expected_fuel, rtol=0.0, atol=tolerance),
        "상태 질량과 사용 연료 기록이 일치하지 않습니다",
    )
    clipped_action = np.asarray(info["clipped_action"], dtype=np.float64)
    _require(env.action_space.contains(clipped_action), "제한된 행동이 허용 범위를 벗어났습니다")


def run_random_action_check(
    config: dict, episodes: int = 100, seed: int = 20260907
) -> RandomRunSummary:
    """서로 다른 초기조건에서 무작위 행동 에피소드를 실행한다."""
    if episodes < 1:
        raise ValueError("episodes는 1 이상이어야 합니다")
    rng = np.random.default_rng(seed)
    outcomes: Counter[str] = Counter()
    total_steps = 0
    min_mass = np.inf
    max_time = 0.0
    max_abs_state = np.zeros(7, dtype=np.float64)

    for episode_index in range(episodes):
        env = RocketLandingEnv(config)
        try:
            state, _ = env.reset(seed=seed + episode_index)
            _require(np.all(np.isfinite(state)), "초기 상태에 NaN 또는 Inf가 있습니다")
            initial_mass = float(state[6])
            episode_steps = 0
            previous_time = 0.0
            max_steps = int(np.ceil(env.max_time / env.dt)) + 1
            for _ in range(max_steps):
                previous_state = state.copy()
                action = rng.uniform(env.action_space.low, env.action_space.high)
                state, _, terminated, truncated, info = env.step(action)
                _check_random_step(
                    env,
                    state,
                    previous_state,
                    previous_time,
                    initial_mass,
                    info,
                )
                total_steps += 1
                episode_steps += 1
                min_mass = min(min_mass, float(state[6]))
                max_time = max(max_time, float(info["time_s"]))
                max_abs_state = np.maximum(max_abs_state, np.abs(state))
                previous_time = float(info["time_s"])
                if terminated or truncated:
                    break
            else:
                raise VerificationError("최대 허용 스텝 안에 에피소드가 종료되지 않았습니다")

            outcome = str(info["outcome"])
            _require(outcome in EXPECTED_OUTCOMES, f"알 수 없는 종료 결과입니다: {outcome}")
            episode_log = env.episode_log
            _require(episode_log["outcome"] == outcome, "종료 결과와 에피소드 기록이 다릅니다")
            _require(
                len(episode_log["steps"]) == episode_steps, "에피소드 스텝 수가 기록과 다릅니다"
            )
            _require(
                np.array_equal(state, np.asarray(episode_log["steps"][-1]["state"])),
                "반환 상태와 마지막 에피소드 기록이 다릅니다",
            )
            outcomes[outcome] += 1
        finally:
            env.close()

    _require(sum(outcomes.values()) == episodes, "무작위 에피소드 수가 요청값과 다릅니다")
    return RandomRunSummary(
        episodes=episodes,
        steps=total_steps,
        outcomes=dict(sorted(outcomes.items())),
        min_mass_kg=float(min_mass),
        max_time_s=max_time,
        max_abs_state=tuple(float(value) for value in max_abs_state),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력해야 합니다")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="검증할 YAML 설정 파일")
    parser.add_argument(
        "--episodes", type=_positive_integer, default=100, help="무작위 에피소드 수"
    )
    parser.add_argument("--seed", type=int, default=20260907, help="검증 재현용 기준 seed")
    args = parser.parse_args()

    config = load_config(args.config)
    sanity = verify_model_sanity(config)
    termination = verify_termination_cases(config)
    random_run = run_random_action_check(config, args.episodes, args.seed)

    print("1주차 검증 통과")
    print(f"- 좌표계와 물리 규약: {len(sanity)}개 항목")
    print("- 종료 상황: " + ", ".join(f"{name}={flag}" for name, flag in termination.items()))
    print(f"- 무작위 에피소드: {random_run.episodes}회, 총 {random_run.steps}스텝")
    print(
        "- 종료 결과 분포: "
        + ", ".join(f"{key}={value}" for key, value in random_run.outcomes.items())
    )
    print(f"- 최소 질량: {random_run.min_mass_kg:.6f} kg")
    print(f"- 최장 에피소드: {random_run.max_time_s:.6f} s")
    print("- NaN, Inf, 지면 침범, 건조 질량 침범: 없음")


if __name__ == "__main__":
    main()
