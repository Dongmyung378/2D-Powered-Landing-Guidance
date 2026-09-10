"""Verify simulator conventions, termination rules, and numerical stability."""

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
    """Raised when a verification condition fails."""


@dataclass(frozen=True, slots=True)
class RandomRunSummary:
    """Reproducible summary of the random-action verification."""

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
    """Check coordinate, sign, unit, and propellant-flow conventions."""
    parameters = PlanarDynamicsParameters.from_config(config)
    state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    gimbal = parameters.gimbal_limit_rad

    _require(
        STATE_NAMES == ("x", "z", "vx", "vz", "theta", "omega", "mass")
        and STATE_UNITS == ("m", "m", "m/s", "m/s", "rad", "rad/s", "kg"),
        "state-vector order or units differ from the specification",
    )
    _require(
        ACTION_NAMES == ("throttle", "gimbal_angle") and ACTION_UNITS == ("1", "rad"),
        "control-vector order or units differ from the specification",
    )

    free_fall = state_derivative(0.0, state.as_array(), [0.0, 0.0], parameters)
    _require(
        free_fall[0] == 0.0
        and free_fall[2] == 0.0
        and np.isclose(free_fall[3], -parameters.gravity_m_s2),
        "free-fall gravity is inconsistent with the upward +z convention",
    )

    upright = thrust_vector(state, [1.0, 0.0], parameters)
    positive = thrust_vector(state, [1.0, gimbal], parameters)
    negative = thrust_vector(state, [1.0, -gimbal], parameters)
    _require(
        np.isclose(upright[0], 0.0)
        and upright[1] > 0.0
        and positive[0] > 0.0
        and negative[0] < 0.0,
        "gimbal and horizontal-thrust signs differ from the specification",
    )
    _require(
        angular_acceleration_rad_s2(state, [1.0, gimbal], parameters) < 0.0
        and angular_acceleration_rad_s2(state, [1.0, -gimbal], parameters) > 0.0,
        "gimbal and rotational-torque signs differ from the specification",
    )

    expected_flow = parameters.max_thrust_n / (
        parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    )
    actual_flow = propellant_mass_flow_rate_kg_s(state, [1.0, 0.0], parameters)
    _require(
        np.isclose(actual_flow, expected_flow, rtol=0.0, atol=1e-12),
        "specific-impulse mass flow differs from the specification",
    )
    return (
        "state and control units",
        "gravity and +z sign",
        "gimbal and thrust signs",
        "rotational-torque sign",
        "propellant flow",
    )


def verify_termination_cases(config: dict) -> dict[str, str]:
    """Check every outcome and its terminated or truncated semantics."""
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
        _require(info["outcome"] == expected, f"failed to reproduce {expected}")
        _require(terminated and not truncated, f"incorrect termination flags for {expected}")
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
    _require(info["outcome"] == "timeout", "failed to reproduce timeout")
    _require(truncated and not terminated, "incorrect termination flags for timeout")
    observed["timeout"] = "truncated"
    _require(set(observed) == EXPECTED_OUTCOMES, "termination coverage is incomplete")
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
    _require(np.all(np.isfinite(state)), "state contains NaN or Inf")
    _require(state[1] >= env.ground - tolerance, "state penetrated the ground")
    _require(
        state[6] >= env.parameters.dry_mass_kg - tolerance,
        "mass dropped below dry mass",
    )
    _require(state[6] <= previous_state[6] + tolerance, "mass increased")
    _require(
        previous_time < info["time_s"] <= env.max_time + tolerance,
        "time failed to increase within the configured limit",
    )
    _require(np.isfinite(info["fuel_used_kg"]), "fuel use contains NaN or Inf")
    expected_fuel = initial_mass - state[6]
    _require(
        np.isclose(info["fuel_used_kg"], expected_fuel, rtol=0.0, atol=tolerance),
        "state mass and recorded fuel use differ",
    )
    clipped_action = np.asarray(info["clipped_action"], dtype=np.float64)
    _require(
        env.action_space.contains(clipped_action),
        "clipped action is outside the action space",
    )


def run_random_action_check(
    config: dict, episodes: int = 100, seed: int = 20260907
) -> RandomRunSummary:
    """Run random-action episodes from independently sampled initial states."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
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
            _require(np.all(np.isfinite(state)), "initial state contains NaN or Inf")
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
                raise VerificationError("episode did not finish within the maximum step count")

            outcome = str(info["outcome"])
            _require(outcome in EXPECTED_OUTCOMES, f"unknown outcome: {outcome}")
            episode_log = env.episode_log
            _require(episode_log["outcome"] == outcome, "outcome differs from episode record")
            _require(
                len(episode_log["steps"]) == episode_steps,
                "episode step count differs from the record",
            )
            _require(
                np.array_equal(state, np.asarray(episode_log["steps"][-1]["state"])),
                "returned state differs from the final recorded state",
            )
            outcomes[outcome] += 1
        finally:
            env.close()

    _require(sum(outcomes.values()) == episodes, "random episode count differs from the request")
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
        raise argparse.ArgumentTypeError("value must be an integer of at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file to verify")
    parser.add_argument(
        "--episodes", type=_positive_integer, default=100, help="number of random episodes"
    )
    parser.add_argument("--seed", type=int, default=20260907, help="base seed for reproduction")
    args = parser.parse_args()

    config = load_config(args.config)
    sanity = verify_model_sanity(config)
    termination = verify_termination_cases(config)
    random_run = run_random_action_check(config, args.episodes, args.seed)

    print("Simulator verification passed")
    print(f"- coordinate and physics contracts: {len(sanity)} checks")
    print("- outcomes: " + ", ".join(f"{name}={flag}" for name, flag in termination.items()))
    print(f"- random episodes: {random_run.episodes}, total steps: {random_run.steps}")
    print(
        "- outcome distribution: "
        + ", ".join(f"{key}={value}" for key, value in random_run.outcomes.items())
    )
    print(f"- minimum mass: {random_run.min_mass_kg:.6f} kg")
    print(f"- longest episode: {random_run.max_time_s:.6f} s")
    print("- NaN, Inf, ground penetration, dry-mass violation: none")


if __name__ == "__main__":
    main()
