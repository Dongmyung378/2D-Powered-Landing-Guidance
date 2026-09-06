"""Seeded Gymnasium environment for planar rocket landing.

Ground contact uses the model's point-position z, not landing-leg geometry.
Physics is integrated with the existing Euler/RK4 model, including fuel cutoff.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    applied_thrust_n,
    clip_control,
    propellant_mass_flow_rate_kg_s,
    simulate_planar,
)
from powered_landing_guidance.model import (
    ACTION_NAMES,
    ACTION_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    State,
)

INITIAL_KEYS = ("x_m", "z_m", "vx_m_s", "vz_m_s", "theta_deg", "omega_deg_s", "mass_kg")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


class RocketLandingEnv(gym.Env):
    """Physical SI actions/observations; sparse outcome reward, no rendering yet."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = deepcopy(config)
        validate_config(self.config)
        self.parameters = PlanarDynamicsParameters.from_config(self.config)
        sim = self.config["simulation"]
        self.dt = _number(sim["dt_s"], "dt_s")
        self.max_time = _number(sim["max_time_s"], "max_time_s")
        self.ground = _number(sim["ground_z_m"], "ground_z_m")
        self.method = sim["integrator"]
        if self.method not in ("euler", "rk4"):
            raise ValueError("integrator must be euler or rk4")
        self._nominal = self._initial_vector(self.config["initial_state"])
        self._ranges = self._sampling_ranges(self.config.get("initial_state_sampling", {}))
        landing = self.config["landing_success"]
        self._limits = np.array(
            [
                _number(landing[key], key)
                for key in (
                    "max_abs_x_m",
                    "max_abs_vx_m_s",
                    "max_abs_vz_m_s",
                    "max_abs_theta_deg",
                    "max_abs_omega_deg_s",
                )
            ]
        )
        self._limits[3:] = np.deg2rad(self._limits[3:])
        for key in (
            "require_mass_at_or_above_dry_mass",
            "evaluate_immediately_before_ground_contact",
        ):
            if landing.get(key) is not True:
                raise ValueError(f"landing_success.{key} must be true for this model")
        p = self.parameters
        self.action_space = spaces.Box(
            low=np.array([p.throttle_min, -p.gimbal_limit_rad]),
            high=np.array([p.throttle_max, p.gimbal_limit_rad]),
            dtype=np.float64,
        )
        low = np.full(7, -np.inf)
        low[1], low[6] = self.ground, p.dry_mass_kg
        self.observation_space = spaces.Box(low=low, high=np.full(7, np.inf), dtype=np.float64)
        self._state: NDArray[np.float64] | None = None
        self._done = False
        self._time = 0.0
        self._log: dict[str, Any] = {}

    def _initial_vector(self, values: dict[str, Any]) -> NDArray[np.float64]:
        if not isinstance(values, dict) or set(values) != set(INITIAL_KEYS):
            raise ValueError("initial_state must contain exactly the seven documented fields")
        vector = np.array([_number(values[key], key) for key in INITIAL_KEYS])
        vector[4:6] = np.deg2rad(vector[4:6])
        return self._valid_state(vector)

    def _valid_state(self, values: Any) -> NDArray[np.float64]:
        vector = State.from_array(values).as_array()
        if vector[1] < self.ground or vector[6] < self.parameters.dry_mass_kg:
            raise ValueError("initial state must be above ground and at or above dry mass")
        return vector

    def _sampling_ranges(self, values: Any) -> dict[str, tuple[float, float]]:
        if not isinstance(values, dict) or set(values) - set(INITIAL_KEYS):
            raise ValueError("initial_state_sampling contains unknown fields")
        ranges = {}
        for key, bounds in values.items():
            if not isinstance(bounds, list | tuple) or len(bounds) != 2:
                raise ValueError(f"sampling {key} requires [low, high]")
            lo, hi = (_number(v, key) for v in bounds)
            if lo > hi:
                raise ValueError(f"sampling {key} requires low <= high")
            if key == "z_m" and lo < self.ground:
                raise ValueError("sampling altitude cannot be below ground")
            if key == "mass_kg" and lo < self.parameters.dry_mass_kg:
                raise ValueError("sampling mass cannot be below dry mass")
            ranges[key] = (lo, hi)
        return ranges

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else options
        if set(options) - {"initial_state", "randomize"}:
            raise ValueError("unknown reset option")
        randomize = options.get("randomize", True)
        if not isinstance(randomize, bool):
            raise ValueError("randomize must be boolean")
        if "initial_state" in options:
            # Explicit vectors are already SI/radians, unlike YAML values.
            state = self._valid_state(options["initial_state"])
        elif randomize:
            values = dict(self.config["initial_state"])
            for key in INITIAL_KEYS:
                if key in self._ranges:
                    values[key] = float(self.np_random.uniform(*self._ranges[key]))
            state = self._initial_vector(values)
        else:
            state = self._nominal.copy()
        self._state, self._time, self._done = state, 0.0, False
        self._log = {
            "schema_version": 1,
            "config": deepcopy(self.config),
            "seed_argument": seed,
            "rng_state_after_reset": deepcopy(self.np_random.bit_generator.state),
            "state_names": list(STATE_NAMES),
            "state_units": list(STATE_UNITS),
            "action_names": list(ACTION_NAMES),
            "action_units": list(ACTION_UNITS),
            "initial_state": state.tolist(),
            "steps": [],
            "outcome": "running",
        }
        return state.copy(), self._info("running")

    def _info(self, outcome: str) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "is_success": outcome == "success",
            "time_s": self._time,
            "fuel_used_kg": float(self._log["initial_state"][6] - self._state[6]),
        }

    def _contact_outcome(self, state: NDArray[np.float64]) -> str:
        # Compare physical attitude modulo a full turn, but retain unwrapped logs.
        theta = np.arctan2(np.sin(state[4]), np.cos(state[4]))
        checks = np.abs([state[0], state[2], state[3], theta, state[5]]) <= self._limits
        if np.all(checks):
            return "success"
        if checks[0] and checks[3] and checks[4]:
            return "hard_landing"
        return "crash"

    def _advance(self, state, command, duration) -> NDArray[np.float64]:
        if duration <= 0.0:
            return state.copy()
        result = simulate_planar(
            State.from_array(state),
            command,
            self.parameters,
            duration,
            duration,
            method=self.method,
        )[1][-1]
        return State.from_array(result).as_array()

    def _contact(self, state, command, duration, end):
        """Locate first contact in a small integration interval.

        A conservative acceleration bound also checks a possible dip below ground
        when both endpoints are positive (descent followed by powered ascent).
        """
        if state[1] <= self.ground:
            return 0.0, state.copy()
        bracket_end = duration
        if end[1] > self.ground:
            height_tolerance = (
                64 * np.finfo(float).eps * max(1.0, abs(self.ground), abs(state[1]), abs(end[1]))
            )
            if end[1] - self.ground <= height_tolerance and end[3] <= 0:
                contact_state = end.copy()
                contact_state[1] = self.ground
                return duration, contact_state
            max_accel = self.parameters.max_thrust_n / self.parameters.dry_mass_kg
            max_accel += self.parameters.gravity_m_s2
            if min(state[1], end[1]) - self.ground > max_accel * duration**2 / 8:
                return None
            minimum = minimize_scalar(
                lambda t: self._advance(state, command, t)[1],
                bounds=(0.0, duration),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if minimum.fun > self.ground:
                return None
            bracket_end = float(minimum.x)
        contact_time = brentq(
            lambda t: self._advance(state, command, t)[1] - self.ground,
            0.0,
            bracket_end,
            xtol=1e-12,
        )
        contact_state = self._advance(state, command, contact_time)
        contact_state[1] = self.ground
        return contact_time, contact_state

    def step(self, action):
        if self._state is None or self._done:
            raise RuntimeError("call reset before step and after episode termination")
        command = clip_control(action, self.parameters)
        raw = np.asarray(action, dtype=np.float64).copy()
        before = self._state.copy()
        target = min(self._time + self.dt, self.max_time)
        state, time = before.copy(), self._time
        outcome = "running"
        mass_tol = self.parameters.mass_tolerance_kg
        if state[1] <= self.ground:
            outcome = self._contact_outcome(state)
        elif state[6] <= self.parameters.dry_mass_kg + mass_tol:
            outcome = "fuel_depletion"
        while outcome == "running" and time < target:
            # Fixed upper substep bound limits contact search and rotation error.
            h = min(target - time, 0.02)
            current = State.from_array(state)
            rate = propellant_mass_flow_rate_kg_s(current, command, self.parameters)
            fuel_time = (state[6] - self.parameters.dry_mass_kg) / rate if rate > 0 else np.inf
            h = min(h, fuel_time)
            end = self._advance(state, command, h)
            contact = self._contact(state, command, h, end)
            if contact is not None:
                delta, state = contact
                time += delta
                outcome = self._contact_outcome(state)
                break
            state = end
            time = target if h == target - time else time + h
            if h == fuel_time or state[6] <= self.parameters.dry_mass_kg + mass_tol:
                state[6] = self.parameters.dry_mass_kg
                outcome = "fuel_depletion"
        if outcome == "running" and time >= self.max_time:
            outcome = "timeout"
        terminated = outcome not in ("running", "timeout")
        truncated = outcome == "timeout"
        reward = 1.0 if outcome == "success" else (-1.0 if terminated else 0.0)
        self._state, self._time, self._done = state, time, terminated or truncated
        info = self._info(outcome)
        info["commanded_action"] = raw.tolist()
        info["clipped_action"] = command.as_array().tolist()
        info["thrust_start_n"] = applied_thrust_n(
            State.from_array(before), command, self.parameters
        )
        info["thrust_end_n"] = applied_thrust_n(State.from_array(state), command, self.parameters)
        self._log["steps"].append(
            {
                "index": len(self._log["steps"]),
                "state": state.tolist(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                **deepcopy(info),
            }
        )
        self._log["outcome"] = outcome
        return state.copy(), reward, terminated, truncated, info

    @property
    def episode_log(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible snapshot, including unfinished episodes."""
        return deepcopy(self._log)

    def save_episode(self, path: str | Path) -> None:
        """Write a snapshot; never overwrite an existing file."""
        if self._state is None:
            raise RuntimeError("call reset before saving an episode")
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self._log, stream, ensure_ascii=False, indent=2, allow_nan=False)
