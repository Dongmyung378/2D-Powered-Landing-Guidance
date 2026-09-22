"""Reproducible initial-condition sets for controller comparisons."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

INITIAL_CONDITION_SAMPLER = "integrated_uniform_v1"
_DIGEST_PREFIX = b"powered-landing-guidance:integrated-initial-conditions:v1\n"
_CONTROLLER_DIGEST_PREFIX = b"powered-landing-guidance:integrated-controller:v1\n"
_TEACHER_DIGEST_PREFIX = b"powered-landing-guidance:planar-teacher:v1\n"
_TEACHER_CONFIG_KEYS = (
    "conventions",
    "simulation",
    "vehicle",
    "environment",
    "landing_success",
    "integrated_landing_controller",
    "optimal_control",
    "teacher_pipeline",
)


def sample_integrated_initial_states(
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> NDArray[np.float64]:
    """Sample a reproducible medium-difficulty batch from configured ranges."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    settings = config["integrated_landing_controller"]
    ranges = settings["evaluation"]
    target_x = float(settings["target_x_m"])
    low_x, high_x = (float(value) for value in ranges["x_m"])
    min_abs_x = float(ranges["min_abs_x_m"])
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=episodes)
    negative_x = rng.uniform(low_x, target_x - min_abs_x, size=episodes)
    positive_x = rng.uniform(target_x + min_abs_x, high_x, size=episodes)

    states = np.empty((episodes, 7), dtype=np.float64)
    states[:, 0] = np.where(signs < 0.0, negative_x, positive_x)
    states[:, 1] = rng.uniform(*ranges["z_m"], size=episodes)
    states[:, 2] = rng.uniform(*ranges["vx_m_s"], size=episodes)
    states[:, 3] = rng.uniform(*ranges["vz_m_s"], size=episodes)
    states[:, 4] = np.deg2rad(rng.uniform(*ranges["theta_deg"], size=episodes))
    states[:, 5] = np.deg2rad(rng.uniform(*ranges["omega_deg_s"], size=episodes))
    states[:, 6] = rng.uniform(*ranges["mass_kg"], size=episodes)
    return states


def initial_condition_sha256(initial_states: NDArray[np.float64]) -> str:
    """Hash a state matrix with an explicit schema and byte order."""
    states = np.asarray(initial_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    if not np.all(np.isfinite(states)):
        raise ValueError("initial_states must contain only finite values")

    canonical = np.ascontiguousarray(states, dtype="<f8")
    shape = np.asarray(canonical.shape, dtype="<u8")
    digest = hashlib.sha256()
    digest.update(_DIGEST_PREFIX)
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def integrated_controller_sha256(config: dict[str, Any]) -> str:
    """Hash the integrated-controller settings used by the frozen baseline."""
    encoded = json.dumps(
        config["integrated_landing_controller"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(_CONTROLLER_DIGEST_PREFIX + encoded).hexdigest()


def teacher_configuration_sha256(config: dict[str, Any]) -> str:
    """Hash every setting that defines generation and replay of the planar teacher."""
    try:
        payload = {key: config[key] for key in _TEACHER_CONFIG_KEYS}
    except KeyError as error:
        raise ValueError(f"teacher configuration is missing {error.args[0]}") from error
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(_TEACHER_DIGEST_PREFIX + encoded).hexdigest()


def frozen_baseline_initial_states(config: dict[str, Any]) -> NDArray[np.float64]:
    """Recreate a frozen benchmark set and reject any protocol drift."""
    protocol = config.get("baseline_protocol")
    if not isinstance(protocol, dict) or protocol.get("frozen") is not True:
        raise ValueError("configuration does not contain a frozen baseline protocol")

    expected_controller_hash = protocol.get("controller_sha256")
    controller_hash = integrated_controller_sha256(config)
    if controller_hash != expected_controller_hash:
        raise ValueError("frozen integrated-controller settings have changed")

    conditions = protocol.get("initial_conditions")
    if not isinstance(conditions, dict):
        raise ValueError("baseline protocol is missing initial_conditions")
    if conditions.get("sampler") != INITIAL_CONDITION_SAMPLER:
        raise ValueError(f"unsupported initial-condition sampler: {conditions.get('sampler')}")

    states = sample_integrated_initial_states(
        config,
        episodes=conditions["episodes"],
        seed=conditions["seed"],
    )
    if initial_condition_sha256(states) != conditions.get("sha256"):
        raise ValueError("frozen initial-condition set does not match its SHA-256")
    states.flags.writeable = False
    return states


def frozen_teacher_initial_states(config: dict[str, Any]) -> NDArray[np.float64]:
    """Recreate the frozen Day 21 nominal set and reject any teacher-setting drift."""
    protocol = config.get("teacher_protocol")
    if not isinstance(protocol, dict) or protocol.get("frozen") is not True:
        raise ValueError("configuration does not contain a frozen teacher protocol")

    expected_configuration_hash = protocol.get("configuration_sha256")
    configuration_hash = teacher_configuration_sha256(config)
    if configuration_hash != expected_configuration_hash:
        raise ValueError("frozen teacher settings have changed")

    conditions = protocol.get("nominal_test_set")
    if not isinstance(conditions, dict):
        raise ValueError("teacher protocol is missing nominal_test_set")
    pipeline = config["teacher_pipeline"]
    for key in ("sampler", "episodes", "seed"):
        if conditions.get(key) != pipeline.get(key):
            raise ValueError(f"teacher nominal test set does not match teacher_pipeline.{key}")
    if conditions.get("sampler") != INITIAL_CONDITION_SAMPLER:
        raise ValueError(f"unsupported initial-condition sampler: {conditions.get('sampler')}")

    states = sample_integrated_initial_states(
        config,
        episodes=conditions["episodes"],
        seed=conditions["seed"],
    )
    if initial_condition_sha256(states) != conditions.get("sha256"):
        raise ValueError("frozen teacher initial-condition set does not match its SHA-256")
    states.flags.writeable = False
    return states
