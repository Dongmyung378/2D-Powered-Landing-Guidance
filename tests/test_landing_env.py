"""Day 5 API, event timing, reproducibility and logging regression tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


def env_with(**simulation):
    config = deepcopy(CONFIG)
    config["simulation"].update(simulation)
    return RocketLandingEnv(config)


def test_gymnasium_api_checker():
    env = env_with()
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize(
    ("initial", "outcome"),
    [
        ([0, 0.001, 0, -0.1, 0, 0, 900], "success"),
        ([0, 0.001, 0, -3, 0, 0, 900], "hard_landing"),
        ([3, 0.001, 0, -0.1, 0, 0, 900], "crash"),
        ([0, 0.001, 0, -0.1, np.deg2rad(10), 0, 900], "crash"),
        ([0, 0.001, 0, -0.1, 0, np.deg2rad(10), 900], "crash"),
        ([0, 10, 0, 0, 0, 0, 750], "fuel_depletion"),
    ],
)
def test_outcomes_and_terminal_log(initial, outcome):
    env = env_with()
    env.reset(seed=5, options={"initial_state": initial})
    obs, reward, terminated, truncated, info = env.step([0, 0])
    assert terminated and not truncated
    assert info["outcome"] == outcome
    assert reward == (1 if outcome == "success" else -1)
    assert env.observation_space.contains(obs)
    assert env.episode_log["steps"][-1]["outcome"] == outcome
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])


def test_contact_uses_impact_velocity_and_analytic_time():
    env = env_with(dt_s=0.5)
    env.reset(options={"initial_state": [0, 1, 0, -1, 0, 0, 900]})
    obs, _, terminated, _, info = env.step([0, 0])
    g = CONFIG["simulation"]["gravity_m_s2"]
    impact_time = (np.sqrt(1 + 2 * g) - 1) / g
    assert terminated
    assert info["time_s"] == pytest.approx(impact_time, abs=1e-10)
    assert obs[3] == pytest.approx(-1 - g * impact_time, abs=1e-9)
    assert obs[1] == 0.0
    assert info["outcome"] == "hard_landing"


def test_contact_during_dip_with_both_endpoints_above_ground():
    env = env_with()
    env.reset(options={"initial_state": [0, 0.0001, 0, -0.05, 0, 0, 1000]})
    obs, _, terminated, _, info = env.step([1, 0])
    assert terminated and info["outcome"] == "success"
    assert 0 < info["time_s"] < 0.01
    assert obs[3] < 0


def test_fuel_event_ends_at_exact_burnout_time():
    env = env_with(dt_s=0.5)
    env.reset(options={"initial_state": [0, 100, 0, 0, 0, 0, 751]})
    obs, _, terminated, truncated, info = env.step([1, 0.1])
    p = env.parameters
    expected_time = p.specific_impulse_s * p.standard_gravity_m_s2 / p.max_thrust_n
    assert terminated and not truncated and info["outcome"] == "fuel_depletion"
    assert info["time_s"] == pytest.approx(expected_time, abs=1e-10)
    assert obs[6] == 750
    assert info["thrust_start_n"] == 20000
    assert info["thrust_end_n"] == 0


def test_contact_precedes_later_fuel_event_and_timeout():
    env = env_with(dt_s=0.02, max_time_s=0.03)
    env.reset(options={"initial_state": [0, 0.001, 0, -1, 0, 0, 750.1]})
    _, _, terminated, truncated, info = env.step([1, 0])
    assert terminated and not truncated and info["outcome"] == "success"
    assert info["time_s"] < 0.002


def test_contact_priority_at_initial_ground_and_dry_mass():
    env = env_with()
    env.reset(options={"initial_state": [0, 0, 0, 0, 0, 0, 750]})
    _, _, _, _, info = env.step([1, 0])
    assert info["outcome"] == "success" and info["time_s"] == 0


def test_timeout_shortens_final_step_and_truncates():
    env = env_with(dt_s=0.02, max_time_s=0.055)
    env.reset(options={"randomize": False})
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step([0, 0])
    assert not terminated and truncated and reward == 0
    assert info["outcome"] == "timeout" and info["time_s"] == 0.055


def test_same_seed_reproduces_initial_state_trajectory_and_log():
    a, b = env_with(), env_with()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)
    for action in ([0.2, 0.01], [0.7, -0.03], [0, 0]):
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:] == rb[1:]
    assert a.episode_log == b.episode_log
    different, _ = b.reset(seed=43)
    assert not np.array_equal(oa, different)
    continued, _ = a.reset()
    assert not np.array_equal(oa, continued)


def test_sampling_bounds_and_degree_conversion():
    env = env_with()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert -20 <= obs[0] <= 20 and 80 <= obs[1] <= 120
        assert abs(obs[4]) <= np.deg2rad(5)
        assert abs(obs[5]) <= np.deg2rad(2)
        assert 950 <= obs[6] <= 1000
    config = deepcopy(CONFIG)
    config["initial_state"]["theta_deg"] = 30
    nominal, _ = RocketLandingEnv(config).reset(options={"randomize": False})
    assert nominal[4] == pytest.approx(np.pi / 6)


def test_actions_are_clipped_and_recorded_without_aliasing():
    env = env_with()
    obs, _ = env.reset(seed=1)
    initial = obs.copy()
    obs[:] = 0
    action = np.array([2.0, 1.0])
    _, _, _, _, info = env.step(action)
    assert info["clipped_action"] == [1.0, env.parameters.gimbal_limit_rad]
    assert info["commanded_action"] == [2.0, 1.0]
    action[:] = 0
    info["clipped_action"][0] = 999
    snapshot = env.episode_log
    assert snapshot["initial_state"] == initial.tolist()
    assert snapshot["steps"][0]["commanded_action"] == [2, 1]
    snapshot["steps"].clear()
    assert len(env.episode_log["steps"]) == 1


def test_json_log_can_replay_exactly(tmp_path):
    env = env_with(dt_s=0.02, max_time_s=0.05)
    env.reset(seed=123)
    for action in ([0.1, 0.01], [0.3, -0.02], [0.2, 0]):
        env.step(action)
    path = tmp_path / "episode.json"
    env.save_episode(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    replay = RocketLandingEnv(saved["config"])
    replay.reset(options={"initial_state": saved["initial_state"]})
    for record in saved["steps"]:
        obs, reward, terminated, truncated, info = replay.step(record["commanded_action"])
        np.testing.assert_array_equal(obs, record["state"])
        assert [reward, terminated, truncated] == [
            record["reward"],
            record["terminated"],
            record["truncated"],
        ]
        assert info["outcome"] == record["outcome"]
    with pytest.raises(FileExistsError):
        env.save_episode(path)


@pytest.mark.parametrize("action", [[1], [0, np.nan], [np.inf, 0], [[0, 0]]])
def test_invalid_actions_do_not_advance_state(action):
    env = env_with()
    env.reset(seed=1)
    before = env.episode_log
    with pytest.raises(ValueError):
        env.step(action)
    assert env.episode_log == before


@pytest.mark.parametrize(
    "initial", [[0, -1, 0, 0, 0, 0, 900], [0, 1, 0, 0, 0, 0, 749], [0, np.nan, 0, 0, 0, 0, 900]]
)
def test_invalid_initial_states(initial):
    with pytest.raises(ValueError):
        env_with().reset(options={"initial_state": initial})


@pytest.mark.parametrize(
    "ranges",
    [
        {"mass_kg": [700, 900]},
        {"z_m": [-1, 100]},
        {"x_m": [5, 1]},
        {"x_m": [0, np.inf]},
        {"unknown": [0, 1]},
        {"x_m": [1]},
    ],
)
def test_invalid_sampling_ranges_rejected(ranges):
    config = deepcopy(CONFIG)
    config["initial_state_sampling"] = ranges
    with pytest.raises(ValueError):
        RocketLandingEnv(config)


def test_step_requires_reset():
    with pytest.raises(RuntimeError, match="reset"):
        env_with().step([0, 0])


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 1.0),
        (2, 1.0),
        (3, -2.0),
        (4, np.deg2rad(5.0)),
        (5, np.deg2rad(5.0)),
    ],
)
def test_inclusive_success_thresholds_at_contact(index, value):
    initial = np.array([0, 0, 0, 0, 0, 0, 900], dtype=float)
    initial[index] = value
    env = env_with()
    env.reset(options={"initial_state": initial})
    assert env.step([0, 0])[4]["outcome"] == "success"


def test_nonzero_ground_level_and_euler_environment():
    env = env_with(ground_z_m=5.0, integrator="euler")
    env.reset(options={"initial_state": [0, 5.001, 0, -0.1, 0, 0, 900]})
    obs, _, terminated, _, info = env.step([0, 0])
    assert terminated and obs[1] == 5
    assert info["time_s"] == pytest.approx(0.01, abs=1e-10)


def test_fuel_event_precedes_later_contact():
    env = env_with(dt_s=0.5)
    env.reset(options={"initial_state": [0, 0.1, 0, -1, 0, 0, 750.01]})
    obs, _, _, _, info = env.step([1, 0])
    assert obs[1] > 0 and info["outcome"] == "fuel_depletion"


def test_contact_at_time_limit_has_priority_over_timeout():
    config = deepcopy(CONFIG)
    config["simulation"].update(dt_s=0.02, max_time_s=0.03)
    env = RocketLandingEnv(config)
    g = config["simulation"]["gravity_m_s2"]
    env.reset(options={"initial_state": [0, 0.5 * g * 0.03**2, 0, 0, 0, 0, 900]})
    env.step([0, 0])
    _, _, terminated, truncated, info = env.step([0, 0])
    assert terminated and not truncated and info["outcome"] == "success"


@pytest.mark.parametrize("seed", range(5))
def test_nominal_random_policy_episode_is_finite_and_logs_termination(seed):
    # Run realistic initial conditions through an entire episode, not just a near-ground fixture.
    env = env_with(max_time_s=0.2)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        obs, _, terminated, truncated, info = env.step(rng.uniform([0, -0.2], [1, 0.2]))
        assert np.all(np.isfinite(obs)) and env.observation_space.contains(obs)
        if terminated or truncated:
            break
    assert info["outcome"] == "timeout"
    assert env.episode_log["outcome"] == "timeout"
