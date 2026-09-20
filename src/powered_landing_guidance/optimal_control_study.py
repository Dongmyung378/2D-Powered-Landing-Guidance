"""Day 19 objective trade-off and mesh-accuracy study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.optimal_control import (
    LandingOptimalControlProblem,
    ObjectiveScales,
    ObjectiveTerms,
)
from powered_landing_guidance.planar_optimal_control import solve_planar_landing

STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega", "mass")
PROFILE_NAMES = ("fuel_priority", "baseline", "smooth_control")


@dataclass(frozen=True, slots=True)
class ObjectiveProfile:
    """Named weights for the common normalized objective terms."""

    name: str
    fuel: float
    touchdown: float
    smoothness: float


@dataclass(frozen=True, slots=True)
class StudySettings:
    """Reproducible Day 19 sweep settings parsed from the project configuration."""

    mesh_intervals: tuple[int, ...]
    replay_dt_s: float
    final_state_tolerances: tuple[float, ...]
    objective_profiles: tuple[ObjectiveProfile, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> StudySettings:
        validate_config(config)
        settings = config["optimal_control"]["study"]
        tolerances = settings["final_state_tolerances"]
        tolerance_vector = (
            float(tolerances["x_m"]),
            float(tolerances["z_m"]),
            float(tolerances["vx_m_s"]),
            float(tolerances["vz_m_s"]),
            float(np.deg2rad(tolerances["theta_deg"])),
            float(np.deg2rad(tolerances["omega_deg_s"])),
            float(tolerances["mass_kg"]),
        )
        profiles_by_name = settings["objective_profiles"]
        profiles = tuple(
            ObjectiveProfile(
                name=name,
                fuel=float(profiles_by_name[name]["fuel"]),
                touchdown=float(profiles_by_name[name]["touchdown"]),
                smoothness=float(profiles_by_name[name]["smoothness"]),
            )
            for name in PROFILE_NAMES
        )
        return cls(
            mesh_intervals=tuple(int(value) for value in settings["mesh_intervals"]),
            replay_dt_s=float(settings["replay_dt_s"]),
            final_state_tolerances=tolerance_vector,
            objective_profiles=profiles,
        )

    @property
    def tolerance_by_state(self) -> dict[str, float]:
        return dict(zip(STATE_NAMES, self.final_state_tolerances, strict=True))


@dataclass(frozen=True, slots=True)
class StudyCase:
    """Solver cost, timing, controls, and replay disagreement for one setting."""

    profile: str
    intervals: int
    weights: dict[str, float]
    solve_time_s: float
    duration_s: float
    objective: ObjectiveTerms
    fuel_used_kg: float
    max_throttle_rate_per_s: float
    max_gimbal_rate_deg_s: float
    final_state: dict[str, float]
    final_state_error: dict[str, float]
    max_node_state_error: dict[str, float]
    max_final_error_ratio: float
    max_node_error_ratio: float
    max_terminal_violation: float
    stage_a_iterations: int
    stage_b_iterations: int
    replay_passed: bool


@dataclass(frozen=True, slots=True)
class OptimalControlStudyResult:
    """Trade-off and mesh cases sharing one initial condition and tolerance policy."""

    objective_scales: ObjectiveScales
    final_state_tolerances: dict[str, float]
    tradeoff_cases: tuple[StudyCase, ...]
    mesh_cases: tuple[StudyCase, ...]
    all_replays_passed: bool


def _finite_state(initial_state: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(initial_state, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("initial_state must have shape (7,) and contain finite values")
    return values


def _case_key(intervals: int, profile: ObjectiveProfile) -> tuple[int, float, float, float]:
    return intervals, profile.fuel, profile.touchdown, profile.smoothness


def _evaluate_case(
    base_problem: LandingOptimalControlProblem,
    config: dict[str, Any],
    settings: StudySettings,
    intervals: int,
    profile: ObjectiveProfile,
) -> StudyCase:
    problem = replace(
        base_problem,
        intervals=intervals,
        fuel_weight=profile.fuel,
        touchdown_weight=profile.touchdown,
        smoothness_weight=profile.smoothness,
    )
    start = perf_counter()
    result = solve_planar_landing(
        problem,
        config,
        integration_step_s=settings.replay_dt_s,
    )
    solve_time_s = perf_counter() - start
    objective = problem.objective(result.states[-1], result.controls, result.duration_s)

    interpolated_nodes = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(7)
        ]
    )
    final_error = np.abs(result.rollout_states[-1] - result.states[-1])
    node_error = np.max(np.abs(interpolated_nodes - result.states), axis=0)
    tolerances = np.asarray(settings.final_state_tolerances)
    final_error_ratio = final_error / tolerances
    node_error_ratio = node_error / tolerances
    terminal_violation = float(np.max(problem.terminal_violations(result.rollout_states[-1])))

    if intervals > 1:
        control_step_s = result.duration_s / intervals
        control_rates = np.diff(result.controls, axis=0) / control_step_s
        max_throttle_rate = float(np.max(np.abs(control_rates[:, 0])))
        max_gimbal_rate = float(np.rad2deg(np.max(np.abs(control_rates[:, 1]))))
    else:
        max_throttle_rate = 0.0
        max_gimbal_rate = 0.0

    p = problem.parameters
    replay_passed = bool(
        np.max(final_error_ratio) <= 1.0
        and terminal_violation <= problem.feasibility_tolerance
        and np.min(result.rollout_states[:, 1]) >= problem.ground_z_m - tolerances[1]
        and np.min(result.rollout_states[:, 6])
        >= p.dry_mass_kg + problem.min_propellant_reserve_kg - tolerances[6]
    )
    final = result.rollout_states[-1]
    return StudyCase(
        profile=profile.name,
        intervals=intervals,
        weights={
            "fuel": profile.fuel,
            "touchdown": profile.touchdown,
            "smoothness": profile.smoothness,
        },
        solve_time_s=solve_time_s,
        duration_s=result.duration_s,
        objective=objective,
        fuel_used_kg=problem.initial_state.mass - final[6],
        max_throttle_rate_per_s=max_throttle_rate,
        max_gimbal_rate_deg_s=max_gimbal_rate,
        final_state=dict(zip(STATE_NAMES, final.tolist(), strict=True)),
        final_state_error=dict(zip(STATE_NAMES, final_error.tolist(), strict=True)),
        max_node_state_error=dict(zip(STATE_NAMES, node_error.tolist(), strict=True)),
        max_final_error_ratio=float(np.max(final_error_ratio)),
        max_node_error_ratio=float(np.max(node_error_ratio)),
        max_terminal_violation=terminal_violation,
        stage_a_iterations=result.stage_a.iterations,
        stage_b_iterations=result.stage_b.iterations,
        replay_passed=replay_passed,
    )


def run_optimal_control_study(
    config: dict[str, Any],
    initial_state: ArrayLike,
) -> OptimalControlStudyResult:
    """Run the configured objective and mesh sweeps, reusing duplicate cases."""
    state = _finite_state(initial_state)
    settings = StudySettings.from_config(config)
    base_problem = LandingOptimalControlProblem.from_config(config, state)
    profiles = {profile.name: profile for profile in settings.objective_profiles}
    baseline = profiles["baseline"]
    cache: dict[tuple[int, float, float, float], StudyCase] = {}

    def evaluate(intervals: int, profile: ObjectiveProfile) -> StudyCase:
        key = _case_key(intervals, profile)
        if key not in cache:
            cache[key] = _evaluate_case(base_problem, config, settings, intervals, profile)
        return cache[key]

    tradeoff = tuple(
        evaluate(base_problem.intervals, profile) for profile in settings.objective_profiles
    )
    mesh = tuple(evaluate(intervals, baseline) for intervals in settings.mesh_intervals)
    all_cases = (*tradeoff, *mesh)
    return OptimalControlStudyResult(
        objective_scales=base_problem.objective_scales,
        final_state_tolerances=settings.tolerance_by_state,
        tradeoff_cases=tradeoff,
        mesh_cases=mesh,
        all_replays_passed=all(case.replay_passed for case in all_cases),
    )
