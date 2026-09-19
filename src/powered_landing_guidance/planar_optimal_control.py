"""Day 18 full planar 3-DoF landing teacher with CasADi/IPOPT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.controllers import IntegratedLandingController
from powered_landing_guidance.dynamics import clip_control, simulate_planar
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.model import State
from powered_landing_guidance.optimal_control import (
    TERMINAL_RESIDUAL_NAMES,
    LandingOptimalControlProblem,
)

STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega", "mass")


@dataclass(frozen=True, slots=True)
class PlanarGuess:
    """Full-state initial guess and its source episode metadata."""

    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    duration_s: float
    source: str
    source_outcome: str | None = None


@dataclass(frozen=True, slots=True)
class PlanarSolverStage:
    """Solver status plus independently recomputed constraint diagnostics."""

    status: str
    iterations: int
    objective: float
    max_hard_violation: float
    max_terminal_violation: float
    hard_violations: dict[str, float]
    terminal_violations: dict[str, float]
    max_rk4_defects: dict[str, float]
    worst_constraint: str
    worst_violation: float


@dataclass(frozen=True, slots=True)
class PlanarLandingResult:
    """Full-state optimal nodes and an independent fine-step simulator replay."""

    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    duration_s: float
    guess_source: str
    source_outcome: str | None
    stage_a: PlanarSolverStage
    stage_b: PlanarSolverStage
    rollout_times_s: NDArray[np.float64]
    rollout_states: NDArray[np.float64]


class PlanarSolveError(RuntimeError):
    """A failed solve stage with the best available residual breakdown."""

    def __init__(
        self,
        stage: str,
        status: str,
        diagnostics: PlanarSolverStage | None = None,
    ) -> None:
        self.stage = stage
        self.status = status
        self.diagnostics = diagnostics
        detail = ""
        if diagnostics is not None:
            detail = f"; worst {diagnostics.worst_constraint}={diagnostics.worst_violation:.3e}"
        super().__init__(f"{stage} IPOPT solve failed: {status}{detail}")


def _initial_vector(problem: LandingOptimalControlProblem) -> NDArray[np.float64]:
    return problem.initial_state.as_array()


def _rhs_symbolic(state, control, problem: LandingOptimalControlProblem):
    p = problem.parameters
    thrust = p.max_thrust_n * control[0]
    direction = state[4] + control[1]
    return ca.vertcat(
        state[2],
        state[3],
        thrust * ca.sin(direction) / state[6],
        thrust * ca.cos(direction) / state[6] - p.gravity_m_s2,
        state[5],
        -p.engine_lever_arm_m * thrust * ca.sin(control[1]) / p.moment_of_inertia_kg_m2,
        -thrust / (p.specific_impulse_s * p.standard_gravity_m_s2),
    )


def _rk4_symbolic(state, control, step_s, problem: LandingOptimalControlProblem):
    k1 = _rhs_symbolic(state, control, problem)
    k2 = _rhs_symbolic(state + step_s * k1 / 2, control, problem)
    k3 = _rhs_symbolic(state + step_s * k2 / 2, control, problem)
    k4 = _rhs_symbolic(state + step_s * k3, control, problem)
    return state + step_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _rk4_numeric(
    state: NDArray[np.float64],
    control: NDArray[np.float64],
    step_s: float,
    problem: LandingOptimalControlProblem,
) -> NDArray[np.float64]:
    return rk4_step(
        lambda _time, value: problem.dynamics(value, control),
        0.0,
        state,
        step_s,
    )


def _guess_from_controls(
    problem: LandingOptimalControlProblem,
    controls: NDArray[np.float64],
    duration_s: float,
    source: str,
    source_outcome: str | None = None,
) -> PlanarGuess:
    states = np.empty((problem.intervals + 1, 7), dtype=np.float64)
    states[0] = _initial_vector(problem)
    step_s = duration_s / problem.intervals
    for k in range(problem.intervals):
        states[k + 1] = _rk4_numeric(states[k], controls[k], step_s, problem)
    return PlanarGuess(states, controls, duration_s, source, source_outcome)


def pid_initial_guess(
    problem: LandingOptimalControlProblem,
    config: dict[str, Any],
) -> PlanarGuess:
    """Sample the existing integrated PID episode, then reintegrate on the NLP mesh."""
    env = RocketLandingEnv(config)
    controller = IntegratedLandingController.from_config(config)
    try:
        state, info = env.reset(options={"initial_state": problem.initial_state.as_array()})
        end_times: list[float] = []
        actions: list[NDArray[np.float64]] = []
        while True:
            raw = controller.command(state, time_s=float(info["time_s"]))
            actions.append(clip_control(raw, problem.parameters).as_array())
            state, _, terminated, truncated, info = env.step(raw)
            end_times.append(float(info["time_s"]))
            if terminated or truncated:
                break
        outcome = str(info["outcome"])
    finally:
        env.close()

    low, high = problem.duration_bounds_s
    duration_s = float(np.clip(end_times[-1], low, high))
    midpoints = (np.arange(problem.intervals) + 0.5) * duration_s / problem.intervals
    indices = np.searchsorted(end_times, midpoints, side="right")
    sampled = np.asarray(actions, dtype=np.float64)[np.minimum(indices, len(actions) - 1)]
    p = problem.parameters
    sampled[:, 0] = np.clip(sampled[:, 0], p.throttle_min, p.throttle_max)
    sampled[:, 1] = np.clip(sampled[:, 1], -p.gimbal_limit_rad, p.gimbal_limit_rad)
    return _guess_from_controls(problem, sampled, duration_s, "pid", outcome)


def _add_terminal_constraints(opti, final, problem: LandingOptimalControlProblem) -> None:
    x_limit, vx_limit, vz_limit, theta_limit, omega_limit = (
        float(value) for value in problem.terminal_limits
    )
    opti.subject_to(
        opti.bounded(
            problem.target_x_m - x_limit,
            final[0],
            problem.target_x_m + x_limit,
        )
    )
    opti.subject_to(final[1] == problem.ground_z_m)
    opti.subject_to(opti.bounded(-vx_limit, final[2], vx_limit))
    opti.subject_to(opti.bounded(-vz_limit, final[3], 0.0))
    opti.subject_to(opti.bounded(-theta_limit, final[4], theta_limit))
    opti.subject_to(opti.bounded(-omega_limit, final[5], omega_limit))


def _add_relaxed_terminal_constraints(
    opti,
    final,
    slack,
    problem: LandingOptimalControlProblem,
) -> None:
    x_limit, vx_limit, vz_limit, theta_limit, omega_limit = (
        float(value) for value in problem.terminal_limits
    )
    opti.subject_to(slack >= 0)
    opti.subject_to(
        opti.bounded(
            problem.target_x_m - x_limit * (1 + slack[0]),
            final[0],
            problem.target_x_m + x_limit * (1 + slack[0]),
        )
    )
    opti.subject_to(final[1] - problem.ground_z_m <= slack[1])
    opti.subject_to(opti.bounded(-vx_limit * (1 + slack[2]), final[2], vx_limit * (1 + slack[2])))
    opti.subject_to(opti.bounded(-vz_limit * (1 + slack[3]), final[3], vz_limit * slack[3]))
    opti.subject_to(
        opti.bounded(
            -theta_limit * (1 + slack[4]),
            final[4],
            theta_limit * (1 + slack[4]),
        )
    )
    opti.subject_to(
        opti.bounded(
            -omega_limit * (1 + slack[5]),
            final[5],
            omega_limit * (1 + slack[5]),
        )
    )


def _stage_b_objective(
    problem: LandingOptimalControlProblem,
    final,
    controls,
):
    p = problem.parameters
    x_limit, vx_limit, vz_limit, theta_limit, omega_limit = (
        float(value) for value in problem.terminal_limits
    )
    fuel = (problem.initial_state.mass - final[6]) / (problem.initial_state.mass - p.dry_mass_kg)
    touchdown = (
        ((final[0] - problem.target_x_m) / x_limit) ** 2
        + (final[2] / vx_limit) ** 2
        + ((final[3] - problem.target_touchdown_vz_m_s) / vz_limit) ** 2
        + (final[4] / theta_limit) ** 2
        + (final[5] / omega_limit) ** 2
    ) / 5
    if problem.intervals > 1:
        smoothness = (
            ca.sumsqr((controls[0, 1:] - controls[0, :-1]) / (p.throttle_max - p.throttle_min))
            + ca.sumsqr((controls[1, 1:] - controls[1, :-1]) / (2 * p.gimbal_limit_rad))
        ) / (problem.intervals - 1)
    else:
        smoothness = 0
    return (
        problem.fuel_weight * fuel
        + problem.touchdown_weight * touchdown
        + problem.smoothness_weight * smoothness
    )


def _build_problem(
    problem: LandingOptimalControlProblem,
    stage: str,
    guess: PlanarGuess,
):
    n = problem.intervals
    p = problem.parameters
    opti = ca.Opti()
    states = opti.variable(7, n + 1)
    controls = opti.variable(2, n)
    duration = opti.variable()
    step_s = duration / n
    low, high = problem.duration_bounds_s

    opti.subject_to(states[:, 0] == ca.DM(_initial_vector(problem)))
    opti.subject_to(opti.bounded(low, duration, high))
    opti.subject_to(states[1, :] >= problem.ground_z_m)
    opti.subject_to(states[6, :] >= p.dry_mass_kg + problem.min_propellant_reserve_kg)
    opti.subject_to(opti.bounded(-problem.max_abs_tilt_rad, states[4, :], problem.max_abs_tilt_rad))
    opti.subject_to(
        opti.bounded(
            -problem.max_abs_angular_rate_rad_s,
            states[5, :],
            problem.max_abs_angular_rate_rad_s,
        )
    )
    opti.subject_to(opti.bounded(p.throttle_min, controls[0, :], p.throttle_max))
    opti.subject_to(opti.bounded(-p.gimbal_limit_rad, controls[1, :], p.gimbal_limit_rad))
    for k in range(n):
        opti.subject_to(
            states[:, k + 1] == _rk4_symbolic(states[:, k], controls[:, k], step_s, problem)
        )

    final = states[:, n]
    if stage == "A":
        slack = opti.variable(6)
        _add_relaxed_terminal_constraints(opti, final, slack, problem)
        opti.minimize(ca.sumsqr(slack))
        opti.set_initial(slack, problem.terminal_violations(guess.states[-1]))
    else:
        _add_terminal_constraints(opti, final, problem)
        opti.minimize(_stage_b_objective(problem, final, controls))

    opti.set_initial(states, guess.states.T)
    opti.set_initial(controls, guess.controls.T)
    opti.set_initial(duration, guess.duration_s)
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"print_level": 0, "tol": 1e-9, "max_iter": 2000},
    )
    return opti, states, controls, duration


def _diagnostics(
    problem: LandingOptimalControlProblem,
    stage: str,
    status: str,
    iterations: int,
    objective: float,
    states: NDArray[np.float64],
    controls: NDArray[np.float64],
    duration_s: float,
) -> PlanarSolverStage:
    p = problem.parameters
    step_s = duration_s / problem.intervals
    defects = np.asarray(
        [
            states[k + 1] - _rk4_numeric(states[k], controls[k], step_s, problem)
            for k in range(problem.intervals)
        ]
    )
    defect_max = np.max(np.abs(defects), axis=0)
    scales = np.asarray(
        (
            max(abs(problem.initial_state.x - problem.target_x_m), 1.0),
            max(problem.initial_state.z - problem.ground_z_m, 1.0),
            max(abs(problem.initial_state.vx), 1.0),
            max(abs(problem.initial_state.vz), 1.0),
            problem.max_abs_tilt_rad,
            problem.max_abs_angular_rate_rad_s,
            problem.initial_state.mass,
        )
    )
    available_fuel = problem.initial_state.mass - p.dry_mass_kg
    low, high = problem.duration_bounds_s
    hard = {
        "initial_state": float(np.max(np.abs(states[0] - _initial_vector(problem)) / scales)),
        "rk4_continuity": float(np.max(defect_max / scales)),
        "altitude": float(np.max(np.maximum(problem.ground_z_m - states[:, 1], 0)) / scales[1]),
        "propellant_reserve": float(
            np.max(
                np.maximum(
                    p.dry_mass_kg + problem.min_propellant_reserve_kg - states[:, 6],
                    0,
                )
            )
            / available_fuel
        ),
        "throttle": max(
            0.0,
            float(np.max(p.throttle_min - controls[:, 0])),
            float(np.max(controls[:, 0] - p.throttle_max)),
        )
        / (p.throttle_max - p.throttle_min),
        "gimbal": float(
            np.max(np.maximum(np.abs(controls[:, 1]) - p.gimbal_limit_rad, 0)) / p.gimbal_limit_rad
        ),
        "tilt": float(
            np.max(np.maximum(np.abs(states[:, 4]) - problem.max_abs_tilt_rad, 0))
            / problem.max_abs_tilt_rad
        ),
        "angular_rate": float(
            np.max(
                np.maximum(
                    np.abs(states[:, 5]) - problem.max_abs_angular_rate_rad_s,
                    0,
                )
            )
            / problem.max_abs_angular_rate_rad_s
        ),
        "duration": max(0.0, low - duration_s, duration_s - high) / (high - low),
    }
    terminal_values = problem.terminal_violations(states[-1])
    terminal = dict(zip(TERMINAL_RESIDUAL_NAMES, terminal_values.tolist(), strict=True))
    max_terminal = float(np.max(terminal_values))
    max_hard = max(hard.values())
    if stage == "B":
        max_hard = max(max_hard, max_terminal)
    combined = {**hard, **{f"terminal_{name}": value for name, value in terminal.items()}}
    worst_constraint, worst_violation = max(combined.items(), key=lambda item: item[1])
    return PlanarSolverStage(
        status=status,
        iterations=iterations,
        objective=objective,
        max_hard_violation=max_hard,
        max_terminal_violation=max_terminal,
        hard_violations=hard,
        terminal_violations=terminal,
        max_rk4_defects=dict(zip(STATE_NAMES, defect_max.tolist(), strict=True)),
        worst_constraint=worst_constraint,
        worst_violation=worst_violation,
    )


def _solve_stage(
    problem: LandingOptimalControlProblem,
    stage: str,
    guess: PlanarGuess,
) -> tuple[PlanarSolverStage, PlanarGuess]:
    opti, states, controls, duration = _build_problem(problem, stage, guess)
    try:
        solution = opti.solve()
    except RuntimeError as error:
        stats = opti.stats()
        status = str(stats.get("return_status", "unknown"))
        diagnostics = None
        try:
            candidate_states = np.asarray(opti.debug.value(states), dtype=np.float64).T
            candidate_controls = np.asarray(opti.debug.value(controls), dtype=np.float64).T
            candidate_duration = float(opti.debug.value(duration))
            candidate_objective = float(opti.debug.value(opti.f))
            if (
                np.all(np.isfinite(candidate_states))
                and np.all(np.isfinite(candidate_controls))
                and np.isfinite(candidate_duration)
                and np.isfinite(candidate_objective)
            ):
                diagnostics = _diagnostics(
                    problem,
                    stage,
                    status,
                    int(stats.get("iter_count", 0)),
                    candidate_objective,
                    candidate_states,
                    candidate_controls,
                    candidate_duration,
                )
        except (RuntimeError, ValueError, TypeError):
            pass
        raise PlanarSolveError(stage, status, diagnostics) from error

    stats = solution.stats()
    values = PlanarGuess(
        states=np.asarray(solution.value(states), dtype=np.float64).T,
        controls=np.asarray(solution.value(controls), dtype=np.float64).T,
        duration_s=float(solution.value(duration)),
        source=guess.source,
        source_outcome=guess.source_outcome,
    )
    diagnostics = _diagnostics(
        problem,
        stage,
        str(stats["return_status"]),
        int(stats["iter_count"]),
        float(solution.value(opti.f)),
        values.states,
        values.controls,
        values.duration_s,
    )
    return diagnostics, values


def _rollout(
    problem: LandingOptimalControlProblem,
    controls: NDArray[np.float64],
    duration_s: float,
    integration_step_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    interval_s = duration_s / problem.intervals
    current = problem.initial_state.as_array()
    times = [0.0]
    states = [current.copy()]
    for k, action in enumerate(controls):
        local_times, local_states = simulate_planar(
            State.from_array(current),
            action,
            problem.parameters,
            duration_s=interval_s,
            dt_s=integration_step_s,
        )
        times.extend((k * interval_s + local_times[1:]).tolist())
        states.extend(local_states[1:].tolist())
        current = local_states[-1]
    return np.asarray(times), np.asarray(states)


def solve_planar_landing(
    problem: LandingOptimalControlProblem,
    config: dict[str, Any],
    *,
    integration_step_s: float = 0.02,
) -> PlanarLandingResult:
    """Solve the full seven-state planar landing problem and replay its controls."""
    if not np.isfinite(integration_step_s) or integration_step_s <= 0:
        raise ValueError("integration_step_s must be positive and finite")
    if abs(problem.initial_state.theta) > problem.max_abs_tilt_rad:
        raise ValueError("initial attitude exceeds the optimal-control tilt limit")
    if abs(problem.initial_state.omega) > problem.max_abs_angular_rate_rad_s:
        raise ValueError("initial angular rate exceeds the optimal-control limit")

    guess = pid_initial_guess(problem, config)
    stage_a, candidate = _solve_stage(problem, "A", guess)
    if (
        stage_a.max_hard_violation > problem.feasibility_tolerance
        or stage_a.max_terminal_violation > problem.feasibility_tolerance
    ):
        raise PlanarSolveError("A", "feasibility_tolerance_not_met", stage_a)
    stage_b, solution = _solve_stage(problem, "B", candidate)
    if stage_b.max_hard_violation > problem.feasibility_tolerance:
        raise PlanarSolveError("B", "constraint_tolerance_not_met", stage_b)
    rollout_times, rollout_states = _rollout(
        problem,
        solution.controls,
        solution.duration_s,
        integration_step_s,
    )
    return PlanarLandingResult(
        times_s=np.linspace(0.0, solution.duration_s, problem.intervals + 1),
        states=solution.states,
        controls=solution.controls,
        duration_s=solution.duration_s,
        guess_source=solution.source,
        source_outcome=solution.source_outcome,
        stage_a=stage_a,
        stage_b=stage_b,
        rollout_times_s=rollout_times,
        rollout_states=rollout_states,
    )
