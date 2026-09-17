"""Day 16 vertical multiple-shooting teacher with CasADi and IPOPT."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.dynamics import simulate_planar
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.model import State
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem


@dataclass(frozen=True, slots=True)
class SolverStage:
    """IPOPT termination and independently checked constraint residuals."""

    status: str
    iterations: int
    objective: float
    max_hard_violation: float
    max_terminal_violation: float
    max_rk4_defect_z_m: float
    max_rk4_defect_vz_m_s: float
    max_rk4_defect_mass_kg: float


@dataclass(frozen=True, slots=True)
class VerticalLandingResult:
    """Optimal nodes and an independent, finer-step simulator rollout."""

    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    throttle: NDArray[np.float64]
    duration_s: float
    stage_a: SolverStage
    stage_b: SolverStage
    rollout_times_s: NDArray[np.float64]
    rollout_states: NDArray[np.float64]


class VerticalSolveError(RuntimeError):
    """A solver failure with the stage and IPOPT status preserved."""

    def __init__(self, stage: str, status: str, diagnostics: SolverStage | None = None) -> None:
        self.stage = stage
        self.status = status
        self.diagnostics = diagnostics
        super().__init__(f"{stage} IPOPT solve failed: {status}")


def _vertical_rhs(state, throttle, problem: LandingOptimalControlProblem):
    p = problem.parameters
    thrust = p.max_thrust_n * throttle
    return ca.vertcat(
        state[1],
        thrust / state[2] - p.gravity_m_s2,
        -thrust / (p.specific_impulse_s * p.standard_gravity_m_s2),
    )


def _rk4_symbolic(state, throttle, step_s, problem: LandingOptimalControlProblem):
    k1 = _vertical_rhs(state, throttle, problem)
    k2 = _vertical_rhs(state + 0.5 * step_s * k1, throttle, problem)
    k3 = _vertical_rhs(state + 0.5 * step_s * k2, throttle, problem)
    k4 = _vertical_rhs(state + step_s * k3, throttle, problem)
    return state + step_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _rk4_numeric(
    state: NDArray[np.float64],
    throttle: float,
    step_s: float,
    problem: LandingOptimalControlProblem,
) -> NDArray[np.float64]:
    p = problem.parameters
    thrust = p.max_thrust_n * throttle

    def rhs(_time: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            (
                value[1],
                thrust / value[2] - p.gravity_m_s2,
                -thrust / (p.specific_impulse_s * p.standard_gravity_m_s2),
            ),
            dtype=np.float64,
        )

    return rk4_step(rhs, 0.0, state, step_s)


def initial_guess(
    problem: LandingOptimalControlProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Build a bounded-throttle guess from constant-acceleration descent."""
    start = problem.initial_state
    target_vz = problem.target_touchdown_vz_m_s
    descent_speed = -start.vz - target_vz
    low, high = problem.duration_bounds_s
    travel_time = 2 * (start.z - problem.ground_z_m) / descent_speed if descent_speed > 0 else 0
    duration = float(np.clip(travel_time, low, high)) if travel_time > 0 else (low + high) / 2
    acceleration = (target_vz - start.vz) / duration
    p = problem.parameters
    step_s = duration / problem.intervals
    states = np.empty((problem.intervals + 1, 3), dtype=np.float64)
    throttle = np.empty(problem.intervals, dtype=np.float64)
    states[0] = (start.z, start.vz, start.mass)
    for k in range(problem.intervals):
        command = states[k, 2] * (acceleration + p.gravity_m_s2) / p.max_thrust_n
        throttle[k] = np.clip(command, p.throttle_min, p.throttle_max)
        states[k + 1] = _rk4_numeric(states[k], throttle[k], step_s, problem)
    return states, throttle, duration


def _build_problem(
    problem: LandingOptimalControlProblem,
    stage: str,
    guess: tuple[NDArray[np.float64], NDArray[np.float64], float],
):
    n = problem.intervals
    p = problem.parameters
    opti = ca.Opti()
    x = opti.variable(3, n + 1)
    u = opti.variable(1, n)
    duration = opti.variable()
    step_s = duration / n

    opti.subject_to(
        x[:, 0]
        == ca.DM((problem.initial_state.z, problem.initial_state.vz, problem.initial_state.mass))
    )
    low_duration, high_duration = problem.duration_bounds_s
    opti.subject_to(opti.bounded(low_duration, duration, high_duration))
    opti.subject_to(x[0, :] >= problem.ground_z_m)
    opti.subject_to(x[2, :] >= p.dry_mass_kg + problem.min_propellant_reserve_kg)
    opti.subject_to(opti.bounded(p.throttle_min, u, p.throttle_max))
    for k in range(n):
        opti.subject_to(x[:, k + 1] == _rk4_symbolic(x[:, k], u[0, k], step_s, problem))

    vz_limit = float(problem.terminal_limits[2])
    if stage == "A":
        height_slack = opti.variable()
        velocity_slack = opti.variable()
        opti.subject_to(height_slack >= 0)
        opti.subject_to(velocity_slack >= 0)
        opti.subject_to(x[0, n] - problem.ground_z_m <= height_slack)
        opti.subject_to(x[1, n] >= -vz_limit * (1 + velocity_slack))
        opti.subject_to(x[1, n] <= vz_limit * velocity_slack)
        opti.minimize(height_slack**2 + velocity_slack**2)
        opti.set_initial(height_slack, max(0.0, guess[0][-1, 0] - problem.ground_z_m))
        opti.set_initial(
            velocity_slack,
            max(0.0, (-vz_limit - guess[0][-1, 1]) / vz_limit, guess[0][-1, 1] / vz_limit),
        )
    else:
        opti.subject_to(x[0, n] == problem.ground_z_m)
        opti.subject_to(opti.bounded(-vz_limit, x[1, n], 0.0))
        fuel = (problem.initial_state.mass - x[2, n]) / (problem.initial_state.mass - p.dry_mass_kg)
        touchdown = ((x[1, n] - problem.target_touchdown_vz_m_s) / vz_limit) ** 2
        smoothness = (
            ca.sumsqr((u[0, 1:] - u[0, :-1]) / (p.throttle_max - p.throttle_min)) / (n - 1)
            if n > 1
            else 0
        )
        opti.minimize(
            problem.fuel_weight * fuel
            + problem.touchdown_weight * touchdown
            + problem.smoothness_weight * smoothness
        )

    opti.set_initial(x, guess[0].T)
    opti.set_initial(u, guess[1].reshape(1, -1))
    opti.set_initial(duration, guess[2])
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "tol": 1e-9, "max_iter": 1000})
    return opti, x, u, duration


def _diagnostics(
    problem: LandingOptimalControlProblem,
    stage: str,
    status: str,
    iterations: int,
    objective: float,
    states: NDArray[np.float64],
    throttle: NDArray[np.float64],
    duration_s: float,
) -> SolverStage:
    p = problem.parameters
    step_s = duration_s / problem.intervals
    defects = np.asarray(
        [
            states[k + 1] - _rk4_numeric(states[k], throttle[k], step_s, problem)
            for k in range(problem.intervals)
        ]
    )
    defect_max = np.max(np.abs(defects), axis=0)
    z_scale = max(problem.initial_state.z - problem.ground_z_m, 1.0)
    vz_scale = max(abs(problem.initial_state.vz), 1.0)
    mass_scale = problem.initial_state.mass
    initial_error = np.max(
        np.abs(states[0] - (problem.initial_state.z, problem.initial_state.vz, mass_scale))
        / (z_scale, vz_scale, mass_scale)
    )
    altitude_violation = np.max(np.maximum(problem.ground_z_m - states[:, 0], 0)) / z_scale
    reserve_violation = np.max(
        np.maximum(p.dry_mass_kg + problem.min_propellant_reserve_kg - states[:, 2], 0)
    ) / (mass_scale - p.dry_mass_kg)
    throttle_violation = max(
        0.0,
        float(np.max(p.throttle_min - throttle)),
        float(np.max(throttle - p.throttle_max)),
    ) / (p.throttle_max - p.throttle_min)
    duration_violation = max(
        0.0,
        problem.duration_bounds_s[0] - duration_s,
        duration_s - problem.duration_bounds_s[1],
    ) / (problem.duration_bounds_s[1] - problem.duration_bounds_s[0])
    hard = max(
        float(initial_error),
        float(np.max(defect_max / (z_scale, vz_scale, mass_scale))),
        float(altitude_violation),
        float(reserve_violation),
        float(throttle_violation),
        duration_violation,
    )
    vz_limit = float(problem.terminal_limits[2])
    terminal = max(
        abs(states[-1, 0] - problem.ground_z_m),
        max(0.0, -vz_limit - states[-1, 1], states[-1, 1]) / vz_limit,
    )
    if stage == "B":
        hard = max(hard, terminal)
    return SolverStage(
        status=status,
        iterations=iterations,
        objective=objective,
        max_hard_violation=hard,
        max_terminal_violation=terminal,
        max_rk4_defect_z_m=float(defect_max[0]),
        max_rk4_defect_vz_m_s=float(defect_max[1]),
        max_rk4_defect_mass_kg=float(defect_max[2]),
    )


def _solve_stage(
    problem: LandingOptimalControlProblem,
    stage: str,
    guess: tuple[NDArray[np.float64], NDArray[np.float64], float],
) -> tuple[SolverStage, tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    opti, x, u, duration = _build_problem(problem, stage, guess)
    try:
        solution = opti.solve()
    except RuntimeError as error:
        stats = opti.stats()
        status = str(stats.get("return_status", "unknown"))
        diagnostics = None
        try:
            candidate = (
                np.asarray(opti.debug.value(x), dtype=np.float64).T,
                np.asarray(opti.debug.value(u), dtype=np.float64).reshape(-1),
                float(opti.debug.value(duration)),
            )
            if all(np.all(np.isfinite(value)) for value in candidate):
                diagnostics = _diagnostics(
                    problem,
                    stage,
                    status,
                    int(stats.get("iter_count", 0)),
                    float(opti.debug.value(opti.f)),
                    *candidate,
                )
        except (RuntimeError, ValueError, TypeError):
            pass
        raise VerticalSolveError(stage, status, diagnostics) from error
    stats = solution.stats()
    values = (
        np.asarray(solution.value(x), dtype=np.float64).T,
        np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        float(solution.value(duration)),
    )
    diagnostics = _diagnostics(
        problem,
        stage,
        str(stats["return_status"]),
        int(stats["iter_count"]),
        float(solution.value(opti.f)),
        *values,
    )
    return diagnostics, values


def _rollout(
    problem: LandingOptimalControlProblem,
    throttle: NDArray[np.float64],
    duration_s: float,
    integration_step_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    step_s = duration_s / problem.intervals
    initial = problem.initial_state
    state = State(0.0, initial.z, 0.0, initial.vz, 0.0, 0.0, initial.mass)
    times = [0.0]
    states = [state.as_array()[[1, 3, 6]]]
    for k, command in enumerate(throttle):
        local_times, local_states = simulate_planar(
            state,
            (float(command), 0.0),
            problem.parameters,
            duration_s=step_s,
            dt_s=integration_step_s,
        )
        times.extend((k * step_s + local_times[1:]).tolist())
        states.extend(local_states[1:][:, [1, 3, 6]].tolist())
        state = State.from_array(local_states[-1])
    return np.asarray(times, dtype=np.float64), np.asarray(states, dtype=np.float64)


def solve_vertical_landing(
    problem: LandingOptimalControlProblem,
    *,
    integration_step_s: float = 0.02,
) -> VerticalLandingResult:
    """Solve a single nominal vertical descent and independently roll it out."""
    initial = problem.initial_state
    if any(abs(value) > 1e-12 for value in (initial.x, initial.vx, initial.theta, initial.omega)):
        raise ValueError("Day 16 vertical solver requires zero horizontal and attitude state")
    if not np.isfinite(integration_step_s) or integration_step_s <= 0:
        raise ValueError("integration_step_s must be positive and finite")
    stage_a, candidate = _solve_stage(problem, "A", initial_guess(problem))
    if (
        stage_a.max_hard_violation > problem.feasibility_tolerance
        or stage_a.max_terminal_violation > problem.feasibility_tolerance
    ):
        raise VerticalSolveError("A", "feasibility_tolerance_not_met", stage_a)
    stage_b, solution = _solve_stage(problem, "B", candidate)
    if stage_b.max_hard_violation > problem.feasibility_tolerance:
        raise VerticalSolveError("B", "constraint_tolerance_not_met", stage_b)
    states, throttle, duration_s = solution
    rollout_times, rollout_states = _rollout(problem, throttle, duration_s, integration_step_s)
    return VerticalLandingResult(
        times_s=np.linspace(0.0, duration_s, problem.intervals + 1),
        states=states,
        throttle=throttle,
        duration_s=duration_s,
        stage_a=stage_a,
        stage_b=stage_b,
        rollout_times_s=rollout_times,
        rollout_states=rollout_states,
    )
