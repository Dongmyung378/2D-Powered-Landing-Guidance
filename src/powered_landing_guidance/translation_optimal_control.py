"""Day 17 ideal-vector planar translation teacher with CasADi/IPOPT."""

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
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem

STATE_NAMES = ("x", "z", "vx", "vz", "mass")
CONTROL_NAMES = ("throttle", "thrust_angle")


@dataclass(frozen=True, slots=True)
class TranslationGuess:
    """Dynamically consistent seed for the five-state multiple-shooting problem."""

    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    duration_s: float
    source: str
    pid_outcome: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationSolverStage:
    """IPOPT status and separately recomputed normalized constraint residuals."""

    status: str
    iterations: int
    objective: float
    max_hard_violation: float
    max_terminal_violation: float
    max_rk4_defects: dict[str, float]


@dataclass(frozen=True, slots=True)
class TranslationLandingResult:
    """Optimal nodes and ideal-vector simulator replay at a finer time step."""

    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    duration_s: float
    guess_source: str
    pid_outcome: str | None
    stage_a: TranslationSolverStage
    stage_b: TranslationSolverStage
    rollout_times_s: NDArray[np.float64]
    rollout_states: NDArray[np.float64]


class TranslationSolveError(RuntimeError):
    """A failed stage with its status and any available residual diagnostics."""

    def __init__(
        self, stage: str, status: str, diagnostics: TranslationSolverStage | None = None
    ) -> None:
        self.stage = stage
        self.status = status
        self.diagnostics = diagnostics
        super().__init__(f"{stage} IPOPT solve failed: {status}")


def _initial_vector(problem: LandingOptimalControlProblem) -> NDArray[np.float64]:
    initial = problem.initial_state
    return np.asarray((initial.x, initial.z, initial.vx, initial.vz, initial.mass))


def _rhs_symbolic(state, control, problem: LandingOptimalControlProblem):
    p = problem.parameters
    thrust = p.max_thrust_n * control[0]
    return ca.vertcat(
        state[2],
        state[3],
        thrust * ca.sin(control[1]) / state[4],
        thrust * ca.cos(control[1]) / state[4] - p.gravity_m_s2,
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
    p = problem.parameters
    thrust = p.max_thrust_n * control[0]
    horizontal = thrust * np.sin(control[1])
    vertical = thrust * np.cos(control[1])

    def rhs(_time: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            (
                value[2],
                value[3],
                horizontal / value[4],
                vertical / value[4] - p.gravity_m_s2,
                -thrust / (p.specific_impulse_s * p.standard_gravity_m_s2),
            )
        )

    return rk4_step(rhs, 0.0, state, step_s)


def _guess_from_controls(
    problem: LandingOptimalControlProblem,
    controls: NDArray[np.float64],
    duration_s: float,
    source: str,
    pid_outcome: str | None = None,
) -> TranslationGuess:
    states = np.empty((problem.intervals + 1, 5), dtype=np.float64)
    states[0] = _initial_vector(problem)
    step_s = duration_s / problem.intervals
    for k in range(problem.intervals):
        states[k + 1] = _rk4_numeric(states[k], controls[k], step_s, problem)
    return TranslationGuess(states, controls, duration_s, source, pid_outcome)


def analytic_initial_guess(problem: LandingOptimalControlProblem) -> TranslationGuess:
    """Seed vertical acceleration and cubic horizontal position tracking."""
    initial = problem.initial_state
    ground = problem.ground_z_m
    target_vz = problem.target_touchdown_vz_m_s
    descent_speed = -initial.vz - target_vz
    low, high = problem.duration_bounds_s
    estimated_time = 2 * (initial.z - ground) / descent_speed if descent_speed > 0 else 0.0
    duration_s = (
        float(np.clip(estimated_time, low, high)) if estimated_time > 0 else (low + high) / 2
    )
    vertical_accel = (target_vz - initial.vz) / duration_s
    displacement = problem.target_x_m - initial.x
    cubic_a2 = (3 * displacement - 2 * initial.vx * duration_s) / duration_s**2
    cubic_a3 = (-2 * displacement + initial.vx * duration_s) / duration_s**3
    p = problem.parameters
    angle_limit = problem.max_abs_thrust_angle_rad
    controls = np.empty((problem.intervals, 2), dtype=np.float64)
    states = np.empty((problem.intervals + 1, 5), dtype=np.float64)
    states[0] = _initial_vector(problem)
    step_s = duration_s / problem.intervals
    for k in range(problem.intervals):
        midpoint = (k + 0.5) * step_s
        horizontal_accel = 2 * cubic_a2 + 6 * cubic_a3 * midpoint
        angle = np.arctan2(horizontal_accel, p.gravity_m_s2 + vertical_accel)
        angle = float(np.clip(angle, -angle_limit, angle_limit))
        throttle = (
            states[k, 4]
            * np.hypot(horizontal_accel, p.gravity_m_s2 + vertical_accel)
            / p.max_thrust_n
        )
        controls[k] = (np.clip(throttle, p.throttle_min, p.throttle_max), angle)
        states[k + 1] = _rk4_numeric(states[k], controls[k], step_s, problem)
    return TranslationGuess(states, controls, duration_s, "analytic")


def pid_initial_guess(
    problem: LandingOptimalControlProblem, config: dict[str, Any]
) -> TranslationGuess:
    """Sample the existing full-model PID trajectory as an ideal-vector seed."""
    env = RocketLandingEnv(config)
    controller = IntegratedLandingController.from_config(config)
    try:
        state, info = env.reset(options={"initial_state": problem.initial_state.as_array()})
        end_times: list[float] = []
        actions: list[tuple[float, float]] = []
        while True:
            raw = controller.command(state, time_s=float(info["time_s"]))
            applied = clip_control(raw, problem.parameters)
            actions.append((applied.throttle, state[4] + applied.gimbal_angle))
            state, _, terminated, truncated, info = env.step(raw)
            end_times.append(float(info["time_s"]))
            if terminated or truncated:
                break
        outcome = str(info["outcome"])
    finally:
        env.close()
    low, high = problem.duration_bounds_s
    duration_s = float(np.clip(end_times[-1], low, high))
    sample_times = (np.arange(problem.intervals) + 0.5) * duration_s / problem.intervals
    indices = np.searchsorted(end_times, sample_times, side="right")
    sampled = np.asarray(actions, dtype=np.float64)[np.minimum(indices, len(actions) - 1)]
    p = problem.parameters
    sampled[:, 0] = np.clip(sampled[:, 0], p.throttle_min, p.throttle_max)
    angle_limit = problem.max_abs_thrust_angle_rad
    sampled[:, 1] = np.clip(sampled[:, 1], -angle_limit, angle_limit)
    return _guess_from_controls(problem, sampled, duration_s, "pid", outcome)


def _terminal_violations(
    problem: LandingOptimalControlProblem, state: NDArray[np.float64]
) -> NDArray[np.float64]:
    x_limit, vx_limit, vz_limit = problem.terminal_limits[:3]
    return np.asarray(
        (
            max(0.0, abs(state[0] - problem.target_x_m) - x_limit) / x_limit,
            abs(state[1] - problem.ground_z_m),
            max(0.0, abs(state[2]) - vx_limit) / vx_limit,
            max(0.0, -vz_limit - state[3], state[3]) / vz_limit,
        )
    )


def _build_problem(problem: LandingOptimalControlProblem, stage: str, guess: TranslationGuess):
    n = problem.intervals
    p = problem.parameters
    opti = ca.Opti()
    states = opti.variable(5, n + 1)
    controls = opti.variable(2, n)
    duration = opti.variable()
    step_s = duration / n
    low, high = problem.duration_bounds_s
    angle_limit = problem.max_abs_thrust_angle_rad
    opti.subject_to(states[:, 0] == ca.DM(_initial_vector(problem)))
    opti.subject_to(opti.bounded(low, duration, high))
    opti.subject_to(states[1, :] >= problem.ground_z_m)
    opti.subject_to(states[4, :] >= p.dry_mass_kg + problem.min_propellant_reserve_kg)
    opti.subject_to(opti.bounded(p.throttle_min, controls[0, :], p.throttle_max))
    opti.subject_to(opti.bounded(-angle_limit, controls[1, :], angle_limit))
    for k in range(n):
        opti.subject_to(
            states[:, k + 1] == _rk4_symbolic(states[:, k], controls[:, k], step_s, problem)
        )

    x_limit, vx_limit, vz_limit = (float(value) for value in problem.terminal_limits[:3])
    final = states[:, n]
    if stage == "A":
        slack = opti.variable(4)
        opti.subject_to(slack >= 0)
        opti.subject_to(
            opti.bounded(
                problem.target_x_m - x_limit * (1 + slack[0]),
                final[0],
                problem.target_x_m + x_limit * (1 + slack[0]),
            )
        )
        opti.subject_to(final[1] - problem.ground_z_m <= slack[1])
        opti.subject_to(
            opti.bounded(-vx_limit * (1 + slack[2]), final[2], vx_limit * (1 + slack[2]))
        )
        opti.subject_to(opti.bounded(-vz_limit * (1 + slack[3]), final[3], vz_limit * slack[3]))
        opti.minimize(ca.sumsqr(slack))
        opti.set_initial(slack, _terminal_violations(problem, guess.states[-1]))
    else:
        opti.subject_to(
            opti.bounded(problem.target_x_m - x_limit, final[0], problem.target_x_m + x_limit)
        )
        opti.subject_to(final[1] == problem.ground_z_m)
        opti.subject_to(opti.bounded(-vx_limit, final[2], vx_limit))
        opti.subject_to(opti.bounded(-vz_limit, final[3], 0.0))
        fuel = (problem.initial_state.mass - final[4]) / (
            problem.initial_state.mass - p.dry_mass_kg
        )
        touchdown = (
            ((final[0] - problem.target_x_m) / x_limit) ** 2
            + (final[2] / vx_limit) ** 2
            + ((final[3] - problem.target_touchdown_vz_m_s) / vz_limit) ** 2
        ) / 3
        smoothness = (
            (
                ca.sumsqr((controls[0, 1:] - controls[0, :-1]) / (p.throttle_max - p.throttle_min))
                + ca.sumsqr((controls[1, 1:] - controls[1, :-1]) / (2 * angle_limit))
            )
            / (n - 1)
            if n > 1
            else 0
        )
        opti.minimize(
            problem.fuel_weight * fuel
            + problem.touchdown_weight * touchdown
            + problem.smoothness_weight * smoothness
        )

    opti.set_initial(states, guess.states.T)
    opti.set_initial(controls, guess.controls.T)
    opti.set_initial(duration, guess.duration_s)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "tol": 1e-9, "max_iter": 1500})
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
) -> TranslationSolverStage:
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
            problem.initial_state.mass,
        )
    )
    initial_error = np.max(np.abs(states[0] - _initial_vector(problem)) / scales)
    altitude = np.max(np.maximum(problem.ground_z_m - states[:, 1], 0)) / scales[1]
    reserve = np.max(
        np.maximum(p.dry_mass_kg + problem.min_propellant_reserve_kg - states[:, 4], 0)
    ) / (problem.initial_state.mass - p.dry_mass_kg)
    throttle = max(
        0.0,
        float(np.max(p.throttle_min - controls[:, 0])),
        float(np.max(controls[:, 0] - p.throttle_max)),
    ) / (p.throttle_max - p.throttle_min)
    angle_limit = problem.max_abs_thrust_angle_rad
    angle = np.max(np.maximum(np.abs(controls[:, 1]) - angle_limit, 0)) / angle_limit
    low, high = problem.duration_bounds_s
    duration = max(0.0, low - duration_s, duration_s - high) / (high - low)
    hard = max(
        float(initial_error),
        float(np.max(defect_max / scales)),
        float(altitude),
        float(reserve),
        float(throttle),
        float(angle),
        duration,
    )
    terminal = float(np.max(_terminal_violations(problem, states[-1])))
    if stage == "B":
        hard = max(hard, terminal)
    return TranslationSolverStage(
        status=status,
        iterations=iterations,
        objective=objective,
        max_hard_violation=hard,
        max_terminal_violation=terminal,
        max_rk4_defects=dict(zip(STATE_NAMES, defect_max.tolist(), strict=True)),
    )


def _solve_stage(
    problem: LandingOptimalControlProblem, stage: str, guess: TranslationGuess
) -> tuple[TranslationSolverStage, TranslationGuess]:
    opti, states, controls, duration = _build_problem(problem, stage, guess)
    try:
        solution = opti.solve()
    except RuntimeError as error:
        stats = opti.stats()
        status = str(stats.get("return_status", "unknown"))
        diagnostics = None
        try:
            candidate = (
                np.asarray(opti.debug.value(states), dtype=np.float64).T,
                np.asarray(opti.debug.value(controls), dtype=np.float64).T,
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
        raise TranslationSolveError(stage, status, diagnostics) from error
    stats = solution.stats()
    values = TranslationGuess(
        np.asarray(solution.value(states), dtype=np.float64).T,
        np.asarray(solution.value(controls), dtype=np.float64).T,
        float(solution.value(duration)),
        guess.source,
        guess.pid_outcome,
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
    step_s = duration_s / problem.intervals
    initial = problem.initial_state
    translation = np.asarray((initial.x, initial.z, initial.vx, initial.vz, initial.mass))
    times = [0.0]
    states = [translation.copy()]
    for k, (throttle, angle) in enumerate(controls):
        # Day 17 controls absolute thrust direction. Heading can change between
        # intervals; Day 18 will replace that idealization with attitude dynamics.
        state = State(*translation[:4], float(angle), 0.0, float(translation[4]))
        local_times, local_states = simulate_planar(
            state,
            (float(throttle), 0.0),
            problem.parameters,
            duration_s=step_s,
            dt_s=integration_step_s,
        )
        times.extend((k * step_s + local_times[1:]).tolist())
        states.extend(local_states[1:][:, [0, 1, 2, 3, 6]].tolist())
        translation = local_states[-1, [0, 1, 2, 3, 6]]
    return np.asarray(times), np.asarray(states)


def solve_translation_landing(
    problem: LandingOptimalControlProblem,
    *,
    guess_source: str = "analytic",
    pid_config: dict[str, Any] | None = None,
    integration_step_s: float = 0.02,
) -> TranslationLandingResult:
    """Solve the ideal-vector 2D translational subset and replay it at finer steps."""
    initial = problem.initial_state
    if abs(initial.theta) > 1e-12 or abs(initial.omega) > 1e-12:
        raise ValueError("Day 17 translation solver requires zero initial attitude and rate")
    if not np.isfinite(integration_step_s) or integration_step_s <= 0:
        raise ValueError("integration_step_s must be positive and finite")
    if guess_source == "analytic":
        guess = analytic_initial_guess(problem)
    elif guess_source == "pid":
        if pid_config is None:
            raise ValueError("pid_config is required for a PID initial guess")
        guess = pid_initial_guess(problem, pid_config)
    else:
        raise ValueError("guess_source must be 'analytic' or 'pid'")

    stage_a, candidate = _solve_stage(problem, "A", guess)
    if (
        stage_a.max_hard_violation > problem.feasibility_tolerance
        or stage_a.max_terminal_violation > problem.feasibility_tolerance
    ):
        raise TranslationSolveError("A", "feasibility_tolerance_not_met", stage_a)
    stage_b, solution = _solve_stage(problem, "B", candidate)
    if stage_b.max_hard_violation > problem.feasibility_tolerance:
        raise TranslationSolveError("B", "constraint_tolerance_not_met", stage_b)
    rollout_times, rollout_states = _rollout(
        problem, solution.controls, solution.duration_s, integration_step_s
    )
    return TranslationLandingResult(
        times_s=np.linspace(0.0, solution.duration_s, problem.intervals + 1),
        states=solution.states,
        controls=solution.controls,
        duration_s=solution.duration_s,
        guess_source=solution.source,
        pid_outcome=solution.pid_outcome,
        stage_a=stage_a,
        stage_b=stage_b,
        rollout_times_s=rollout_times,
        rollout_states=rollout_states,
    )
