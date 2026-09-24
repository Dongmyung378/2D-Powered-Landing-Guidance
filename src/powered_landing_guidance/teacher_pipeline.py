"""Day 20 multi-initial-condition optimal-control teacher pipeline."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.config import validate_config
from powered_landing_guidance.evaluation import (
    INITIAL_CONDITION_SAMPLER,
    initial_condition_sha256,
    sample_integrated_initial_states,
)
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem
from powered_landing_guidance.optimal_control_study import STATE_NAMES, StudySettings
from powered_landing_guidance.planar_optimal_control import (
    PlanarLandingResult,
    PlanarSolveError,
    control_sequence_initial_guess,
    solve_planar_landing,
)

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TeacherPipelineSettings:
    """Validated execution, retry, and replay settings for the teacher batch."""

    sampler: str
    episodes: int
    seed: int
    minimum_cases: int
    runner: str
    attempt_timeout_s: float
    max_retries: int
    warm_start: str
    retry_initial_guess: str
    worker_restart_after_attempts: int
    replay_dt_s: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TeacherPipelineSettings:
        validate_config(config)
        settings = config["teacher_pipeline"]
        return cls(
            sampler=str(settings["sampler"]),
            episodes=int(settings["episodes"]),
            seed=int(settings["seed"]),
            minimum_cases=int(settings["minimum_cases"]),
            runner=str(settings["runner"]),
            attempt_timeout_s=float(settings["attempt_timeout_s"]),
            max_retries=int(settings["max_retries"]),
            warm_start=str(settings["warm_start"]),
            retry_initial_guess=str(settings["retry_initial_guess"]),
            worker_restart_after_attempts=int(settings["worker_restart_after_attempts"]),
            replay_dt_s=float(settings["replay_dt_s"]),
        )


@dataclass(frozen=True, slots=True)
class TeacherPipelineResult:
    """Batch metadata plus accepted fixed-mesh trajectories."""

    settings: TeacherPipelineSettings
    intervals: int
    batch_sha256: str
    initial_states: NDArray[np.float64]
    cases: tuple[dict[str, Any], ...]
    successful_case_indices: tuple[int, ...]
    solutions: tuple[PlanarLandingResult, ...]
    wall_time_s: float

    @property
    def completion_passed(self) -> bool:
        return len(self.cases) >= self.settings.minimum_cases

    @property
    def successful_cases(self) -> int:
        return len(self.solutions)


@dataclass(frozen=True, slots=True)
class TeacherSolveBatchResult:
    """Reusable solver-batch result for caller-provided initial conditions."""

    intervals: int
    batch_sha256: str
    initial_states: NDArray[np.float64]
    cases: tuple[dict[str, Any], ...]
    successful_case_indices: tuple[int, ...]
    solutions: tuple[PlanarLandingResult, ...]
    wall_time_s: float

    @property
    def successful_cases(self) -> int:
        return len(self.solutions)


def initial_guess_plan(
    previous_success_available: bool,
    max_retries: int,
    retry_initial_guess: str = "pid",
) -> tuple[str, ...]:
    """Return the deterministic warm-start then PID retry sequence."""
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if retry_initial_guess != "pid":
        raise ValueError("retry_initial_guess must be 'pid'")
    first = "previous_success" if previous_success_available else "pid"
    return (first, *(retry_initial_guess for _ in range(max_retries)))


def _silence_worker_output() -> None:
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
    except OSError:
        pass


def _worker_loop(config: dict[str, Any], requests: Any, responses: Any) -> None:
    _silence_worker_output()
    while True:
        request = requests.get()
        if request is None:
            return
        token = request["token"]
        started = perf_counter()
        try:
            problem = LandingOptimalControlProblem.from_config(config, request["initial_state"])
            initial_guess = None
            if request["guess_source"] == "previous_success":
                initial_guess = control_sequence_initial_guess(
                    problem,
                    request["warm_controls"],
                    request["warm_duration_s"],
                    source="previous_success",
                    source_outcome=f"case_{request['warm_case_index']:04d}",
                )
            result = solve_planar_landing(
                problem,
                config,
                integration_step_s=request["replay_dt_s"],
                initial_guess=initial_guess,
            )
            response = {
                "token": token,
                "kind": "solved",
                "elapsed_s": perf_counter() - started,
                "result": result,
            }
        except PlanarSolveError as error:
            response = {
                "token": token,
                "kind": "solver_failure",
                "elapsed_s": perf_counter() - started,
                "stage": error.stage,
                "solver_status": error.status,
                "diagnostics": asdict(error.diagnostics) if error.diagnostics is not None else None,
                "message": str(error),
            }
        except Exception as error:  # noqa: BLE001 - worker must preserve unexpected failures
            response = {
                "token": token,
                "kind": "worker_error",
                "elapsed_s": perf_counter() - started,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        responses.put(response)


class _SequentialSolverWorker:
    def __init__(self, config: dict[str, Any], restart_after_attempts: int) -> None:
        self._context = mp.get_context("spawn")
        self._config = config
        self._restart_after_attempts = restart_after_attempts
        self._token = 0
        self._attempts = 0
        self._process: Any = None
        self._requests: Any = None
        self._responses: Any = None

    def _start(self) -> None:
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_loop,
            args=(self._config, self._requests, self._responses),
            name="teacher-solver",
        )
        self._process.start()
        self._attempts = 0

    def _stop(self, *, force: bool) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive() and not force:
            self._requests.put(None)
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        for channel in (self._requests, self._responses):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._process = None
        self._requests = None
        self._responses = None
        self._attempts = 0

    def _ensure_ready(self) -> None:
        if self._process is None or not self._process.is_alive():
            self._stop(force=True)
            self._start()
        elif self._attempts >= self._restart_after_attempts:
            self._stop(force=False)
            self._start()

    def solve(self, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        self._ensure_ready()
        self._token += 1
        token = self._token
        request = {**request, "token": token}
        self._requests.put(request)
        started = perf_counter()
        deadline = started + timeout_s
        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0.0:
                self._stop(force=True)
                return {
                    "kind": "timeout",
                    "elapsed_s": perf_counter() - started,
                    "timeout_s": timeout_s,
                }
            try:
                response = self._responses.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    exit_code = self._process.exitcode
                    self._stop(force=True)
                    return {
                        "kind": "worker_crash",
                        "elapsed_s": perf_counter() - started,
                        "exit_code": exit_code,
                    }
                continue
            self._attempts += 1
            if response.get("token") != token:
                self._stop(force=True)
                return {
                    "kind": "protocol_error",
                    "elapsed_s": perf_counter() - started,
                    "message": "worker response token did not match request",
                }
            response.pop("token", None)
            return response

    def close(self) -> None:
        self._stop(force=False)


def _validate_replay(
    problem: LandingOptimalControlProblem,
    result: PlanarLandingResult,
    tolerances: NDArray[np.float64],
) -> dict[str, Any]:
    interpolated_nodes = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(7)
        ]
    )
    final_error = np.abs(result.rollout_states[-1] - result.states[-1])
    node_error = np.max(np.abs(interpolated_nodes - result.states), axis=0)
    terminal_violations = problem.terminal_violations(result.rollout_states[-1])
    p = problem.parameters
    checks = {
        "final_state_agreement": bool(np.all(final_error <= tolerances)),
        "terminal_constraints": bool(np.max(terminal_violations) <= problem.feasibility_tolerance),
        "altitude": bool(np.min(result.rollout_states[:, 1]) >= problem.ground_z_m - tolerances[1]),
        "propellant_reserve": bool(
            np.min(result.rollout_states[:, 6])
            >= p.dry_mass_kg + problem.min_propellant_reserve_kg - tolerances[6]
        ),
        "tilt": bool(
            np.max(np.abs(result.rollout_states[:, 4])) <= problem.max_abs_tilt_rad + tolerances[4]
        ),
        "angular_rate": bool(
            np.max(np.abs(result.rollout_states[:, 5]))
            <= problem.max_abs_angular_rate_rad_s + tolerances[5]
        ),
        "gimbal": bool(np.max(np.abs(result.controls[:, 1])) <= p.gimbal_limit_rad + 1e-12),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "final_state_error": dict(zip(STATE_NAMES, final_error.tolist(), strict=True)),
        "max_node_state_error": dict(zip(STATE_NAMES, node_error.tolist(), strict=True)),
        "max_final_error_ratio": float(np.max(final_error / tolerances)),
        "max_node_error_ratio": float(np.max(node_error / tolerances)),
        "max_terminal_violation": float(np.max(terminal_violations)),
        "terminal_violations": dict(
            zip(("x", "z", "vx", "vz", "theta", "omega"), terminal_violations.tolist(), strict=True)
        ),
    }


def _attempt_metadata(
    attempt_index: int,
    guess_source: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "attempt_index": attempt_index,
        "initial_guess": guess_source,
        "status": response["kind"],
        "elapsed_s": float(response["elapsed_s"]),
    }
    for key in (
        "timeout_s",
        "exit_code",
        "stage",
        "solver_status",
        "diagnostics",
        "exception_type",
        "message",
    ):
        if key in response:
            metadata[key] = response[key]
    return metadata


def _success_metadata(
    problem: LandingOptimalControlProblem,
    result: PlanarLandingResult,
    validation: dict[str, Any],
) -> dict[str, Any]:
    final = result.rollout_states[-1]
    objective = problem.objective(result.states[-1], result.controls, result.duration_s)
    return {
        "duration_s": result.duration_s,
        "guess_source": result.guess_source,
        "guess_source_outcome": result.source_outcome,
        "stage_a": {
            "status": result.stage_a.status,
            "iterations": result.stage_a.iterations,
            "max_hard_violation": result.stage_a.max_hard_violation,
            "max_terminal_violation": result.stage_a.max_terminal_violation,
        },
        "stage_b": {
            "status": result.stage_b.status,
            "iterations": result.stage_b.iterations,
            "objective": result.stage_b.objective,
            "max_hard_violation": result.stage_b.max_hard_violation,
        },
        "objective": asdict(objective),
        "fuel_used_kg": problem.initial_state.mass - final[6],
        "final_state": dict(zip(STATE_NAMES, final.tolist(), strict=True)),
        "replay_validation": validation,
    }


def solve_teacher_batch(
    config: dict[str, Any],
    initial_states: NDArray[np.float64],
    *,
    attempt_timeout_s: float,
    max_retries: int,
    retry_initial_guess: str,
    worker_restart_after_attempts: int,
    replay_dt_s: float,
    progress: ProgressCallback | None = None,
) -> TeacherSolveBatchResult:
    """Solve caller-provided states with timeout, warm-start, retry, and replay checks."""
    validate_config(config)
    states = np.array(initial_states, dtype=np.float64, copy=True)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    if not np.all(np.isfinite(states)):
        raise ValueError("initial_states must contain only finite values")
    if not np.isfinite(attempt_timeout_s) or attempt_timeout_s <= 0.0:
        raise ValueError("attempt_timeout_s must be positive and finite")
    if not isinstance(worker_restart_after_attempts, int) or isinstance(
        worker_restart_after_attempts, bool
    ):
        raise ValueError("worker_restart_after_attempts must be a positive integer")
    if worker_restart_after_attempts <= 0:
        raise ValueError("worker_restart_after_attempts must be a positive integer")
    if not np.isfinite(replay_dt_s) or replay_dt_s <= 0.0:
        raise ValueError("replay_dt_s must be positive and finite")
    initial_guess_plan(False, max_retries, retry_initial_guess)
    states.flags.writeable = False
    batch_sha256 = initial_condition_sha256(states)
    study_settings = StudySettings.from_config(config)
    tolerances = np.asarray(study_settings.final_state_tolerances, dtype=np.float64)
    worker = _SequentialSolverWorker(config, worker_restart_after_attempts)
    cases: list[dict[str, Any]] = []
    solutions: list[PlanarLandingResult] = []
    successful_case_indices: list[int] = []
    previous_success: tuple[int, NDArray[np.float64], float] | None = None
    pipeline_started = perf_counter()

    try:
        for case_index, initial_state in enumerate(states):
            attempts: list[dict[str, Any]] = []
            accepted: tuple[PlanarLandingResult, dict[str, Any]] | None = None
            plan = initial_guess_plan(
                previous_success is not None,
                max_retries,
                retry_initial_guess,
            )
            for attempt_index, guess_source in enumerate(plan):
                request: dict[str, Any] = {
                    "initial_state": initial_state,
                    "guess_source": guess_source,
                    "replay_dt_s": replay_dt_s,
                }
                if guess_source == "previous_success" and previous_success is not None:
                    warm_case_index, warm_controls, warm_duration = previous_success
                    request.update(
                        {
                            "warm_case_index": warm_case_index,
                            "warm_controls": warm_controls,
                            "warm_duration_s": warm_duration,
                        }
                    )
                response = worker.solve(request, attempt_timeout_s)
                attempt = _attempt_metadata(attempt_index, guess_source, response)
                if response["kind"] == "solved":
                    result = response["result"]
                    problem = LandingOptimalControlProblem.from_config(config, initial_state)
                    validation = _validate_replay(problem, result, tolerances)
                    attempt["status"] = "accepted" if validation["passed"] else "replay_failed"
                    attempt["replay_validation"] = validation
                    if validation["passed"]:
                        accepted = result, _success_metadata(problem, result, validation)
                attempts.append(attempt)
                if accepted is not None:
                    break

            case: dict[str, Any] = {
                "case_index": case_index,
                "initial_state": dict(zip(STATE_NAMES, initial_state.tolist(), strict=True)),
                "status": "success" if accepted is not None else "failed",
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
            if accepted is not None:
                result, success_metadata = accepted
                case.update(success_metadata)
                solutions.append(result)
                successful_case_indices.append(case_index)
                previous_success = (case_index, result.controls.copy(), result.duration_s)
            else:
                case["failure"] = attempts[-1] if attempts else {"status": "not_attempted"}
            cases.append(case)
            if progress is not None:
                progress(case_index + 1, len(states), case)
    finally:
        worker.close()

    return TeacherSolveBatchResult(
        intervals=int(config["optimal_control"]["intervals"]),
        batch_sha256=batch_sha256,
        initial_states=states,
        cases=tuple(cases),
        successful_case_indices=tuple(successful_case_indices),
        solutions=tuple(solutions),
        wall_time_s=perf_counter() - pipeline_started,
    )


def run_teacher_pipeline(
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> TeacherPipelineResult:
    """Solve the configured Day 20 batch with its frozen execution policy."""
    settings = TeacherPipelineSettings.from_config(config)
    if settings.sampler != INITIAL_CONDITION_SAMPLER:
        raise ValueError(f"unsupported initial-condition sampler: {settings.sampler}")
    initial_states = sample_integrated_initial_states(config, settings.episodes, settings.seed)
    batch = solve_teacher_batch(
        config,
        initial_states,
        attempt_timeout_s=settings.attempt_timeout_s,
        max_retries=settings.max_retries,
        retry_initial_guess=settings.retry_initial_guess,
        worker_restart_after_attempts=settings.worker_restart_after_attempts,
        replay_dt_s=settings.replay_dt_s,
        progress=progress,
    )
    return TeacherPipelineResult(
        settings=settings,
        intervals=batch.intervals,
        batch_sha256=batch.batch_sha256,
        initial_states=batch.initial_states,
        cases=batch.cases,
        successful_case_indices=batch.successful_case_indices,
        solutions=batch.solutions,
        wall_time_s=batch.wall_time_s,
    )


def build_teacher_pipeline_report(result: TeacherPipelineResult) -> dict[str, Any]:
    """Return the complete JSON metadata record, including every failed attempt."""
    failures = [case for case in result.cases if case["status"] != "success"]
    attempts = [attempt for case in result.cases for attempt in case["attempts"]]
    timeout_count = sum(attempt["status"] == "timeout" for attempt in attempts)
    retried_cases = sum(case["attempt_count"] > 1 for case in result.cases)
    warm_start_successes = sum(
        case["status"] == "success" and case.get("guess_source") == "previous_success"
        for case in result.cases
    )
    if not result.completion_passed:
        status = "incomplete"
    elif failures:
        status = "completed_with_failures"
    else:
        status = "complete"
    return {
        "schema_version": 1,
        "problem": "day20-multi-initial-condition-teacher-pipeline",
        "solver": "CasADi/IPOPT",
        "status": status,
        "settings": asdict(result.settings),
        "batch": {
            "sampler": result.settings.sampler,
            "episodes": result.settings.episodes,
            "seed": result.settings.seed,
            "sha256": result.batch_sha256,
        },
        "summary": {
            "executed_cases": len(result.cases),
            "successful_cases": result.successful_cases,
            "failed_cases": len(failures),
            "success_rate": result.successful_cases / len(result.cases) if result.cases else 0.0,
            "total_attempts": len(attempts),
            "retried_cases": retried_cases,
            "timeout_attempts": timeout_count,
            "warm_start_successes": warm_start_successes,
            "wall_time_s": result.wall_time_s,
        },
        "completion_gate": {
            "criterion": f"execute at least {result.settings.minimum_cases} initial conditions",
            "passed": result.completion_passed,
        },
        "cases": list(result.cases),
        "failures": failures,
    }


def teacher_trajectory_arrays(result: TeacherPipelineResult) -> dict[str, NDArray[Any]]:
    """Pack accepted fixed-mesh trajectories into non-object NumPy arrays."""
    intervals = result.intervals
    if result.solutions:
        states = np.stack([solution.states for solution in result.solutions])
        controls = np.stack([solution.controls for solution in result.solutions])
        times = np.stack([solution.times_s for solution in result.solutions])
        durations = np.asarray([solution.duration_s for solution in result.solutions])
    else:
        states = np.empty((0, intervals + 1, 7), dtype=np.float64)
        controls = np.empty((0, intervals, 2), dtype=np.float64)
        times = np.empty((0, intervals + 1), dtype=np.float64)
        durations = np.empty((0,), dtype=np.float64)
    return {
        "case_indices": np.asarray(result.successful_case_indices, dtype=np.int64),
        "initial_states": result.initial_states[
            np.asarray(result.successful_case_indices, dtype=np.int64)
        ],
        "times_s": times,
        "states_x_z_vx_vz_theta_omega_mass": states,
        "controls_throttle_gimbal_rad": controls,
        "durations_s": durations,
    }
