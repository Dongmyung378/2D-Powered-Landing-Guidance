"""Fixed-step numerical integrators used by the simulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatVector = NDArray[np.float64]
type DerivativeFunction = Callable[[float, FloatVector], ArrayLike]
type IntegrationMethod = Literal["euler", "rk4"]


def _vector(values: ArrayLike, label: str) -> FloatVector:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    return vector


def _step_size(dt_s: float) -> float:
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be a positive finite number")
    return float(dt_s)


def _evaluate(rhs: DerivativeFunction, time_s: float, state: FloatVector) -> FloatVector:
    derivative = _vector(rhs(time_s, state), "derivative")
    if derivative.shape != state.shape:
        raise ValueError(
            f"derivative shape {derivative.shape} does not match state shape {state.shape}"
        )
    return derivative


def euler_step(
    rhs: DerivativeFunction,
    time_s: float,
    state: ArrayLike,
    dt_s: float,
) -> FloatVector:
    """Advance one explicit Euler step."""
    current = _vector(state, "state")
    step = _step_size(dt_s)
    return current + step * _evaluate(rhs, time_s, current)


def rk4_step(
    rhs: DerivativeFunction,
    time_s: float,
    state: ArrayLike,
    dt_s: float,
) -> FloatVector:
    """Advance one classical fourth-order Runge-Kutta step."""
    current = _vector(state, "state")
    step = _step_size(dt_s)

    k1 = _evaluate(rhs, time_s, current)
    k2 = _evaluate(rhs, time_s + 0.5 * step, current + 0.5 * step * k1)
    k3 = _evaluate(rhs, time_s + 0.5 * step, current + 0.5 * step * k2)
    k4 = _evaluate(rhs, time_s + step, current + step * k3)
    return current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_fixed_step(
    rhs: DerivativeFunction,
    initial_state: ArrayLike,
    time_span_s: tuple[float, float],
    dt_s: float,
    *,
    method: IntegrationMethod = "rk4",
) -> tuple[FloatVector, NDArray[np.float64]]:
    """Integrate over a time span and take a shortened final step when necessary."""
    start_s, end_s = map(float, time_span_s)
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("time_span_s must contain finite values with end > start")
    step = _step_size(dt_s)
    steppers = {"euler": euler_step, "rk4": rk4_step}
    try:
        stepper = steppers[method]
    except KeyError as error:
        raise ValueError(f"unsupported integration method: {method}") from error

    times = [start_s]
    states = [_vector(initial_state, "initial_state").copy()]
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(start_s), abs(end_s)) * 8.0

    while times[-1] < end_s - tolerance:
        current_time = times[-1]
        next_time = min(current_time + step, end_s)
        actual_step = next_time - current_time
        states.append(stepper(rhs, current_time, states[-1], actual_step))
        times.append(next_time)

    times[-1] = end_s
    return np.asarray(times, dtype=np.float64), np.stack(states)
