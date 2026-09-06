"""Compare Day 2 numerical trajectories with constant-acceleration analytic solutions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import PlanarDynamicsParameters, simulate_planar


def _terminal_errors(
    initial: State,
    control: Control,
    parameters: PlanarDynamicsParameters,
    duration_s: float,
) -> tuple[float, float, float]:
    times, states = simulate_planar(
        initial,
        control,
        parameters,
        duration_s=duration_s,
        dt_s=0.02,
        method="rk4",
    )
    thrust_n = control.throttle * parameters.max_thrust_n
    mass_flow_rate = thrust_n / (
        parameters.specific_impulse_s * parameters.standard_gravity_m_s2
    )
    if mass_flow_rate == 0.0:
        expected_mass = initial.mass
        expected_z = (
            initial.z
            + initial.vz * duration_s
            - 0.5 * parameters.gravity_m_s2 * duration_s**2
        )
        expected_vz = initial.vz - parameters.gravity_m_s2 * duration_s
    else:
        exhaust_velocity = parameters.specific_impulse_s * parameters.standard_gravity_m_s2
        expected_mass = initial.mass - mass_flow_rate * duration_s
        log_mass_ratio = np.log(initial.mass / expected_mass)
        integrated_log_ratio = (
            initial.mass - expected_mass - expected_mass * log_mass_ratio
        ) / mass_flow_rate
        expected_z = (
            initial.z
            + initial.vz * duration_s
            + exhaust_velocity * integrated_log_ratio
            - 0.5 * parameters.gravity_m_s2 * duration_s**2
        )
        expected_vz = (
            initial.vz
            + exhaust_velocity * log_mass_ratio
            - parameters.gravity_m_s2 * duration_s
        )
    return (
        abs(states[-1, 1] - expected_z),
        abs(states[-1, 3] - expected_vz),
        abs(states[-1, 6] - expected_mass),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    parameters = PlanarDynamicsParameters.from_config(config)
    initial = State(0.0, 100.0, 0.0, -3.0, 0.0, 0.0, 1000.0)

    free_fall_errors = _terminal_errors(initial, Control(0.0, 0.0), parameters, 2.0)
    ascent_errors = _terminal_errors(initial, Control(1.0, 0.0), parameters, 2.0)
    maximum_error = max(*free_fall_errors, *ascent_errors)

    print(f"Free-fall terminal errors [z, vz, mass]: {free_fall_errors}")
    print(f"Vertical-thrust terminal errors [z, vz, mass]: {ascent_errors}")
    print(f"Maximum analytic error: {maximum_error:.3e}")
    if not np.isfinite(maximum_error) or maximum_error > 1e-9:
        print("Day 2 verification: FAIL")
        return 1
    print("Day 2 verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
