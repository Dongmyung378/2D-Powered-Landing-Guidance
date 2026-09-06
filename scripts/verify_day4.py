"""Verify Day 4 mass depletion, dry-mass floor, and thrust cutoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    applied_thrust_n,
    propellant_mass_flow_rate_kg_s,
    simulate_planar,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parameters = PlanarDynamicsParameters.from_config(load_config(root / "configs/default.yaml"))
    full_thrust = Control(1.0, 0.0)
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    mass_flow_rate = propellant_mass_flow_rate_kg_s(initial, full_thrust, parameters)

    duration_s = 10.0
    _, nominal_states = simulate_planar(initial, full_thrust, parameters, duration_s, 0.02)
    expected_mass = initial.mass - mass_flow_rate * duration_s
    nominal_mass_error = abs(nominal_states[-1, 6] - expected_mass)

    near_dry = State(
        0.0,
        100.0,
        0.0,
        0.0,
        0.0,
        0.0,
        parameters.dry_mass_kg + 1.0,
    )
    burnout_time_s = 1.0 / mass_flow_rate
    _, cutoff_states = simulate_planar(near_dry, full_thrust, parameters, 1.0, 1.0)
    dry_state = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, parameters.dry_mass_kg)
    dry_thrust_n = applied_thrust_n(dry_state, full_thrust, parameters)

    print(f"Full-throttle mass flow: {mass_flow_rate:.6f} kg/s")
    print(f"10 s expected mass: {expected_mass:.6f} kg")
    print(f"10 s simulated mass: {nominal_states[-1, 6]:.6f} kg")
    print(f"Mass analytic error: {nominal_mass_error:.3e} kg")
    print(f"Near-dry burnout time: {burnout_time_s:.6f} s")
    print(f"Post-burnout mass: {cutoff_states[-1, 6]:.6f} kg")
    print(f"Thrust at dry mass: {dry_thrust_n:.6f} N")

    passed = (
        nominal_mass_error < 1e-9
        and np.min(nominal_states[:, 6]) >= parameters.dry_mass_kg
        and cutoff_states[-1, 6] == parameters.dry_mass_kg
        and dry_thrust_n == 0.0
    )
    print(f"Day 4 verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
