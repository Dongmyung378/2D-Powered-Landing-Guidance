"""Verify Day 3 actuator clipping and rotational sign conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from powered_landing_guidance import Control, State, load_config
from powered_landing_guidance.dynamics import (
    PlanarDynamicsParameters,
    angular_acceleration_rad_s2,
    clip_control,
    simulate_planar,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parameters = PlanarDynamicsParameters.from_config(load_config(root / "configs/default.yaml"))
    raw_command = [1.5, np.deg2rad(30.0)]
    applied = clip_control(raw_command, parameters)
    initial = State(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1000.0)

    positive_command = Control(0.7, np.deg2rad(5.0))
    negative_command = Control(0.7, np.deg2rad(-5.0))
    positive_alpha = angular_acceleration_rad_s2(initial, positive_command, parameters)
    negative_alpha = angular_acceleration_rad_s2(initial, negative_command, parameters)
    _, positive_states = simulate_planar(initial, positive_command, parameters, 0.5, 0.01)
    _, negative_states = simulate_planar(initial, negative_command, parameters, 0.5, 0.01)

    print(
        "Clipped command [throttle, gimbal_deg]: "
        f"[{applied.throttle:.3f}, {np.rad2deg(applied.gimbal_angle):.3f}]"
    )
    print(f"Positive-gimbal angular acceleration: {positive_alpha:.6f} rad/s^2")
    print(f"Negative-gimbal angular acceleration: {negative_alpha:.6f} rad/s^2")
    print(f"Terminal omega for positive gimbal: {positive_states[-1, 5]:.6f} rad/s")
    print(f"Terminal omega for negative gimbal: {negative_states[-1, 5]:.6f} rad/s")

    passed = (
        applied.throttle == parameters.throttle_max
        and applied.gimbal_angle == parameters.gimbal_limit_rad
        and positive_alpha < 0.0 < negative_alpha
        and positive_states[-1, 5] < 0.0 < negative_states[-1, 5]
    )
    print(f"Day 3 verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
