"""Canonical state and control definitions for the planar vehicle model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

STATE_NAMES = ("x", "z", "vx", "vz", "theta", "omega", "mass")
STATE_UNITS = ("m", "m", "m/s", "m/s", "rad", "rad/s", "kg")
ACTION_NAMES = ("throttle", "gimbal_angle")
ACTION_UNITS = ("1", "rad")


def _finite_vector(values: ArrayLike, expected_size: int, label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (expected_size,):
        msg = f"{label} must have shape ({expected_size},), got {vector.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{label} must contain only finite values"
        raise ValueError(msg)
    return vector


@dataclass(frozen=True, slots=True)
class State:
    """State ordered as ``[x, z, vx, vz, theta, omega, mass]`` in SI units."""

    x: float
    z: float
    vx: float
    vz: float
    theta: float
    omega: float
    mass: float

    names: ClassVar[tuple[str, ...]] = STATE_NAMES
    units: ClassVar[tuple[str, ...]] = STATE_UNITS

    def __post_init__(self) -> None:
        vector = self.as_array()
        if not np.all(np.isfinite(vector)):
            raise ValueError("state must contain only finite values")
        if self.mass <= 0:
            raise ValueError("state mass must be positive")

    def as_array(self) -> NDArray[np.float64]:
        """Return the canonical seven-element state vector."""
        return np.asarray(
            (self.x, self.z, self.vx, self.vz, self.theta, self.omega, self.mass),
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: ArrayLike) -> State:
        """Build a state after validating vector shape and finiteness."""
        vector = _finite_vector(values, len(STATE_NAMES), "state")
        return cls(*vector.tolist())


@dataclass(frozen=True, slots=True)
class Control:
    """Control ordered as ``[throttle, gimbal_angle]`` in SI-compatible units."""

    throttle: float
    gimbal_angle: float

    names: ClassVar[tuple[str, ...]] = ACTION_NAMES
    units: ClassVar[tuple[str, ...]] = ACTION_UNITS

    def __post_init__(self) -> None:
        vector = self.as_array()
        if not np.all(np.isfinite(vector)):
            raise ValueError("control must contain only finite values")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be within [0, 1]")

    def as_array(self) -> NDArray[np.float64]:
        """Return the canonical two-element control vector."""
        return np.asarray((self.throttle, self.gimbal_angle), dtype=np.float64)

    @classmethod
    def from_array(cls, values: ArrayLike) -> Control:
        """Build a control after validating vector shape and finiteness."""
        vector = _finite_vector(values, len(ACTION_NAMES), "control")
        return cls(*vector.tolist())
