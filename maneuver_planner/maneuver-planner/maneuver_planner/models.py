"""Small data contracts shared by guidance, simulation, and evaluation.

Epochs are seconds relative to one caller-chosen origin (not an implied UTC date).
Frame names identify fixed, body-centered inertial axes. No frame conversion is
performed. NumPy arrays are copied and marked read-only at object construction.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


def positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _array(values, shape: tuple[int, ...], name: str) -> Vector:
    result = np.array(values, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}.")
    result.setflags(write=False)
    return result


def _frame(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("An explicit inertial frame name is required.")


@dataclass(frozen=True)
class CentralBody:
    name: str
    mu_m3_s2: float
    radius_m: float = 0.0

    def __post_init__(self):
        positive(self.mu_m3_s2, "mu_m3_s2")
        if not isfinite(self.radius_m) or self.radius_m < 0:
            raise ValueError("radius_m must be finite and nonnegative.")
        if not self.name:
            raise ValueError("The central body needs a name.")


@dataclass(frozen=True)
class CartesianState:
    epoch_s: float
    position_m: Vector
    velocity_m_s: Vector
    frame: str

    def __post_init__(self):
        if not isfinite(self.epoch_s):
            raise ValueError("epoch_s must be finite.")
        _frame(self.frame)
        object.__setattr__(self, "position_m", _array(self.position_m, (3,), "position_m"))
        object.__setattr__(self, "velocity_m_s", _array(self.velocity_m_s, (3,), "velocity_m_s"))
        if np.linalg.norm(self.position_m) == 0:
            raise ValueError("Position cannot be the central-body origin.")

    @property
    def vector(self) -> Vector:
        """Return a new six-component array [x, y, z, vx, vy, vz]."""
        return np.concatenate((self.position_m, self.velocity_m_s))


@dataclass(frozen=True)
class TransferRequest:
    """Request a circular target orbit in the initial orbit's plane and direction."""

    initial_state: CartesianState
    body: CentralBody
    target_radius_m: float

    def __post_init__(self):
        positive(self.target_radius_m, "target_radius_m")
        if self.target_radius_m <= self.body.radius_m:
            raise ValueError("The target radius must be outside the central body.")


@dataclass(frozen=True)
class Impulse:
    epoch_s: float
    delta_v_m_s: Vector
    frame: str

    def __post_init__(self):
        if not isfinite(self.epoch_s):
            raise ValueError("Burn epoch must be finite.")
        _frame(self.frame)
        object.__setattr__(self, "delta_v_m_s", _array(self.delta_v_m_s, (3,), "delta_v_m_s"))

    @property
    def magnitude_m_s(self) -> float:
        return float(np.linalg.norm(self.delta_v_m_s))


@dataclass(frozen=True)
class ManeuverCandidate:
    """A proposed sequence of inertial impulses, not an executed trajectory."""

    tool_name: str
    initial_state: CartesianState
    body: CentralBody
    burns: tuple[Impulse, ...]
    end_epoch_s: float
    description: str = ""

    def __post_init__(self):
        if not isfinite(self.end_epoch_s) or self.end_epoch_s < self.initial_state.epoch_s:
            raise ValueError("Candidate end must be at or after its initial epoch.")
        object.__setattr__(self, "burns", tuple(self.burns))
        previous = self.initial_state.epoch_s
        for burn in self.burns:
            if burn.frame != self.initial_state.frame:
                raise ValueError("Every burn must use the initial state's inertial frame.")
            if not previous <= burn.epoch_s <= self.end_epoch_s:
                raise ValueError("Burns must be ordered and inside the candidate time interval.")
            previous = burn.epoch_s

    @property
    def duration_s(self) -> float:
        return self.end_epoch_s - self.initial_state.epoch_s

    @property
    def total_delta_v_m_s(self) -> float:
        return sum(burn.magnitude_m_s for burn in self.burns)

    def to_dict(self) -> dict:
        """Export plain JSON-compatible commands for another simulator or language."""
        return {
            "schema_version": 1,
            "tool_name": self.tool_name,
            "units": {"length": "m", "time": "s", "velocity": "m/s"},
            "epoch_reference": "seconds relative to the caller's common origin",
            "frame": self.initial_state.frame,
            "central_body": {"name": self.body.name, "mu_m3_s2": self.body.mu_m3_s2,
                             "radius_m": self.body.radius_m},
            "initial_state": {"epoch_s": self.initial_state.epoch_s,
                              "position_m": self.initial_state.position_m.tolist(),
                              "velocity_m_s": self.initial_state.velocity_m_s.tolist()},
            "burns": [{"epoch_s": burn.epoch_s, "delta_v_m_s": burn.delta_v_m_s.tolist(),
                       "frame": burn.frame} for burn in self.burns],
            "end_epoch_s": self.end_epoch_s,
            "predicted_duration_s": self.duration_s,
            "predicted_total_delta_v_m_s": self.total_delta_v_m_s,
            "description": self.description,
        }


@dataclass(frozen=True)
class Trajectory:
    """Sampled states; duplicate epochs represent the two sides of a burn."""

    epochs_s: Vector
    states: Vector
    frame: str

    def __post_init__(self):
        times = np.asarray(self.epochs_s)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("Trajectory epochs must be a nonempty one-dimensional array.")
        _frame(self.frame)
        object.__setattr__(self, "epochs_s", _array(times, (len(times),), "epochs_s"))
        object.__setattr__(self, "states", _array(self.states, (len(times), 6), "states"))
        if np.any(np.diff(self.epochs_s) < 0):
            raise ValueError("Trajectory epochs must be nondecreasing.")

    @property
    def positions_m(self) -> Vector:
        return self.states[:, :3]

    @property
    def velocities_m_s(self) -> Vector:
        return self.states[:, 3:]

    def state(self, index: int) -> CartesianState:
        return CartesianState(float(self.epochs_s[index]), self.positions_m[index],
                              self.velocities_m_s[index], self.frame)


@dataclass(frozen=True)
class BurnRecord:
    command: Impulse
    before: CartesianState
    after: CartesianState


@dataclass(frozen=True)
class ExecutionResult:
    candidate: ManeuverCandidate
    trajectory: Trajectory
    coast_segments: tuple[Trajectory, ...]
    burn_records: tuple[BurnRecord, ...]
    terminal_state: CartesianState
    # This optional extra coast is for verification, not maneuver duration or cost.
    verification_trajectory: Trajectory | None = None
