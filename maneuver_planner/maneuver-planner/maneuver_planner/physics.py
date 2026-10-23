"""Two-body propagation, ideal impulse execution, and a replaceable coast executor."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from scipy.integrate import solve_ivp

from .models import (BurnRecord, CartesianState, CentralBody, ExecutionResult,
                     Impulse, ManeuverCandidate, Trajectory, positive)


@dataclass(frozen=True)
class IntegratorSettings:
    rtol: float = 1e-12
    position_atol_m: float = 1e-6
    velocity_atol_m_s: float = 1e-9
    max_step_s: float = float("inf")

    def __post_init__(self):
        for name in ("rtol", "position_atol_m", "velocity_atol_m_s"):
            positive(getattr(self, name), name)
        if self.max_step_s <= 0 or np.isnan(self.max_step_s):
            raise ValueError("max_step_s must be positive, or infinity.")


def two_body_derivative(epoch_s: float, state_vector, mu_m3_s2: float):
    """Return [velocity, acceleration]; the autonomous dynamics do not use epoch_s."""
    radius = np.linalg.norm(state_vector[:3])
    if radius == 0:
        raise ValueError("Two-body gravity is singular at the origin.")
    return np.concatenate((state_vector[3:], -mu_m3_s2 * state_vector[:3] / radius**3))


def propagate_two_body(initial: CartesianState, end_epoch_s: float, body: CentralBody,
                       *, sample_step_s: float = 30.0,
                       settings: IntegratorSettings | None = None) -> Trajectory:
    """Coast to an absolute epoch using DOP853, with no burns or guidance inside.

    sample_step_s controls saved output spacing; integration steps are adaptive.
    The requested endpoints are always included exactly. Only forward propagation
    is supported. This point-mass model does not implement collision constraints.
    """
    if not isfinite(end_epoch_s) or end_epoch_s < initial.epoch_s:
        raise ValueError("End epoch must be finite and at or after the initial epoch.")
    positive(sample_step_s, "sample_step_s")
    duration = end_epoch_s - initial.epoch_s
    if duration == 0:
        return Trajectory([initial.epoch_s], [initial.vector], initial.frame)
    settings = settings or IntegratorSettings()
    times = np.linspace(initial.epoch_s, end_epoch_s, max(1, ceil(duration / sample_step_s)) + 1)
    solution = solve_ivp(
        two_body_derivative, (initial.epoch_s, end_epoch_s), initial.vector,
        args=(body.mu_m3_s2,), method="DOP853", t_eval=times, rtol=settings.rtol,
        atol=[settings.position_atol_m]*3 + [settings.velocity_atol_m_s]*3,
        max_step=settings.max_step_s,
    )
    if not solution.success or solution.t[-1] != end_epoch_s:
        raise RuntimeError(f"Propagation failed: {solution.message}")
    return Trajectory(solution.t, solution.y.T, initial.frame)


def apply_impulse(state: CartesianState, burn: Impulse) -> CartesianState:
    """Apply a command at its exact epoch; preserve position and the input objects."""
    if burn.frame != state.frame:
        raise ValueError("Burn and state frames differ; transform the command first.")
    if burn.epoch_s != state.epoch_s:
        raise ValueError("Propagate to the burn epoch before applying the impulse.")
    return CartesianState(state.epoch_s, state.position_m,
                          state.velocity_m_s + burn.delta_v_m_s, state.frame)


# A different simulation can supply a callable with this same three-argument contract.
Propagator = Callable[[CartesianState, float, CentralBody], Trajectory]


def execute_candidate(candidate: ManeuverCandidate, *,
                      propagator: Propagator = propagate_two_body,
                      burn_executor: Callable[[CartesianState, Impulse], CartesianState] = apply_impulse,
                      verification_duration_s: float = 0.0) -> ExecutionResult:
    """Execute scheduled impulses with separate coast segments on either side.

    propagator and burn_executor can be replaced independently by compatible
    components. The optional verification coast is separate from the mission's end.
    Every event retains a before/after state even when the epochs are identical.
    """
    if not isfinite(verification_duration_s) or verification_duration_s < 0:
        raise ValueError("Verification duration must be finite and nonnegative.")
    state = candidate.initial_state
    epochs, vectors = [state.epoch_s], [state.vector]
    coasts, records = [], []

    def coast_until(end):
        nonlocal state
        if end == state.epoch_s:
            return None
        segment = propagator(state, end, candidate.body)
        if (segment.frame != state.frame or segment.epochs_s[0] != state.epoch_s
                or segment.epochs_s[-1] != end
                or not np.array_equal(segment.states[0], state.vector)):
            raise ValueError("Propagator must preserve the frame and initial state and reach the requested epoch.")
        coasts.append(segment)
        epochs.extend(segment.epochs_s[1:])
        vectors.extend(segment.states[1:])
        state = segment.state(-1)
        return segment

    for burn in candidate.burns:
        coast_until(burn.epoch_s)
        after = burn_executor(state, burn)
        if after.epoch_s != state.epoch_s or after.frame != state.frame:
            raise ValueError("Burn executor must preserve the command epoch and inertial frame.")
        records.append(BurnRecord(burn, state, after))
        state = after
        epochs.append(state.epoch_s)
        vectors.append(state.vector)
    coast_until(candidate.end_epoch_s)
    terminal = state
    verification = coast_until(candidate.end_epoch_s + verification_duration_s)
    return ExecutionResult(candidate, Trajectory(epochs, vectors, state.frame), tuple(coasts),
                           tuple(records), terminal, verification)
