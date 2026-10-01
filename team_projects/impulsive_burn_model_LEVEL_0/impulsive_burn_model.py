"""
Impulsive burn model for Level 0 (400 km -> 500 km Hohmann transfer).

Basically: a real burn has duration, gravity losses, etc. but for a
chemical thruster firing for a few seconds on a ~93 min orbit, we can just
treat it as instantaneous - position and time don't change, only
velocity does. Standard first-pass assumption in astrodynamics, good
enough for Level 0.

Units: km, s, km/s. mu = 398600.4418 km^3/s^2 for Earth. If we switch to
meters later this all still works, just has to be consistent everywhere.

Not tracking propellant/mass here - Level 0 doesn't ask for it (just
burn -> coast -> burn -> check orbit). Can add the rocket equation back
in later if we need fuel usage.

"""

from dataclasses import dataclass, replace
import numpy as np


@dataclass
class StateVector:
    """
    Spacecraft state at one instant. Placeholder until the real
    state-vector component exists - just need something with
    position/velocity/time for this to work with.

    position, velocity: (3,) arrays, km and km/s (e.g. ECI)
    time: seconds
    """
    position: np.ndarray
    velocity: np.ndarray
    time: float

    def __post_init__(self):
        # force everything to a flat 3-vector regardless of whether it
        # came in as a list, tuple, etc.
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)


@dataclass
class BurnResult:
    """
    What apply_impulsive_burn returns - the new state plus the burn info,
    so whatever logs maneuvers later doesn't have to recompute dv_mag
    itself.
    """
    state: StateVector
    delta_v_vector: np.ndarray
    delta_v_magnitude: float


def apply_impulsive_burn(state: StateVector, delta_v: np.ndarray) -> BurnResult:
    """
    Applies an instant dv to the state.

    position_after = position_before
    time_after = time_before
    velocity_after = velocity_before + delta_v

    That's really it - there's no physics to derive here since we're
    defining the burn as instantaneous. All the actual math happens
    wherever delta_v came from (Hohmann calc, Lambert, whatever).

    delta_v needs to be in the same frame as state.velocity. Doesn't
    matter if it's prograde (see prograde_delta_v below) or some other
    direction.

    This function does NOT check constraints (keep-out zones, dv budget)
    and does NOT propagate the orbit afterward - those are separate
    components.
    """
    dv = np.asarray(delta_v, dtype=float).reshape(3)
    dv_mag = float(np.linalg.norm(dv))

    new_velocity = state.velocity + dv

    # replace() makes a new StateVector instead of mutating state in
    # place, so the caller still has the pre-burn state if they need it
    new_state = replace(state, velocity=new_velocity)

    return BurnResult(
        state=new_state,
        delta_v_vector=dv,
        delta_v_magnitude=dv_mag,
    )


def prograde_delta_v(velocity: np.ndarray, dv_magnitude: float) -> np.ndarray:
    """
    Turns a dv magnitude into a full vector pointed along the current
    velocity direction. Need this because the Hohmann equations only
    give you a magnitude (like "28 m/s"), not a direction, and both
    Level 0 burns are prograde:
      - burn 1 speeds up at periapsis -> raises apoapsis to the transfer
        ellipse
      - burn 2 speeds up at apoapsis -> circularizes at the new altitude

    unit_vector = v / |v|
    delta_v = dv_magnitude * unit_vector

    Negative dv_magnitude just flips it retrograde (slows down instead).
    """
    v = np.asarray(velocity, dtype=float).reshape(3)
    speed = np.linalg.norm(v)

    if speed == 0:
        raise ValueError("Cannot define a prograde direction from zero velocity.")

    return dv_magnitude * (v / speed)