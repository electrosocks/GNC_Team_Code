"""Reusable Clohessy-Wiltshire (CW/HCW) relative-motion equations.

Requires NumPy only. Run `python cw_equations.py` for a two-impulse example.

State order: [x, y, z, x_dot, y_dot, z_dot].
Units: meters, seconds, meters/second; mean motion n in radians/second.
The origin follows a circular reference orbit. Right-handed rotating Hill axes:
    x: radially outward from the central body through the reference spacecraft
    y: along the reference spacecraft's direction of motion
    z: along its orbital angular momentum (x cross y)
Velocities are derivatives IN THIS ROTATING FRAME, not inertial velocity
differences. Converting an inertial relative velocity also requires subtracting
the frame-rotation contribution omega cross relative_position.

Assumptions: point-mass gravity, a circular reference orbit, and separations small
relative to its radius. This is a first-order relative-motion model; it is not an
exact nonlinear two-body propagator or a general absolute-orbit transfer solver.

Equations (with no applied acceleration):
    x_ddot = 3*n**2*x + 2*n*y_dot
    y_ddot = -2*n*x_dot
    z_ddot = -n**2*z

Reference: MIT 16.346 Astrodynamics, Lecture 26, radial xi / along-track eta axes:
https://ocw.mit.edu/courses/16-346-astrodynamics-fall-2008/e4f0632a9f1c98f7e9b25492e1a30eb1_lec_26.pdf
"""

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt

import numpy as np


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _vector(values, length: int, name: str) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with {length} components.")
    return result


def mean_motion(mu_m3_s2: float, reference_radius_m: float) -> float:
    """Return n = sqrt(mu / radius**3). Supply radius from the body center."""
    _positive(mu_m3_s2, "mu_m3_s2")
    _positive(reference_radius_m, "reference_radius_m")
    return sqrt(mu_m3_s2 / reference_radius_m**3)


def cw_system_matrix(n_rad_s: float) -> np.ndarray:
    """Return A for state_dot = A @ state, in the state order stated above."""
    _positive(n_rad_s, "n_rad_s")
    n = n_rad_s
    return np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n*n, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n*n, 0, 0, 0],
    ], dtype=float)


def cw_derivative(time_s: float, state, n_rad_s: float) -> np.ndarray:
    """ODE right-hand side; signature supports solve_ivp(..., args=(n,)).

    time_s is unused because the unforced CW system is time invariant.
    """
    return cw_system_matrix(n_rad_s) @ _vector(state, 6, "state")


def cw_stm(time_s: float, n_rad_s: float) -> np.ndarray:
    """Return the analytical 6x6 state-transition matrix Phi(time_s).

    state(t0 + time_s) = Phi(time_s) @ state(t0). Negative time is supported.
    This matrix propagates an unforced coast; apply impulsive burns separately.
    """
    _positive(n_rad_s, "n_rad_s")
    if not isfinite(time_s):
        raise ValueError("time_s must be finite.")
    n = n_rad_s
    u = n*time_s
    if not isfinite(u):
        raise ValueError("n * time_s must be finite.")
    s, c = sin(u), cos(u)
    one_minus_c = 2*sin(u/2)**2  # Stable for very short propagation intervals.
    s_minus_u = (u**3*(-1/6+u*u*(1/120-u*u/5040))
                 if abs(u) < 1e-3 else s-u)
    return np.array([
        [1+3*one_minus_c, 0, 0, s/n, 2*one_minus_c/n, 0],
        [6*s_minus_u, 1, 0, -2*one_minus_c/n, (u+4*s_minus_u)/n, 0],
        [0, 0, c, 0, 0, s/n],
        [3*n*s, 0, 0, c, 2*s, 0],
        [-6*n*one_minus_c, 0, 0, -2*s, 1-4*one_minus_c, 0],
        [0, 0, -n*s, 0, 0, c],
    ], dtype=float)


def propagate_cw(initial_state, times_s, n_rad_s: float) -> np.ndarray:
    """Propagate a relative state without changing its input.

    times_s is elapsed time from the supplied initial state, not an absolute epoch.
    A scalar time returns shape (6,); a nonempty 1D time array returns shape (N, 6).
    The time array need not be sorted, and can include negative elapsed times.
    """
    state = _vector(initial_state, 6, "initial_state")
    _positive(n_rad_s, "n_rad_s")
    times = np.asarray(times_s, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain finite values.")
    if times.ndim == 0:
        return cw_stm(float(times), n_rad_s) @ state
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_s must be a scalar or a nonempty one-dimensional array.")
    return np.array([cw_stm(float(time), n_rad_s) @ state for time in times])


@dataclass(frozen=True)
class CWTransfer:
    """Proposed local-frame impulses and their predicted CW coast states."""

    departure_delta_v_m_s: np.ndarray
    arrival_delta_v_m_s: np.ndarray
    state_after_departure: np.ndarray
    state_before_arrival: np.ndarray
    final_state: np.ndarray
    transfer_time_s: float
    targeting_condition_number: float

    def __post_init__(self):
        for name, length in (("departure_delta_v_m_s", 3), ("arrival_delta_v_m_s", 3),
                             ("state_after_departure", 6), ("state_before_arrival", 6),
                             ("final_state", 6)):
            values = _vector(getattr(self, name), length, name)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def total_delta_v_m_s(self) -> float:
        return float(np.linalg.norm(self.departure_delta_v_m_s)
                     + np.linalg.norm(self.arrival_delta_v_m_s))


def target_cw(initial_state, target_position_m, transfer_time_s: float,
              n_rad_s: float, *, target_velocity_m_s=None,
              max_condition_number: float = 1e10) -> CWTransfer:
    """Propose two impulses to reach a requested relative position and velocity.

    Burn 1 sets the required departure velocity. Coast under CW for the supplied
    duration. Burn 2 changes arrival velocity to target_velocity_m_s (default zero).
    Both returned delta-v vectors are in the rotating Hill axes AT THEIR OWN burn
    epochs. Transform them separately before commanding an inertial simulator.

    Solve Phi_rv @ v_departure = r_target - Phi_rr @ r_initial using np.linalg.solve.
    Singular/ill-conditioned 3D targeting times are rejected, including integer
    half-periods where cross-track targeting loses a degree of freedom. Some
    special planar problems can still have solutions at those rejected times.
    There are no thrust, collision, approach-corridor, or delta-v-budget checks.
    """
    _positive(transfer_time_s, "transfer_time_s")
    _positive(max_condition_number, "max_condition_number")
    if max_condition_number < 1:
        raise ValueError("max_condition_number must be at least 1.")
    initial = _vector(initial_state, 6, "initial_state")
    position = _vector(target_position_m, 3, "target_position_m")
    velocity = (np.zeros(3) if target_velocity_m_s is None
                else _vector(target_velocity_m_s, 3, "target_velocity_m_s"))
    phi = cw_stm(transfer_time_s, n_rad_s)
    phi_rr, phi_rv = phi[:3, :3], phi[:3, 3:]
    condition = float(np.linalg.cond(phi_rv))
    if not isfinite(condition) or condition > max_condition_number:
        raise ValueError("CW targeting is singular or ill-conditioned at this time; choose another transfer duration.")
    required_velocity = np.linalg.solve(phi_rv, position-phi_rr @ initial[:3])
    delta_v1 = required_velocity-initial[3:]
    departure = np.concatenate((initial[:3], required_velocity))
    arrival = phi @ departure
    delta_v2 = velocity-arrival[3:]
    final = arrival.copy()
    final[3:] += delta_v2
    return CWTransfer(delta_v1, delta_v2, departure, arrival, final,
                       transfer_time_s, condition)


def _demo() -> None:
    # Example: rendezvous from 1 km behind a reference spacecraft in a 500 km
    # circular Earth orbit. These are Hill-relative states, not ECI states.
    mu = 3.986004418e14
    reference_radius = 6_378_137.0+500_000.0
    n = mean_motion(mu, reference_radius)
    initial = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    solution = target_cw(initial, [0.0, 0.0, 0.0], 1200.0, n)
    times = np.linspace(0.0, solution.transfer_time_s, 121)
    trajectory = propagate_cw(solution.state_after_departure, times, n)
    print("CW rendezvous example: circular reference orbit at 500 km")
    print("Hill axes: x radial outward, y along-track, z orbit normal")
    print("Initial relative state:", initial)
    print("Transfer duration (s):", solution.transfer_time_s)
    print("Departure delta-v (m/s):", solution.departure_delta_v_m_s)
    print("Arrival delta-v (m/s):", solution.arrival_delta_v_m_s)
    print("Total delta-v (m/s):", solution.total_delta_v_m_s)
    print("Final relative state after arrival burn:", solution.final_state)
    print("Trajectory array shape:", trajectory.shape)


if __name__ == "__main__":
    _demo()
