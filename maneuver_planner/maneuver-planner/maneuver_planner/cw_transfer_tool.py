"""CW orbit-change adapter and an explicit, optional nonlinear correction step.

CWLevel0Tool.propose uses only the linear CW model. correct_cw_candidate is a
separate simulation-assisted operation, called explicitly by the runner. Neither
routine calls a Hohmann or Lambert maneuver calculator or changes mission state.
All external states and returned impulses use SI and the input inertial frame.
"""

from math import isclose, isfinite, pi, sqrt

import numpy as np

from .cw_equations import cw_stm, mean_motion, target_cw
from .hohmann_tool import HohmannTool
from .models import CartesianState, Impulse, ManeuverCandidate, TransferRequest, positive


def hill_basis(reference: CartesianState) -> np.ndarray:
    """Columns are radial outward, transverse prograde, and orbit normal in ECI."""
    radial = reference.position_m/np.linalg.norm(reference.position_m)
    normal = np.cross(reference.position_m, reference.velocity_m_s)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        raise ValueError("A radial state cannot define a Hill frame.")
    normal = normal/magnitude
    return np.column_stack((radial, np.cross(normal, radial), normal))


def relative_state_hill(chaser: CartesianState, reference: CartesianState) -> np.ndarray:
    """Convert simultaneous inertial states into rotating Hill-relative coordinates.

    Assumes a two-body reference orbit with fixed plane. The reference used by
    CWLevel0Tool is circular. Includes the essential omega cross rho velocity term.
    """
    if chaser.frame != reference.frame or chaser.epoch_s != reference.epoch_s:
        raise ValueError("Chaser and reference need the same frame and epoch.")
    basis = hill_basis(reference)
    rho = basis.T@(chaser.position_m-reference.position_m)
    angular_rate = np.linalg.norm(np.cross(reference.position_m, reference.velocity_m_s))/np.dot(reference.position_m, reference.position_m)
    velocity = basis.T@(chaser.velocity_m_s-reference.velocity_m_s)-np.cross([0, 0, angular_rate], rho)
    return np.concatenate((rho, velocity))


def circular_reference(request: TransferRequest, elapsed_s: float) -> CartesianState:
    """Virtual circular reference on the target orbit, initially aligned with chaser."""
    if not isfinite(elapsed_s):
        raise ValueError("elapsed_s must be finite.")
    basis = hill_basis(request.initial_state)
    radius = request.target_radius_m
    n = mean_motion(request.body.mu_m3_s2, radius)
    c, s = np.cos(n*elapsed_s), np.sin(n*elapsed_s)
    return CartesianState(request.initial_state.epoch_s+elapsed_s,
                          basis@np.array([radius*c, radius*s, 0.0]),
                          basis@np.array([-n*radius*s, n*radius*c, 0.0]),
                          request.initial_state.frame)


class CWLevel0Tool:
    name = "cw_linear_two_impulse"
    description = "CW targeting around a virtual circular reference on the target orbit."

    def __init__(self, transfer_time_s: float | None = None):
        if transfer_time_s is not None:
            positive(transfer_time_s, "transfer_time_s")
        self.transfer_time_s = transfer_time_s

    def check_applicability(self, request: TransferRequest):
        # Reuse the existing circular-state check, not its maneuver calculation.
        return HohmannTool().check_applicability(request)

    def duration(self, request: TransferRequest) -> float:
        radius = float(np.linalg.norm(request.initial_state.position_m))
        if isclose(radius, request.target_radius_m, rel_tol=8*np.finfo(float).eps):
            return 0.0
        if self.transfer_time_s is not None:
            return self.transfer_time_s
        # Comparison clock: half a Kepler period at the mean of the two radii.
        return pi*sqrt(((radius+request.target_radius_m)/2)**3/request.body.mu_m3_s2)

    def linear_solution(self, request: TransferRequest):
        duration = self.duration(request)
        if duration == 0:
            return None
        reference = circular_reference(request, 0.0)
        relative = relative_state_hill(request.initial_state, reference)
        n = mean_motion(request.body.mu_m3_s2, request.target_radius_m)
        return target_cw(relative, [0, 0, 0], duration, n)

    def propose(self, request: TransferRequest) -> ManeuverCandidate:
        applicability = self.check_applicability(request)
        if not applicability.applicable:
            raise ValueError(applicability.reason)
        initial = request.initial_state
        duration = self.duration(request)
        if duration == 0:
            return ManeuverCandidate(self.name, initial, request.body, (), initial.epoch_s,
                                     "The requested circular orbit is already satisfied.")
        solution = self.linear_solution(request)
        initial_reference = circular_reference(request, 0.0)
        arrival_reference = circular_reference(request, duration)
        return ManeuverCandidate(self.name, initial, request.body, (
            Impulse(initial.epoch_s, hill_basis(initial_reference)@solution.departure_delta_v_m_s, initial.frame),
            Impulse(initial.epoch_s+duration, hill_basis(arrival_reference)@solution.arrival_delta_v_m_s, initial.frame),
        ), initial.epoch_s+duration,
            "Pure CW prediction. Each local burn is transformed at its own epoch; no nonlinear correction yet.")

    def diagnostics(self, request: TransferRequest) -> dict:
        solution = self.linear_solution(request)
        reference = circular_reference(request, self.duration(request))
        relative = relative_state_hill(request.initial_state, circular_reference(request, 0.0))
        return {
            "model": "linear CW about a circular virtual reference at the target radius",
            "mean_motion_rad_s": mean_motion(request.body.mu_m3_s2, request.target_radius_m),
            "initial_relative_state_hill": relative.tolist(),
            "initial_separation_to_reference_radius_ratio": float(np.linalg.norm(relative[:3])/request.target_radius_m),
            "cw_predicted_terminal_position_error_m": float(np.linalg.norm(solution.final_state[:3])) if solution else 0.0,
            "targeting_condition_number": solution.targeting_condition_number if solution else None,
            "reference_arrival_position_m": reference.position_m.tolist(),
            "reference_arrival_velocity_m_s": reference.velocity_m_s.tolist(),
            "raw_cw_delta_v_m_s": solution.total_delta_v_m_s if solution else 0.0,
        }


def correct_cw_candidate(request: TransferRequest, candidate: ManeuverCandidate, *,
                         propagator=None, position_tolerance_m: float = 1e-5,
                         max_iterations: int = 12):
    """Refine a CW seed with its CW sensitivity and independently predicted misses.

    At each iteration, propagate under the supplied two-body model, express the
    position miss in the arrival Hill frame, and solve Phi_rv*delta_v0 = -miss.
    Map that departure correction using the INITIAL Hill basis. The arrival burn
    then matches the circular reference's inertial velocity. Return a new candidate
    and a diagnostic trace. This is explicitly CW plus nonlinear correction.
    """
    positive(position_tolerance_m, "position_tolerance_m")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if candidate.body != request.body or candidate.initial_state.frame != request.initial_state.frame or not np.array_equal(candidate.initial_state.vector, request.initial_state.vector) or candidate.initial_state.epoch_s != request.initial_state.epoch_s:
        raise ValueError("Candidate and request must use the same body and initial state.")
    if candidate.duration_s == 0 and not candidate.burns:
        return candidate, {"enabled": True, "applied": False, "iterations": 0, "position_errors_m": [0.0]}
    if len(candidate.burns) != 2 or candidate.burns[0].epoch_s != request.initial_state.epoch_s or candidate.burns[1].epoch_s != candidate.end_epoch_s:
        raise ValueError("CW correction requires departure and arrival impulses at the interval endpoints.")
    if propagator is None:
        from .physics import propagate_two_body
        propagator = propagate_two_body
    initial = request.initial_state
    duration = candidate.duration_s
    initial_basis = hill_basis(circular_reference(request, 0.0))
    target = circular_reference(request, duration)
    arrival_basis = hill_basis(target)
    n = mean_motion(request.body.mu_m3_s2, request.target_radius_m)
    sensitivity = cw_stm(duration, n)[:3, 3:]
    if np.linalg.cond(sensitivity) > 1e10:
        raise ValueError("CW correction is ill-conditioned at this transfer time.")
    departure_velocity = initial.velocity_m_s+candidate.burns[0].delta_v_m_s
    errors = []
    raw_velocity_error = None
    for iteration in range(max_iterations+1):
        coast_initial = CartesianState(initial.epoch_s, initial.position_m, departure_velocity, initial.frame)
        arrival = propagator(coast_initial, candidate.end_epoch_s, request.body).state(-1)
        if arrival.epoch_s != candidate.end_epoch_s or arrival.frame != initial.frame:
            raise ValueError("Correction propagator must reach the requested epoch and preserve the frame.")
        residual = arrival_basis.T@(arrival.position_m-target.position_m)
        error = float(np.linalg.norm(residual))
        errors.append(error)
        if raw_velocity_error is None:
            raw_velocity_error = float(np.linalg.norm(arrival.velocity_m_s+candidate.burns[1].delta_v_m_s-target.velocity_m_s))
        if error <= position_tolerance_m:
            break
        if iteration == max_iterations:
            raise RuntimeError(f"CW nonlinear correction did not converge: position error {error:.6g} m.")
        departure_velocity = departure_velocity+initial_basis@np.linalg.solve(sensitivity, -residual)
    corrected = ManeuverCandidate("cw_with_two_body_correction", initial, request.body, (
        Impulse(initial.epoch_s, departure_velocity-initial.velocity_m_s, initial.frame),
        Impulse(candidate.end_epoch_s, target.velocity_m_s-arrival.velocity_m_s, initial.frame),
    ), candidate.end_epoch_s,
        "CW seed plus explicitly enabled two-body terminal correction; no Hohmann/Lambert fallback.")
    return corrected, {
        "enabled": True, "applied": True, "iterations": iteration,
        "position_errors_m": errors, "position_tolerance_m": position_tolerance_m,
        "raw_two_body_position_error_m": errors[0],
        "raw_two_body_velocity_error_m_s": raw_velocity_error,
        "departure_delta_v_change_m_s": float(np.linalg.norm(corrected.burns[0].delta_v_m_s-candidate.burns[0].delta_v_m_s)),
        "arrival_burn_source": "circular reference velocity minus nonlinear predicted arrival velocity",
    }
