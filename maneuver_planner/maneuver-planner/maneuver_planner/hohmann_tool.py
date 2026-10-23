"""Hohmann mathematics plus a thin adapter to the shared maneuver interface.

No Earth constants, SciPy propagation, plots, file I/O, or mission planning live
here. Assumptions: two-body gravity, circular coplanar orbits, the same direction
of orbital motion, and two ideal tangential impulses.
"""

from dataclasses import dataclass
from math import isclose, pi, sqrt

import numpy as np

from .gnc_tool import Applicability
from .models import Impulse, ManeuverCandidate, TransferRequest, positive
from .orbital_utils import orbital_elements


@dataclass(frozen=True)
class HohmannSolution:
    transfer_semimajor_axis_m: float
    departure_delta_v_m_s: float
    arrival_delta_v_m_s: float
    time_of_flight_s: float

    @property
    def total_delta_v_m_s(self) -> float:
        return abs(self.departure_delta_v_m_s) + abs(self.arrival_delta_v_m_s)


def hohmann_transfer(initial_radius_m: float, target_radius_m: float,
                     mu_m3_s2: float) -> HohmannSolution:
    """Calculate a transfer using only radii and the gravitational parameter.

    Radii are measured from the body center, in meters. Signed burn scalars are
    positive along the local direction of motion (negative for an orbit lowering).
    Radii equal to floating-point precision mean the goal is already met: zero
    burns and zero elapsed time. This avoids a spurious half-orbit coast after
    roundoff in a rotated initial state's norm.
    """
    for value, name in ((initial_radius_m, "initial_radius_m"),
                        (target_radius_m, "target_radius_m"), (mu_m3_s2, "mu_m3_s2")):
        positive(value, name)
    r1, r2, mu = initial_radius_m, target_radius_m, mu_m3_s2
    a = (r1 + r2) / 2
    if isclose(r1, r2, rel_tol=8*np.finfo(float).eps):
        return HohmannSolution(a, 0.0, 0.0, 0.0)
    # Vis-viva at the transfer's two apsides, minus each local circular speed.
    departure = sqrt(mu / r1) * (sqrt(2 * r2 / (r1 + r2)) - 1)
    arrival = sqrt(mu / r2) * (1 - sqrt(2 * r1 / (r1 + r2)))
    return HohmannSolution(a, departure, arrival, pi * sqrt(a**3 / mu))


class HohmannTool:
    name = "hohmann"
    description = "Two tangential impulses between coplanar circular orbits."

    def __init__(self, circularity_tolerance: float = 1e-8):
        positive(circularity_tolerance, "circularity_tolerance")
        self.circularity_tolerance = circularity_tolerance

    def check_applicability(self, request: TransferRequest) -> Applicability:
        radius = float(np.linalg.norm(request.initial_state.position_m))
        if radius <= request.body.radius_m:
            return Applicability(False, "The initial orbit is inside the central body.")
        try:
            elements = orbital_elements(request.initial_state, request.body.mu_m3_s2)
        except ValueError as error:
            return Applicability(False, str(error))
        if elements.eccentricity > self.circularity_tolerance:
            return Applicability(False, f"Initial orbit must be circular; e={elements.eccentricity:.6g}.")
        return Applicability(True, "Circular initial orbit; request preserves its plane and direction.")

    def propose(self, request: TransferRequest) -> ManeuverCandidate:
        applicable = self.check_applicability(request)
        if not applicable.applicable:
            raise ValueError(applicable.reason)
        initial = request.initial_state
        radius = float(np.linalg.norm(initial.position_m))
        solution = hohmann_transfer(radius, request.target_radius_m, request.body.mu_m3_s2)
        end = initial.epoch_s + solution.time_of_flight_s
        burns = ()
        if solution.time_of_flight_s > 0:
            rhat = initial.position_m / radius
            h = np.cross(initial.position_m, initial.velocity_m_s)
            tangent = np.cross(h / np.linalg.norm(h), rhat)
            # Half an ellipse later, the local prograde direction is reversed in
            # inertial axes. This also works for inclined and retrograde planes.
            burns = (
                Impulse(initial.epoch_s, solution.departure_delta_v_m_s * tangent, initial.frame),
                Impulse(end, -solution.arrival_delta_v_m_s * tangent, initial.frame),
            )
        return ManeuverCandidate(self.name, initial, request.body, burns, end, applicable.reason)
