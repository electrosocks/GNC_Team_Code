"""Three signed tangential impulses through a specified intermediate apoapsis."""

from dataclasses import dataclass
from math import isclose, pi, sqrt

import numpy as np

from .gnc_tool import Applicability
from .hohmann_tool import HohmannTool
from .models import Impulse, ManeuverCandidate, TransferRequest, positive


@dataclass(frozen=True)
class BiellipticSolution:
    delta_v_m_s: tuple[float, ...]
    coast_times_s: tuple[float, ...]

    @property
    def total_delta_v_m_s(self):
        return sum(abs(v) for v in self.delta_v_m_s)

    @property
    def time_of_flight_s(self):
        return sum(self.coast_times_s)


def bielliptic_transfer(initial_radius_m, target_radius_m, apoapsis_radius_m, mu_m3_s2):
    """Radii are body-centered meters; positive burn scalars follow local motion."""
    r1, r2, rb, mu = initial_radius_m, target_radius_m, apoapsis_radius_m, mu_m3_s2
    for value, name in ((r1, "initial_radius_m"), (r2, "target_radius_m"),
                        (rb, "apoapsis_radius_m"), (mu, "mu_m3_s2")):
        positive(value, name)
    if rb <= max(r1, r2):
        raise ValueError("Intermediate apoapsis must exceed both orbit radii.")
    if isclose(r1, r2, rel_tol=8*np.finfo(float).eps):
        return BiellipticSolution((), ())
    a1, a2 = (r1+rb)/2, (r2+rb)/2
    def speed(radius, axis):
        return sqrt(mu*(2/radius-1/axis))
    return BiellipticSolution(
        (speed(r1, a1)-sqrt(mu/r1), speed(rb, a2)-speed(rb, a1),
         sqrt(mu/r2)-speed(r2, a2)),
        (pi*sqrt(a1**3/mu), pi*sqrt(a2**3/mu)),
    )


class BiellipticTool:
    name = "bielliptic"
    description = "Three impulses between coplanar circular orbits."

    def __init__(self, apoapsis_radius_m):
        positive(apoapsis_radius_m, "apoapsis_radius_m")
        self.apoapsis_radius_m = apoapsis_radius_m

    def check_applicability(self, request: TransferRequest):
        common = HohmannTool().check_applicability(request)
        if not common.applicable:
            return common
        if self.apoapsis_radius_m <= max(np.linalg.norm(request.initial_state.position_m), request.target_radius_m):
            return Applicability(False, "Intermediate apoapsis must exceed both orbit radii.")
        return Applicability(True, "Circular coplanar orbits with an external intermediate apoapsis.")

    def propose(self, request: TransferRequest):
        applicability = self.check_applicability(request)
        if not applicability.applicable:
            raise ValueError(applicability.reason)
        initial = request.initial_state
        r1 = float(np.linalg.norm(initial.position_m))
        solution = bielliptic_transfer(r1, request.target_radius_m,
                                       self.apoapsis_radius_m, request.body.mu_m3_s2)
        normal = np.cross(initial.position_m, initial.velocity_m_s)
        tangent = np.cross(normal/np.linalg.norm(normal), initial.position_m/r1)
        burns, epoch = [], initial.epoch_s
        for i, dv in enumerate(solution.delta_v_m_s):
            if i:
                epoch += solution.coast_times_s[i-1]
            # The inertial tangent reverses after each half-ellipse.
            burns.append(Impulse(epoch, (-1)**i*dv*tangent, initial.frame))
        return ManeuverCandidate(self.name, initial, request.body, tuple(burns), epoch,
                                 f"Intermediate apoapsis radius {self.apoapsis_radius_m:g} m.")
