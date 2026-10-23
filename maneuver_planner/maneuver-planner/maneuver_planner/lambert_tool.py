"""State-based Lambert adapter; endpoints may belong to arbitrary orbit types."""

from dataclasses import dataclass

import numpy as np

from .gnc_tool import Applicability
from .lambert import solve_lambert
from .models import CartesianState, CentralBody, Impulse, ManeuverCandidate, _array


@dataclass(frozen=True)
class LambertRequest:
    initial_state: CartesianState
    target_state: CartesianState
    body: CentralBody
    plane_normal: np.ndarray | None = None

    def __post_init__(self):
        if self.initial_state.frame != self.target_state.frame:
            raise ValueError("Lambert endpoint states must use the same inertial frame.")
        if self.target_state.epoch_s <= self.initial_state.epoch_s:
            raise ValueError("Lambert target epoch must follow the initial epoch.")
        for state in (self.initial_state, self.target_state):
            if np.linalg.norm(state.position_m) <= self.body.radius_m:
                raise ValueError("Lambert endpoints must be outside the central body.")
        if self.plane_normal is not None:
            object.__setattr__(self, "plane_normal", _array(self.plane_normal, (3,), "plane_normal"))


class LambertTool:
    name = "lambert"
    description = "Zero-revolution targeting between two specified inertial states."

    def check_applicability(self, request: LambertRequest):
        try:
            self._solve(request)
        except (ValueError, RuntimeError) as error:
            return Applicability(False, str(error))
        return Applicability(True, "Two-body endpoint targeting; initial orbit need not be circular.")

    @staticmethod
    def _solve(request):
        initial, target = request.initial_state, request.target_state
        return solve_lambert(initial.position_m, target.position_m,
                             target.epoch_s-initial.epoch_s, request.body.mu_m3_s2,
                             plane_normal=request.plane_normal)

    def propose(self, request: LambertRequest):
        initial, target = request.initial_state, request.target_state
        solution = self._solve(request)
        return ManeuverCandidate(self.name, initial, request.body, (
            Impulse(initial.epoch_s, solution.departure_velocity_m_s-initial.velocity_m_s, initial.frame),
            Impulse(target.epoch_s, target.velocity_m_s-solution.arrival_velocity_m_s, initial.frame),
        ), target.epoch_s, "Specified endpoint/time; full target velocity matched by the arrival burn.")
