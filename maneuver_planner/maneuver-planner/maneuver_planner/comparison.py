"""Four manual Level 0 options; reusable mathematics lives in separate tools."""

from dataclasses import dataclass
from math import isfinite, pi, sqrt

import numpy as np

from .bielliptic_tool import BiellipticTool
from .cw_transfer_tool import CWLevel0Tool, circular_reference
from .hohmann_tool import HohmannTool
from .lambert_tool import LambertRequest, LambertTool
from .models import CartesianState, TransferRequest, positive


@dataclass(frozen=True)
class MethodInfo:
    method_id: int
    name: str
    source: str


METHODS = (
    MethodInfo(0, "Hohmann", "hohmann_tool.py"),
    MethodInfo(1, "Bielliptic", "bielliptic_tool.py"),
    MethodInfo(2, "Lambert", "lambert.py + lambert_tool.py"),
    MethodInfo(3, "CW targeting", "cw_equations.py + cw_transfer_tool.py"),
)


def method_info(method_id):
    for info in METHODS:
        if info.method_id == method_id:
            return info
    raise ValueError("METHOD_ID must be 0 (Hohmann), 1 (bielliptic), 2 (Lambert), or 3 (CW).")


@dataclass(frozen=True)
class ComparisonSettings:
    intermediate_radius_m: float | None = None
    lambert_angle_deg: float = 180.0
    lambert_tof_s: float | None = None
    cw_tof_s: float | None = None

    def __post_init__(self):
        for name in ("intermediate_radius_m", "lambert_tof_s", "cw_tof_s"):
            if getattr(self, name) is not None:
                positive(getattr(self, name), name)
        if not isfinite(self.lambert_angle_deg) or not 0 < self.lambert_angle_deg < 360:
            raise ValueError("Lambert arrival angle must be strictly between 0 and 360 degrees.")


def orbital_basis(request):
    state = request.initial_state
    radial = state.position_m/np.linalg.norm(state.position_m)
    normal = np.cross(state.position_m, state.velocity_m_s)
    positive(float(np.linalg.norm(normal)), "orbital angular momentum")
    normal = normal/np.linalg.norm(normal)
    return radial, np.cross(normal, radial), normal


class Level0Tool:
    """Adapt the circular-orbit demonstration to one selected reusable capability."""

    def __init__(self, method_id, settings=None):
        self.info = method_info(method_id)
        self.settings = settings or ComparisonSettings()
        self.name, self.description = self.info.name.lower(), self.info.source

    def _lambert_request(self, request):
        p, q, normal = orbital_basis(request)
        radius = request.target_radius_m
        angle = np.deg2rad(self.settings.lambert_angle_deg)
        direction = np.cos(angle)*p+np.sin(angle)*q
        duration = self.settings.lambert_tof_s
        if duration is None:
            r1 = np.linalg.norm(request.initial_state.position_m)
            duration = pi*sqrt(((r1+radius)/2)**3/request.body.mu_m3_s2)
        target = CartesianState(request.initial_state.epoch_s+duration,
                                radius*direction,
                                sqrt(request.body.mu_m3_s2/radius)*np.cross(normal, direction),
                                request.initial_state.frame)
        return LambertRequest(request.initial_state, target, request.body, normal)

    def _tool(self, request):
        if self.info.method_id == 0:
            return HohmannTool(), request
        if self.info.method_id == 1:
            rb = self.settings.intermediate_radius_m
            if rb is None:
                rb = 1.5*max(np.linalg.norm(request.initial_state.position_m), request.target_radius_m)
            return BiellipticTool(rb), request
        if self.info.method_id == 2:
            return LambertTool(), self._lambert_request(request)
        return CWLevel0Tool(self.settings.cw_tof_s), request

    def check_applicability(self, request):
        tool, configured = self._tool(request)
        return tool.check_applicability(configured)

    def propose(self, request: TransferRequest):
        tool, configured = self._tool(request)
        return tool.propose(configured)

    def terminal_reference(self, request):
        p, _, normal = orbital_basis(request)
        if self.info.method_id == 2:
            state = self._lambert_request(request).target_state
            return state.position_m, state.velocity_m_s
        if self.info.method_id == 3:
            state = circular_reference(request, CWLevel0Tool(self.settings.cw_tof_s).duration(request))
            return state.position_m, state.velocity_m_s
        same = np.isclose(np.linalg.norm(request.initial_state.position_m), request.target_radius_m,
                          rtol=8*np.finfo(float).eps, atol=0)
        direction = -p if self.info.method_id == 0 and not same else p
        return (request.target_radius_m*direction,
                sqrt(request.body.mu_m3_s2/request.target_radius_m)*np.cross(normal, direction))

    def arrival_angle_deg(self, request, candidate):
        if candidate.duration_s == 0:
            return 0.0
        if self.info.method_id == 1:
            return 360.0
        p, q, _ = orbital_basis(request)
        position, _ = self.terminal_reference(request)
        return float(np.rad2deg(np.arctan2(np.dot(position, q), np.dot(position, p))) % 360)
