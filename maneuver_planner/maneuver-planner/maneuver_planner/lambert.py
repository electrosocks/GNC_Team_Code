"""Zero-revolution two-body Lambert solver, in SI units.

Lancaster-Blanchard/Izzo time equation and algebraic velocity reconstruction,
with a bracketed scalar solve. Reference: Izzo (2015), DOI 10.1007/s10569-014-9587-y,
Eqs. 18, 20 and the velocity reconstruction in Algorithm 1.
This is not the paper's Householder implementation. No Hohmann fallback is used.
"""

from dataclasses import dataclass
from math import asinh, atan2, isfinite, pi, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1

from .models import _array, positive


@dataclass(frozen=True)
class LambertSolution:
    departure_velocity_m_s: np.ndarray
    arrival_velocity_m_s: np.ndarray
    transfer_angle_rad: float
    time_residual_s: float
    iterations: int

    def __post_init__(self):
        for name in ("departure_velocity_m_s", "arrival_velocity_m_s"):
            object.__setattr__(self, name, _array(getattr(self, name), (3,), name))


def solve_lambert(initial_position_m, final_position_m, time_of_flight_s, mu_m3_s2,
                  *, plane_normal=None, revolutions=0, max_iterations=100):
    """Return the zero-revolution velocities connecting two positions in a fixed frame.

    plane_normal specifies the desired angular-momentum direction, including
    retrograde or long-way motion. It must be perpendicular to both positions.
    Without it, noncollinear endpoints select the short path. Exactly antipodal
    endpoints require it. Same-ray/radial and coincident endpoints are explicitly
    unsupported. The caller evaluates body clearance and maneuver constraints.
    """
    r1v = _array(initial_position_m, (3,), "initial_position_m")
    r2v = _array(final_position_m, (3,), "final_position_m")
    positive(time_of_flight_s, "time_of_flight_s")
    positive(mu_m3_s2, "mu_m3_s2")
    if revolutions != 0:
        raise ValueError("Only zero-revolution Lambert transfers are supported.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    r1, r2 = float(np.linalg.norm(r1v)), float(np.linalg.norm(r2v))
    positive(r1, "initial position magnitude")
    positive(r2, "final position magnitude")
    ir1, ir2 = r1v/r1, r2v/r2
    cross = np.cross(ir1, ir2)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(ir1, ir2), -1, 1))
    if sine < 1e-12 and cosine > 0:
        raise ValueError("Same-ray/radial and coincident endpoints are unsupported.")
    if plane_normal is None:
        if sine < 1e-12:
            raise ValueError("Antipodal endpoints require an explicit plane_normal.")
        normal = cross/sine
    else:
        normal = _array(plane_normal, (3,), "plane_normal")
        magnitude = float(np.linalg.norm(normal))
        positive(magnitude, "plane_normal magnitude")
        normal = normal/magnitude
        if max(abs(np.dot(normal, ir1)), abs(np.dot(normal, ir2))) > 1e-12:
            raise ValueError("plane_normal must be perpendicular to both endpoint positions.")
    long_way = np.dot(normal, cross) < 0
    angle = atan2(sine, cosine)
    if long_way:
        angle = 2*pi-angle

    # Nondimensionalize before root finding; SI remains the public interface.
    scale = max(r1, r2)
    velocity_scale = sqrt(mu_m3_s2/scale)
    time_scale = scale/velocity_scale
    rr1, rr2 = r1/scale, r2/scale
    chord = float(np.linalg.norm(r2v/scale-r1v/scale))
    s = (rr1+rr2+chord)/2
    # This expression avoids subtracting nearly equal numbers near 180 degrees.
    lam = sqrt(rr1*rr2)*np.linalg.norm(ir1+ir2)/(2*s)
    if long_way:
        lam = -lam
    desired = sqrt(2/s**3)*(time_of_flight_s/time_scale)
    positive(desired, "nondimensional time of flight")

    def tof(x):
        y = sqrt(1-lam*lam*(1-x*x))
        eta = (1-lam*lam)/(y+lam*x) if lam*x > 0 else y-lam*x
        if 0.8 < x < 1.2:
            # Battin's series removes cancellation at the parabolic boundary x=1.
            q = 4/3*float(hyp2f1(3, 1, 2.5, (1-lam-x*eta)/2))
            return (eta**3*q+4*lam*eta)/2
        denominator = 1-x*x
        if x < 1:
            psi = atan2(eta*sqrt(denominator), x*y+lam*denominator)
        else:
            psi = asinh(eta*sqrt(-denominator))
        return (psi/sqrt(abs(denominator))-x+lam*y)/denominator

    lower, upper = float(np.nextafter(-1.0, 0.0)), 2.0
    if tof(lower) < desired:
        raise RuntimeError("Flight time is outside the numerically resolved zero-revolution range.")
    for _ in range(64):
        if tof(upper) <= desired:
            break
        upper *= 2
    else:
        raise RuntimeError("No finite Lambert root bracket was found.")
    x, root = brentq(lambda value: tof(value)-desired, lower, upper,
                     xtol=1e-14, rtol=4*np.finfo(float).eps,
                     maxiter=max_iterations, full_output=True)
    residual_s = (tof(x)-desired)/sqrt(2/s**3)*time_scale
    if not isfinite(residual_s) or abs(residual_s) > max(1e-8, time_of_flight_s*1e-11):
        raise RuntimeError("Lambert time-of-flight residual exceeds tolerance.")
    y = sqrt(1-lam*lam*(1-x*x))
    gamma = sqrt(s/2)
    rho = (rr1-rr2)/chord
    sigma = min(1.0, sqrt(rr1*rr2)*np.linalg.norm(ir1-ir2)/chord)
    vr1 = gamma*((lam*y-x)-rho*(lam*y+x))/rr1
    vr2 = -gamma*((lam*y-x)+rho*(lam*y+x))/rr2
    transverse = (1-lam*lam)/(y-lam*x) if lam*x < 0 else y+lam*x
    vt = gamma*sigma*transverse
    v1 = velocity_scale*(vr1*ir1+vt/rr1*np.cross(normal, ir1))
    v2 = velocity_scale*(vr2*ir2+vt/rr2*np.cross(normal, ir2))
    return LambertSolution(v1, v2, angle, residual_s, root.iterations)
