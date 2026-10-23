# Adapted from team Lambert.py: relative import and explicit collinearity guard.
import numpy as np
from .lambert_helpers import y, solve_z



def solve_lambert(R1, R2, t, mu, orbit_type="pro"):
    """
    Solve Lambert's orbital boundary-value problem.

    Parameters
    ----------
    R1 : array_like
        Initial position vector [km].
    R2 : array_like
        Final position vector [km].
    t : float
        Time of flight [s].
    mu : float
        Gravitational parameter [km^3/s^2].
    orbit_type : str
        'pro' for prograde or 'retro' for retrograde.

    Returns
    -------
    V1 : ndarray
        Initial velocity vector [km/s].
    V2 : ndarray
        Final velocity vector [km/s].
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    if R1.shape != (3,) or R2.shape != (3,):
        raise ValueError("R1 and R2 must be three-dimensional vectors.")
    if not np.all(np.isfinite(R1)) or not np.all(np.isfinite(R2)):
        raise ValueError("Position vectors must contain finite values.")
    if not np.isfinite(t) or t <= 0:
        raise ValueError("Time of flight t must be positive and finite.")
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Gravitational parameter mu must be positive.")
    if np.linalg.norm(R1) == 0 or np.linalg.norm(R2) == 0:
        raise ValueError("Position vector magnitudes must be nonzero.")

    orbit_type = orbit_type.lower()
    if orbit_type not in ("pro", "retro"):
        print("** Prograde trajectory assumed.")
        orbit_type = "pro"

    # Magnitudes of R1 and R2
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    # Cross product and transfer angle
    c12 = np.cross(R1, R2)

    cos_theta = np.dot(R1, R2) / (r1 * r2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Determine prograde or retrograde direction
    if orbit_type == "pro":
        if c12[2] <= 0:
            theta = 2.0 * np.pi - theta
    else:  # retrograde
        if c12[2] >= 0:
            theta = 2.0 * np.pi - theta

    # Reject the collinear singularity explicitly, including antipodal endpoints.
    if np.linalg.norm(c12) / (r1 * r2) < 1e-10:
        raise ValueError("Collinear Lambert geometry is singular for this formulation.")

    # Equation 5.35
    denominator = 1.0 - np.cos(theta)
    if denominator <= 0:
        raise ValueError(
            "Transfer angle is degenerate; Lambert's solution "
            "cannot be computed with this formulation."
        )

    A = np.sin(theta) * np.sqrt(r1 * r2 / denominator)

    if abs(A) < 1e-14:
        raise ValueError(
            "A is approximately zero; the transfer geometry is "
            "singular for this formulation."
        )

    z = solve_z(t, r1, r2, A, mu)

    # Lagrange coefficients (Equations 5.46a, 5.46b, 5.46d)
    yz = y(z, r1, r2, A)
    f = 1.0 - yz / r1
    g = A * np.sqrt(yz / mu)
    gdot = 1.0 - yz / r2

    if abs(g) < 1e-14:
        raise RuntimeError("Lagrange coefficient g is approximately zero.")

    # Initial and final velocity vectors (Equations 5.28 and 5.29)
    V1 = (R2 - f * R1) / g
    V2 = (gdot * R2 - R1) / g

    return V1, V2