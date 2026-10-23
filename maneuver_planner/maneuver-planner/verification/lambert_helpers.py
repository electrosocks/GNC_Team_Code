# Team Lambert_helpers.py; numerical equations retained.
import numpy as np
from scipy.optimize import brentq


# Equation 5.38
def y(z, r1, r2, A):
    Cz = stumpC(z)
    Sz = stumpS(z)

    if Cz <= 0:
        return np.nan

    return r1 + r2 + A * (z * Sz - 1.0) / np.sqrt(Cz)

# Equation 5.40
def F(z, t, r1, r2, A, mu):
    Cz = stumpC(z)
    Sz = stumpS(z)
    yz = y(z, r1, r2, A)

    if Cz <= 0 or yz < 0 or not np.isfinite(yz):
        return np.nan

    return (
        (yz / Cz)**1.5 * Sz
        + A * np.sqrt(yz)
        - np.sqrt(mu) * t
    )

# Equation 5.43
def dFdz(z, r1, r2, A):
    yz = y(z, r1, r2, A)
    Cz = stumpC(z)
    Sz = stumpS(z)

    if not np.isfinite(yz) or yz <= 0 or Cz <= 0:
        return np.nan

    if abs(z) < 1e-7:
        return (
            np.sqrt(2) / 40 * yz ** 1.5
            + A / 8 * (
                np.sqrt(yz)
                + A / np.sqrt(2 * yz)
            )
        )
    return (
        (yz / Cz) ** 1.5 * (
        (Cz - 3 * Sz / (2 * Cz)) / (2 * z)
        + 3 * Sz ** 2 / (4 * Cz)
    )
        + A / 8 * (
            3 * Sz / Cz * np.sqrt(yz)
            + A * np.sqrt(Cz / yz)
        )
    )

def stumpC(z):
    """Stumpff C(z), with a series expansion near zero."""
    if abs(z) < 1e-4:
        return 0.5 - z/24 + z**2/720 - z**3/40320 + z**4/3628800
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z

    q = np.sqrt(-z)
    return (np.cosh(q) - 1) / (-z)

def stumpS(z):
    """Stumpff S(z), with a series expansion near zero."""
    if abs(z) < 1e-4:
        return 1.0/6.0 - z/120 + z**2/5040 - z**3/362880 + z**4/39916800
    if z > 0:
        q = np.sqrt(z)
        return (q - np.sin(q)) / q**3

    q = np.sqrt(-z)
    return (np.sinh(q) - q) / q**3

def solve_z(t, r1, r2, A, mu):
    """
    Find a root of F(z) on the zero-revolution branch.
    F(z) and y(z) are defined in Lambert.
    """
    # Search the elliptic branch up to, but not including, 4*pi^2.
    z_max = 4 * np.pi**2 - 1e-6
    grid = np.linspace(-100.0, z_max, 20001)

    z_prev = None
    f_prev = None

    def residual(z):
        return F(z, t, r1, r2, A, mu)

    for z_cur in grid:
        y_cur = y(z_cur, r1, r2, A)

        # Stay in the physically admissible domain of this formulation.
        if not np.isfinite(y_cur) or y_cur <= 0:
            z_prev = None
            f_prev = None
            continue

        f_cur = residual(z_cur)

        if not np.isfinite(f_cur):
            z_prev = None
            f_prev = None
            continue

        if f_cur == 0:
            return z_cur

        if f_prev is not None and f_prev * f_cur < 0:
            try:
                return brentq(
                    F, z_prev, z_cur,
                    args=(t, r1, r2, A, mu),
                    xtol=1e-10,
                    rtol=1e-12,
                    maxiter=200
                )
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Root finding failed in [{z_prev}, {z_cur}]."
                ) from exc

        z_prev = z_cur
        f_prev = f_cur

    raise RuntimeError(
        "No valid sign-changing bracket found for z. "
        "Check the geometry, time of flight, and transfer branch."
    )