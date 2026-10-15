import numpy as np


# =======================================================
#                   Stumpff Functions
# =======================================================

def stump_s(z: float) -> float:
    if z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz ** 3)

    elif z < 0:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (sz ** 3)

    else:
        return 1.0 / 6.0


def stump_c(z: float) -> float:
    if z > 0:
        sz = np.sqrt(z)
        return (1.0 - np.cos(sz)) / z

    elif z < 0:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1.0) / (-z)

    else:
        return 0.5


# =======================================================
#                   Lambert Solver
# =======================================================

def solve_lambert(R1, R2, t, mu, trajectory="pro"):
    """
    Solves Lambert's problem for orbital transfer.

    Inputs:
    - R1: Initial position vector (km)
    - R2: Final position vector (km)
    - t: Time of flight from R1 to R2 (seconds)
    - mu: Gravitational parameter (km^3/s^2)
    - trajectory: "pro" for prograde, "retro" for retrograde

    Outputs:
    - V1: Initial velocity vector (km/s)
    - V2: Final velocity vector (km/s)
    """

    # ---------------------------------------------------
    # Convert inputs to NumPy arrays
    # ---------------------------------------------------

    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    # Basic validation
    if R1.shape != (3,) or R2.shape != (3,):
        raise ValueError("R1 and R2 must each contain exactly 3 components.")

    if t <= 0:
        raise ValueError("Time of flight must be greater than zero.")

    if mu <= 0:
        raise ValueError("Gravitational parameter mu must be greater than zero.")

    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    if r1 == 0 or r2 == 0:
        raise ValueError("Position vectors cannot have zero magnitude.")

    # ---------------------------------------------------
    # Determine transfer angle
    # ---------------------------------------------------

    c12 = np.cross(R1, R2)

    dot_val = np.dot(R1, R2) / (r1 * r2)

    # Prevent tiny floating-point errors from pushing
    # acos outside of its valid range [-1, 1]
    dot_val = np.clip(dot_val, -1.0, 1.0)

    theta = np.arccos(dot_val)

    # Determine prograde or retrograde transfer
    if trajectory == "pro":

        if c12[2] <= 0:
            theta = 2.0 * np.pi - theta

    elif trajectory == "retro":

        if c12[2] >= 0:
            theta = 2.0 * np.pi - theta

    else:
        raise ValueError(
            "trajectory must be either 'pro' or 'retro'"
        )

    # ---------------------------------------------------
    # Calculate A
    # ---------------------------------------------------

    denominator = 1.0 - np.cos(theta)

    if np.isclose(denominator, 0.0):
        raise ValueError(
            "Lambert solution is singular for this transfer geometry."
        )

    A = (
        np.sin(theta)
        * np.sqrt(r1 * r2 / denominator)
    )

    if np.isclose(A, 0.0):
        raise ValueError(
            "Lambert solution is undefined for this transfer geometry."
        )

    # ===================================================
    # Helper Functions
    # ===================================================

    def C(z):
        return stump_c(z)

    def S(z):
        return stump_s(z)

    # ---------------------------------------------------
    # y(z)
    # ---------------------------------------------------

    def y(z):
        c_z = C(z)
        s_z = S(z)

        if c_z <= 0 or not np.isfinite(c_z):
            return np.nan

        return (
            r1
            + r2
            + A * (z * s_z - 1.0) / np.sqrt(c_z)
        )

    # ---------------------------------------------------
    # F(z)
    #
    # We want to solve:
    #
    #       F(z) = 0
    #
    # because that corresponds to the desired
    # time of flight.
    # ---------------------------------------------------

    def F(z):
        c_z = C(z)
        s_z = S(z)
        yz = y(z)

        # Some guesses for z create invalid geometry.
        # Do not attempt sqrt(y) when y is negative.
        if (
            not np.isfinite(c_z)
            or not np.isfinite(s_z)
            or not np.isfinite(yz)
            or c_z <= 0
            or yz < 0
        ):
            return np.nan

        return (
            (yz / c_z) ** 1.5 * s_z
            + A * np.sqrt(yz)
            - np.sqrt(mu) * t
        )

    # ---------------------------------------------------
    # Derivative dF/dz
    # ---------------------------------------------------

    def dFdz(z):
        c_z = C(z)
        s_z = S(z)
        yz = y(z)

        if (
            not np.isfinite(c_z)
            or not np.isfinite(s_z)
            or not np.isfinite(yz)
            or c_z <= 0
            or yz <= 0
        ):
            return np.nan

        # Special equation when z = 0
        if np.isclose(z, 0.0):

            y0 = y(0.0)

            return (
                np.sqrt(2.0) / 40.0 * (y0 ** 1.5)
                + A / 8.0
                * (
                    np.sqrt(y0)
                    + A * np.sqrt(1.0 / (2.0 * y0))
                )
            )

        # General derivative for z != 0
        term1 = (yz / c_z) ** 1.5

        term2 = (
            1.0 / (2.0 * z)
            * (
                c_z
                - (3.0 * s_z) / (2.0 * c_z)
            )
        )

        # FIX:
        # coefficient should be 3, not 2
        term3 = (
            3.0 * (s_z ** 2)
            / (4.0 * c_z)
        )

        # FIX:
        # both pieces belong inside the A / 8 factor
        term4 = (
            A / 8.0
            * (
                (3.0 * s_z / c_z) * np.sqrt(yz)
                + A * np.sqrt(c_z / yz)
            )
        )

        return (
            term1 * (term2 + term3)
            + term4
        )

    # ===================================================
    # Find Initial z
    # ===================================================

    z = -100.0

    while True:

        fz = F(z)

        # We only accept a starting point if F(z)
        # is a real, finite value and has reached
        # the positive side.
        if np.isfinite(fz) and fz >= 0:
            break

        z += 0.1

        if z > 1000.0:
            raise RuntimeError(
                "Could not find a valid starting value for z."
            )

    # ===================================================
    # Newton's Method
    # ===================================================

    tol = 1.0e-8
    nmax = 5000

    converged = False

    for _ in range(nmax):

        fz = F(z)
        derivative = dFdz(z)

        if not np.isfinite(fz):
            raise RuntimeError(
                "Lambert solver encountered an invalid F(z)."
            )

        if not np.isfinite(derivative):
            raise RuntimeError(
                "Lambert solver encountered an invalid derivative."
            )

        if np.isclose(derivative, 0.0):
            raise RuntimeError(
                "Newton's method encountered a near-zero derivative."
            )

        ratio = fz / derivative

        z -= ratio

        if abs(ratio) <= tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Lambert solver failed to converge after {nmax} iterations."
        )

    # ===================================================
    # Lagrange Coefficients
    # ===================================================

    yz = y(z)

    if not np.isfinite(yz) or yz <= 0:
        raise RuntimeError(
            "Lambert solver produced an invalid final y(z)."
        )

    f = 1.0 - yz / r1

    g = A * np.sqrt(yz / mu)

    gdot = 1.0 - yz / r2

    if np.isclose(g, 0.0):
        raise RuntimeError(
            "Lambert solver produced g = 0."
        )

    # ===================================================
    # Velocity Vectors
    # ===================================================

    V1 = (R2 - f * R1) / g

    V2 = (gdot * R2 - R1) / g

    return V1, V2
    return V1, V2