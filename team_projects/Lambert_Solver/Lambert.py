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
    - trajectory: 'pro' for prograde, 'retro' for retrograde

    Outputs:
    - V1: Initial velocity vector (km/s)
    - V2: Final velocity vector (km/s)
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    c12 = np.cross(R1, R2)

    # Angle between R1 and R2
    dot_val = np.dot(R1, R2) / (r1 * r2)
    # Clip to avoid numerical issues with acos outside [-1, 1]
    theta = np.arccos(np.clip(dot_val, -1.0, 1.0))

    # Determine whether the orbit is prograde or retrograde
    if trajectory == "pro":
        if c12[2] <= 0:
            theta = 2 * np.pi - theta
    elif trajectory == "retro":
        if c12[2] >= 0:
            theta = 2 * np.pi - theta
    else:
        trajectory = "pro"
        print("\n ** Prograde trajectory assumed.\n")

    # Equation 5.35:
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1.0 - np.cos(theta)))

    # Helper functions using closures for r1, r2, A, mu, t
    def C(z):
        return stump_c(z)

    def S(z):
        return stump_s(z)

    def y(z):
        c_z = C(z)
        s_z = S(z)
        return r1 + r2 + A * (z * s_z - 1.0) / np.sqrt(c_z)

    def F(z):
        c_z = C(z)
        s_z = S(z)
        yz = y(z)
        return (yz / c_z) ** 1.5 * s_z + A * np.sqrt(yz) - np.sqrt(mu) * t

    def dFdz(z):
        c_z = C(z)
        s_z = S(z)
        yz = y(z)
        if z == 0:
            y0 = y(0.0)
            return (
                np.sqrt(2.0) / 40.0 * (y0 ** 1.5)
                + A / 8.0 * (np.sqrt(y0) + A * np.sqrt(1.0 / (2.0 * y0)))
            )
        else:
            term1 = (yz / c_z) ** 1.5
            term2 = (1.0 / (2.0 * z)) * (c_z - (3.0 * s_z) / (2.0 * c_z))
            term3 = (2.0 * (s_z ** 2)) / (4.0 * c_z)
            term4 = (A / 8.0) * ((3.0 * s_z / c_z) * np.sqrt(yz))
            term5 = A * np.sqrt(c_z / yz)
            return term1 * (term2 + term3) + term4 + term5

    # Determine starting value for z where F(z, t) changes sign
    z = -100.0
    while F(z) < 0:
        z += 0.1

    # Newton's method iteration
    tol = 1.0e-8
    nmax = 5000
    ratio = 1.0
    n = 0

    while (abs(ratio) > tol) and (n <= nmax):
        n += 1
        ratio = F(z) / dFdz(z)
        z -= ratio

    if n >= nmax:
        print(f"\n\n **Number of iterations exceeds  {nmax} \n\n")

    # Lagrange coefficients
    yz = y(z)
    f = 1.0 - yz / r1
    g = A * np.sqrt(yz / mu)
    gdot = 1.0 - yz / r2

    # Velocity vectors (Equations 5.28 and 5.29)
    V1 = (1.0 / g) * (R2 - f * R1)
    V2 = (1.0 / g) * (gdot * R2 - R1)

    return V1, V2