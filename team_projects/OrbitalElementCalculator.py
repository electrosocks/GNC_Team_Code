"""

    SPACECRAFT ORBIT STATE ANALYSIS

This program is designed so that a user can input either (1) a state and velocity
vector or (2) input a set of orbital elements. If they choose option (1), the
calculator will output the cooresponding orbital elements. If they choose option
(2), it will output the current position and velocity vectors for that orbit.

Authors: Alex Jenz, Manuel Perez
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R_E = 6378.137  # Earth radius in km
mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)


def get_orbital_elements(r, v, mu):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    K = np.array([0.0, 0.0, 1.0])

    energy = v_mag ** 2 / 2.0 - mu / r_mag
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Semi-major Axis
    a = -mu / (2.0 * energy)

    # Eccentricity Vector & Magnitude
    e_vec = ((v_mag ** 2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    # RAAN
    n = np.cross(K, h)
    n_mag = np.linalg.norm(n)
    if n_mag != 0:
        RAAN = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
        if n[1] < 0:
            RAAN = 2 * np.pi - RAAN
    else:
        RAAN = 0.0

    # Argument of Periapsis
    if n_mag != 0 and e > 1e-8:
        omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    # True Anomaly
    if e > 1e-8:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if r[2] < 0:
            nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1.0, 1.0))
        if r[2] < 0:
            nu = 2 * np.pi - nu

    return a, e, i, RAAN, omega, nu


def get_pos_vel(a, e, i, RAAN, omega, nu, mu):
    p = a * (1.0 - e ** 2)
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pqf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])

    v_mag = np.sqrt(mu / p)
    v_pqf = v_mag * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    R3_om = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    Q = R3_RAAN @ R1_i @ R3_om

    r = Q @ r_pqf
    v = Q @ v_pqf
    return r, v


def display_orbit(a, e, i, RAAN, omega, r_current, r_perigee, r_apogee, R_E):
    theta = np.linspace(0, 2 * np.pi, 600)
    orbitXYZ = np.zeros((3, len(theta)))

    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    R3_om = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    Q = R3_RAAN @ R1_i @ R3_om

    for k in range(len(theta)):
        nu_k = theta[k]
        r_mag_k = (a * (1 - e ** 2)) / (1 + e * np.cos(nu_k))
        r_pqf = np.array([r_mag_k * np.cos(nu_k), r_mag_k * np.sin(nu_k), 0])
        orbitXYZ[:, k] = Q @ r_pqf

    perigeePoint = Q @ np.array([r_perigee, 0, 0])
    apogeePoint = Q @ np.array([-r_apogee, 0, 0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Generate Earth sphere mesh
    u = np.linspace(0, 2 * np.pi, 80)
    v_sphere = np.linspace(0, np.pi, 80)
    X_earth = R_E * np.outer(np.cos(u), np.sin(v_sphere))
    Y_earth = R_E * np.outer(np.sin(u), np.sin(v_sphere))
    Z_earth = R_E * np.outer(np.ones(np.size(u)), np.cos(v_sphere))

    ax.plot_surface(X_earth, Y_earth, Z_earth, color='dodgerblue', alpha=0.8, edgecolor='none')

    ax.plot(orbitXYZ[0, :], orbitXYZ[1, :], orbitXYZ[2, :], color='r', linewidth=2, label='Orbit Path')
    ax.scatter(r_current[0], r_current[1], r_current[2], color='y', s=50, edgecolor='k', label='Current Position')
    ax.scatter(perigeePoint[0], perigeePoint[1], perigeePoint[2], color='g', s=50, label='Perigee')
    ax.scatter(apogeePoint[0], apogeePoint[1], apogeePoint[2], color='m', s=50, label='Apogee')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.set_title('Spacecraft Orbit Analysis & 3D Visualization')
    ax.legend(loc='best')
    plt.show()


def main():
    # User Choice Menu
    print("Choose input mode:")
    print("  (1) State Vectors (r, v)")
    print("  (2) Orbital Elements (a, e, i, RAAN, omega, nu)")
    option = int(input("Enter choice (1 or 2): "))

    if option == 1:
        r_input = input("Enter x, y, z of position vector (space-separated, km): ")
        r = np.array([float(x) for x in r_input.split()])
        v_input = input("Enter x, y, z of velocity vector (space-separated, km/s): ")
        v = np.array([float(x) for x in v_input.split()])

        a, e, i, RAAN, omega, nu = get_orbital_elements(r, v, mu)
    else:
        a = float(input("Enter semi-major axis (km): "))
        e = float(input("Enter eccentricity: "))
        i = np.radians(float(input("Enter inclination (deg): ")))
        RAAN = np.radians(float(input("Enter RAAN (deg): ")))
        omega = np.radians(float(input("Enter argument of periapsis (deg): ")))
        nu = np.radians(float(input("Enter true anomaly (deg): ")))

        r, v = get_pos_vel(a, e, i, RAAN, omega, nu, mu)

    # Comprehensive Orbital Analysis
    r_mag = np.linalg.norm(r)
    speed = np.linalg.norm(v)
    height = r_mag - R_E

    perigeeRadius = a * (1 - e)
    apogeeRadius = a * (1 + e)
    perigeeHeight = perigeeRadius - R_E
    apogeeHeight = apogeeRadius - R_E

    periodSeconds = 2 * np.pi * np.sqrt(a ** 3 / mu)
    periodMinutes = periodSeconds / 60.0
    perigeeSpeed = np.sqrt(mu * (2.0 / perigeeRadius - 1.0 / a))
    apogeeSpeed = np.sqrt(mu * (2.0 / apogeeRadius - 1.0 / a))

    if e < 0.01:
        orbitType = 'Nearly Circular'
    elif e < 1:
        orbitType = 'Elliptical'
    elif abs(e - 1) < 0.01:
        orbitType = 'Approximately Parabolic'
    else:
        orbitType = 'Hyperbolic'

    # Print Detailed Report
    print("\n" + "=" * 68)
    print("                     SPACECRAFT ORBIT ANALYSIS")
    print("=" * 68)
    print("\nSPACECRAFT STATE")
    print("-" * 68)
    print(f"Position (X, Y, Z):   {r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f} km")
    print(f"Current Speed:        {speed:.3f} km/s        | Height:          {height:.2f} km")
    print("\nORBITAL ELEMENTS")
    print("-" * 68)
    print(f"Semi-Major Axis:      {a:.2f} km          | Eccentricity:    {e:.5f}")
    print(f"Inclination:          {np.degrees(i):.2f} degrees     | RAAN:            {np.degrees(RAAN):.2f} degrees")
    print(f"Argument of Periapsis: {np.degrees(omega):.2f} degrees    | True Anomaly:    {np.degrees(nu):.2f} degrees")
    print("\nORBIT CHARACTERISTICS")
    print("-" * 68)
    print(f"Orbit Type:           {orbitType:<16} | Orbital Period:  {periodMinutes:.2f} minutes")
    print(f"Perigee Height:       {perigeeHeight:.2f} km          | Apogee Height:   {apogeeHeight:.2f} km")
    print(f"Perigee Speed:        {perigeeSpeed:.3f} km/s        | Apogee Speed:    {apogeeSpeed:.3f} km/s")
    print("=" * 68)

    # Visualize Orbit
    display_orbit(a, e, i, RAAN, omega, r, perigeeRadius, apogeeRadius, R_E)


if __name__ == "__main__":
    main()