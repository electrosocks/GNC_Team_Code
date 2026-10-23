"""Circular initialization and Cartesian-to-orbital-element diagnostics."""

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin, sqrt

import numpy as np

from .models import CartesianState, CentralBody, positive


def circular_state(radius_m: float, body: CentralBody, *, epoch_s: float = 0.0,
                   inclination_rad: float = 0.0, raan_rad: float = 0.0,
                   argument_of_latitude_rad: float = 0.0,
                   frame: str = "inertial") -> CartesianState:
    """Initialize a circular orbit at any orientation, including retrograde.

    radius_m is measured from the body center. Add body.radius_m to an altitude.
    Angles are radians; the position and velocity are returned in the named frame.
    """
    positive(radius_m, "radius_m")
    if radius_m <= body.radius_m:
        raise ValueError("The orbit must be outside the central body.")
    angles = (inclination_rad, raan_rad, argument_of_latitude_rad)
    if not all(isfinite(value) for value in angles) or not 0 <= inclination_rad <= pi:
        raise ValueError("Angles must be finite and inclination must be in [0, pi].")
    c, s = cos(raan_rad), sin(raan_rad)
    p = np.array([c, s, 0.0])
    q = np.array([-s * cos(inclination_rad), c * cos(inclination_rad), sin(inclination_rad)])
    u = argument_of_latitude_rad
    return CartesianState(
        epoch_s, radius_m * (cos(u) * p + sin(u) * q),
        sqrt(body.mu_m3_s2 / radius_m) * (-sin(u) * p + cos(u) * q), frame,
    )


def orbital_period_s(radius_or_semimajor_axis_m: float, mu_m3_s2: float) -> float:
    """Kepler period for an elliptic orbit (use radius for a circular orbit)."""
    positive(radius_or_semimajor_axis_m, "semimajor axis")
    positive(mu_m3_s2, "mu_m3_s2")
    return 2 * pi * sqrt(radius_or_semimajor_axis_m**3 / mu_m3_s2)


@dataclass(frozen=True)
class OrbitalElements:
    semimajor_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float | None
    argument_of_periapsis_rad: float | None
    true_anomaly_rad: float | None
    argument_of_latitude_rad: float | None
    true_longitude_rad: float | None
    periapsis_radius_m: float
    apoapsis_radius_m: float | None
    specific_energy_m2_s2: float


def orbital_elements(state: CartesianState, mu_m3_s2: float, *,
                     singularity_tolerance: float = 1e-10) -> OrbitalElements:
    """Compute elements without inventing angles for singular circular/equatorial cases.

    None marks an undefined classical angle. Argument of latitude is provided for
    circular inclined orbits; geometric atan2(y, x) longitude for equatorial ones.
    A radial trajectory has no unique orbital plane and is rejected.
    """
    positive(mu_m3_s2, "mu_m3_s2")
    positive(singularity_tolerance, "singularity_tolerance")
    r, v = state.position_m, state.velocity_m_s
    radius = np.linalg.norm(r)
    h = np.cross(r, v)
    hmag = np.linalg.norm(h)
    if hmag == 0:
        raise ValueError("A radial trajectory has no defined orbital plane.")
    hhat = h / hmag
    node = np.cross([0.0, 0.0, 1.0], h)
    nmag = np.linalg.norm(node)
    evec = np.cross(v, h) / mu_m3_s2 - r / radius
    e = float(np.linalg.norm(evec))
    energy = float(np.dot(v, v) / 2 - mu_m3_s2 / radius)
    a = -mu_m3_s2 / (2 * energy) if energy != 0 else float("inf")
    inclined = nmag / hmag > singularity_tolerance
    eccentric = e > singularity_tolerance

    def angle(first, second):
        first = first / np.linalg.norm(first)
        second = second / np.linalg.norm(second)
        return float(atan2(np.dot(hhat, np.cross(first, second)), np.dot(first, second)) % (2*pi))

    return OrbitalElements(
        semimajor_axis_m=float(a), eccentricity=e,
        inclination_rad=atan2(float(np.linalg.norm(h[:2])), float(h[2])),
        raan_rad=atan2(float(node[1]), float(node[0])) % (2*pi) if inclined else None,
        argument_of_periapsis_rad=angle(node, evec) if inclined and eccentric else None,
        true_anomaly_rad=angle(evec, r) if eccentric else None,
        argument_of_latitude_rad=angle(node, r) if inclined and not eccentric else None,
        true_longitude_rad=atan2(float(r[1]), float(r[0])) % (2*pi) if not inclined else None,
        periapsis_radius_m=float(hmag**2 / mu_m3_s2 / (1 + e)),
        apoapsis_radius_m=float(a * (1 + e)) if e < 1 else None,
        specific_energy_m2_s2=energy,
    )
