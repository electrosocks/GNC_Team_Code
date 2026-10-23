"""Independent checks of the executed trajectory, not the tool's reported success.

The Hohmann reference below is derived directly from vis-viva; this module does
not import or call the Hohmann tool. Conservation is checked within coast segments,
never across an intentional impulse. These checks validate the ideal model only.
"""

from dataclasses import asdict, dataclass
from math import atan2, isclose, isfinite, pi, sqrt

import numpy as np

from .models import ExecutionResult, positive
from .orbital_utils import orbital_elements


@dataclass(frozen=True)
class ValidationTolerances:
    position_m: float = 1.0
    velocity_m_s: float = 1e-3
    altitude_m: float = 1.0
    eccentricity: float = 1e-7
    plane_angle_rad: float = 1e-8
    conservation_relative: float = 1e-9
    delta_v_m_s: float = 1e-6
    duration_s: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(value, name)


@dataclass(frozen=True)
class Check:
    name: str
    measured_error: float
    allowed_error: float
    unit: str

    @property
    def passed(self) -> bool:
        return isfinite(self.measured_error) and self.measured_error <= self.allowed_error


@dataclass(frozen=True)
class EvaluationReport:
    checks: tuple[Check, ...]
    metrics: dict[str, float]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "metrics": self.metrics,
                "checks": [{**asdict(check), "passed": check.passed} for check in self.checks]}

    def format_text(self) -> str:
        lines = [f"Circular transfer: {'PASS' if self.passed else 'FAIL'}"]
        for key in ("departure_delta_v_m_s", "arrival_delta_v_m_s", "total_delta_v_m_s",
                    "transfer_duration_s", "terminal_altitude_m", "terminal_eccentricity"):
            lines.append(f"  {key}: {self.metrics[key]:.10g}")
        lines.append("Checks (measured error <= allowed error):")
        for check in self.checks:
            lines.append(f"  {'PASS' if check.passed else 'FAIL'}  {check.name}: "
                         f"{check.measured_error:.3e} <= {check.allowed_error:.3e} {check.unit}")
        return "\n".join(lines)


def evaluate_circular_transfer(result: ExecutionResult, target_radius_m: float, *,
                               tolerances: ValidationTolerances | None = None) -> EvaluationReport:
    """Validate the manually specified Hohmann scenario, with generic body/plane.

    This is a scenario evaluator, not a general evaluator for arbitrary GNC tools.
    With no verification coast, altitude is checked only at the terminal state.
    """
    positive(target_radius_m, "target_radius_m")
    tol = tolerances or ValidationTolerances()
    candidate = result.candidate
    initial, terminal = candidate.initial_state, result.terminal_state
    mu, body_radius = candidate.body.mu_m3_s2, candidate.body.radius_m
    r1 = float(np.linalg.norm(initial.position_m))
    r2 = target_radius_m
    rhat = initial.position_m / r1
    h0 = np.cross(initial.position_m, initial.velocity_m_s)
    tangent = np.cross(h0 / np.linalg.norm(h0), rhat)
    a_transfer = (r1 + r2) / 2
    if isclose(r1, r2, rel_tol=8*np.finfo(float).eps):
        expected_time, expected_dv = 0.0, 0.0
        expected_r, expected_v = initial.position_m, initial.velocity_m_s
    else:
        expected_time = pi * sqrt(a_transfer**3 / mu)
        expected_dv = (abs(sqrt(mu * (2/r1 - 1/a_transfer)) - sqrt(mu/r1))
                       + abs(sqrt(mu/r2) - sqrt(mu * (2/r2 - 1/a_transfer))))
        expected_r = -r2 * rhat
        expected_v = -sqrt(mu/r2) * tangent

    elements = orbital_elements(terminal, mu)
    apsis_error = max(abs(elements.periapsis_radius_m - r2),
                     abs(elements.apoapsis_radius_m - r2) if elements.apoapsis_radius_m is not None
                     else float("inf"))
    h_final = np.cross(terminal.position_m, terminal.velocity_m_s)
    normal0, normalf = h0 / np.linalg.norm(h0), h_final / np.linalg.norm(h_final)
    plane_error = atan2(float(np.linalg.norm(np.cross(normal0, normalf))), float(np.dot(normal0, normalf)))
    energy_drift = momentum_drift = 0.0
    for coast in result.coast_segments:
        r, v = coast.positions_m, coast.velocities_m_s
        energies = np.einsum("ij,ij->i", v, v) / 2 - mu / np.linalg.norm(r, axis=1)
        momenta = np.cross(r, v)
        energy_drift = max(energy_drift, float(np.max(np.abs(energies - energies[0])) / abs(energies[0])))
        momentum_drift = max(momentum_drift, float(np.max(np.linalg.norm(momenta - momenta[0], axis=1))
                                                  / np.linalg.norm(momenta[0])))

    applied_dv = [float(np.linalg.norm(record.after.velocity_m_s - record.before.velocity_m_s))
                  for record in result.burn_records]
    burn_position_error = max((float(np.linalg.norm(record.after.position_m - record.before.position_m))
                               for record in result.burn_records), default=0.0)
    burn_velocity_error = max((float(np.linalg.norm(record.after.velocity_m_s - record.before.velocity_m_s
                                                    - record.command.delta_v_m_s))
                               for record in result.burn_records), default=0.0)
    radii = [float(np.linalg.norm(terminal.position_m))]
    if result.verification_trajectory is not None:
        radii.extend(np.linalg.norm(result.verification_trajectory.positions_m, axis=1))
    max_altitude_error = float(np.max(np.abs(np.asarray(radii) - r2)))
    duration = terminal.epoch_s - initial.epoch_s

    metrics = {
        "departure_delta_v_m_s": applied_dv[0] if applied_dv else 0.0,
        "arrival_delta_v_m_s": applied_dv[-1] if applied_dv else 0.0,
        "total_delta_v_m_s": sum(applied_dv),
        "expected_total_delta_v_m_s": expected_dv,
        "transfer_duration_s": duration,
        "expected_transfer_duration_s": expected_time,
        "terminal_altitude_m": float(np.linalg.norm(terminal.position_m)) - body_radius,
        "target_altitude_m": r2 - body_radius,
        "terminal_semimajor_axis_m": elements.semimajor_axis_m,
        "terminal_eccentricity": elements.eccentricity,
        "terminal_inclination_rad": elements.inclination_rad,
        "max_altitude_error_m": max_altitude_error,
        "terminal_orbit_apsis_error_m": apsis_error,
        "max_energy_relative_drift": energy_drift,
        "max_angular_momentum_relative_drift": momentum_drift,
    }
    checks = (
        Check("terminal position", float(np.linalg.norm(terminal.position_m - expected_r)), tol.position_m, "m"),
        Check("terminal velocity", float(np.linalg.norm(terminal.velocity_m_s - expected_v)), tol.velocity_m_s, "m/s"),
        Check("target semimajor axis", abs(elements.semimajor_axis_m - r2), tol.altitude_m, "m"),
        Check("sampled altitude during verification", max_altitude_error, tol.altitude_m, "m"),
        Check("terminal orbit apsides", apsis_error, tol.altitude_m, "m"),
        Check("terminal eccentricity", elements.eccentricity, tol.eccentricity, ""),
        Check("orbital plane preservation", plane_error, tol.plane_angle_rad, "rad"),
        Check("coast energy conservation", energy_drift, tol.conservation_relative, "relative"),
        Check("coast angular momentum conservation", momentum_drift, tol.conservation_relative, "relative"),
        Check("total applied delta-v", abs(sum(applied_dv) - expected_dv), tol.delta_v_m_s, "m/s"),
        Check("transfer duration", abs(duration - expected_time), tol.duration_s, "s"),
        Check("position continuity at burns", burn_position_error, tol.position_m, "m"),
        Check("applied burn vectors", burn_velocity_error, tol.velocity_m_s, "m/s"),
    )
    return EvaluationReport(checks, metrics)


def evaluate_target_orbit(result: ExecutionResult, target_radius_m: float, *,
                          expected_position_m=None, expected_velocity_m_s=None,
                          element_calculator=orbital_elements,
                          tolerances: ValidationTolerances | None = None) -> EvaluationReport:
    """Common orbital-goal checks for different manually selected formulations.

    Unlike evaluate_circular_transfer, this does not require Hohmann's endpoint,
    time, or cost. Optional terminal vectors check each method's stated geometry.
    The existing Hohmann-specific evaluator remains available for that baseline.
    """
    positive(target_radius_m, "target_radius_m")
    tol = tolerances or ValidationTolerances()
    candidate, terminal = result.candidate, result.terminal_state
    initial, body = candidate.initial_state, candidate.body
    elements = element_calculator(terminal, body.mu_m3_s2)
    h0 = np.cross(initial.position_m, initial.velocity_m_s)
    hf = np.cross(terminal.position_m, terminal.velocity_m_s)
    n0, nf = h0/np.linalg.norm(h0), hf/np.linalg.norm(hf)
    plane_error = atan2(float(np.linalg.norm(np.cross(n0, nf))), float(np.dot(n0, nf)))
    radii = np.array([np.linalg.norm(terminal.position_m)])
    if result.verification_trajectory is not None:
        radii = np.concatenate((radii, np.linalg.norm(result.verification_trajectory.positions_m, axis=1)))
    altitude_error = float(np.max(abs(radii-target_radius_m)))
    apsis_error = max(abs(elements.periapsis_radius_m-target_radius_m),
                     abs(elements.apoapsis_radius_m-target_radius_m)
                     if elements.apoapsis_radius_m is not None else float("inf"))
    energy_drift = momentum_drift = 0.0
    for coast in result.coast_segments:
        rr, vv = coast.positions_m, coast.velocities_m_s
        energy = np.einsum("ij,ij->i", vv, vv)/2-body.mu_m3_s2/np.linalg.norm(rr, axis=1)
        momentum = np.cross(rr, vv)
        # The scale floor only avoids division by zero for a near-parabolic coast.
        scale = max(abs(energy[0]), np.finfo(float).eps*body.mu_m3_s2/np.linalg.norm(rr[0]))
        energy_drift = max(energy_drift, float(np.max(abs(energy-energy[0]))/scale))
        momentum_drift = max(momentum_drift, float(np.max(np.linalg.norm(momentum-momentum[0], axis=1))
                                                  /np.linalg.norm(momentum[0])))
    applied = [float(np.linalg.norm(record.after.velocity_m_s-record.before.velocity_m_s))
               for record in result.burn_records]
    position_jump = max((float(np.linalg.norm(record.after.position_m-record.before.position_m))
                         for record in result.burn_records), default=0.0)
    command_error = max((float(np.linalg.norm(record.after.velocity_m_s-record.before.velocity_m_s
                                             -record.command.delta_v_m_s))
                         for record in result.burn_records), default=0.0)
    min_radius = float(np.min(np.linalg.norm(result.trajectory.positions_m, axis=1)))
    checks = [
        Check("target semimajor axis", abs(elements.semimajor_axis_m-target_radius_m), tol.altitude_m, "m"),
        Check("terminal orbit apsides", apsis_error, tol.altitude_m, "m"),
        Check("sampled altitude during verification", altitude_error, tol.altitude_m, "m"),
        Check("terminal eccentricity", elements.eccentricity, tol.eccentricity, ""),
        Check("orbital plane preservation", plane_error, tol.plane_angle_rad, "rad"),
        Check("coast energy conservation", energy_drift, tol.conservation_relative, "relative"),
        Check("coast angular momentum conservation", momentum_drift, tol.conservation_relative, "relative"),
        Check("total applied versus commanded delta-v", abs(sum(applied)-candidate.total_delta_v_m_s), tol.delta_v_m_s, "m/s"),
        Check("scheduled mission end", abs(terminal.epoch_s-candidate.end_epoch_s), tol.duration_s, "s"),
        Check("position continuity at burns", position_jump, tol.position_m, "m"),
        Check("applied burn vectors", command_error, tol.velocity_m_s, "m/s"),
        Check("no sampled central-body intersection", max(0.0, body.radius_m-min_radius), 0.0, "m"),
    ]
    if expected_position_m is not None:
        checks.append(Check("method arrival position", float(np.linalg.norm(terminal.position_m-expected_position_m)),
                            tol.position_m, "m"))
    if expected_velocity_m_s is not None:
        checks.append(Check("method arrival velocity", float(np.linalg.norm(terminal.velocity_m_s-expected_velocity_m_s)),
                            tol.velocity_m_s, "m/s"))
    metrics = {
        "departure_delta_v_m_s": applied[0] if applied else 0.0,
        "arrival_delta_v_m_s": applied[-1] if applied else 0.0,
        "total_delta_v_m_s": sum(applied),
        "transfer_duration_s": terminal.epoch_s-initial.epoch_s,
        "terminal_altitude_m": float(np.linalg.norm(terminal.position_m))-body.radius_m,
        "terminal_semimajor_axis_m": elements.semimajor_axis_m,
        "terminal_eccentricity": elements.eccentricity,
        "terminal_inclination_rad": elements.inclination_rad,
        "max_altitude_error_m": altitude_error,
        "terminal_orbit_apsis_error_m": apsis_error,
        "max_energy_relative_drift": energy_drift,
        "max_angular_momentum_relative_drift": momentum_drift,
        "minimum_sampled_altitude_m": min_radius-body.radius_m,
    }
    return EvaluationReport(tuple(checks), metrics)
