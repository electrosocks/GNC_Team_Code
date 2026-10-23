"""Physical reference checks, interface boundaries, and end-to-end reuse tests.

Runs with either unittest (standard library) or pytest.
"""

import json
import math
import unittest
from dataclasses import replace

import numpy as np

from maneuver_planner.constants import EARTH
from maneuver_planner.evaluation import evaluate_circular_transfer
from maneuver_planner.gnc_tool import GncTool
from maneuver_planner.hohmann_tool import HohmannTool, hohmann_transfer
from maneuver_planner.models import CartesianState, CentralBody, Impulse, TransferRequest
from maneuver_planner.orbital_utils import circular_state, orbital_elements, orbital_period_s
from maneuver_planner.physics import apply_impulse, execute_candidate, propagate_two_body


class Level0Tests(unittest.TestCase):
    def setUp(self):
        self.r1 = EARTH.radius_m + 400_000
        self.r2 = EARTH.radius_m + 500_000
        self.initial = circular_state(self.r1, EARTH, frame="ECI")
        self.request = TransferRequest(self.initial, EARTH, self.r2)
        self.tool = HohmannTool()

    def run_transfer(self, request):
        candidate = self.tool.propose(request)
        result = execute_candidate(candidate, verification_duration_s=orbital_period_s(
            request.target_radius_m, request.body.mu_m3_s2))
        report = evaluate_circular_transfer(result, request.target_radius_m)
        self.assertTrue(report.passed, report.format_text())
        return result

    def test_initial_circular_state_matches_reference(self):
        np.testing.assert_allclose(self.initial.position_m, [6_778_137.0, 0, 0], rtol=0, atol=0)
        np.testing.assert_allclose(self.initial.velocity_m_s, [0, 7668.558175407055, 0], rtol=0, atol=1e-9)

    def test_one_unforced_orbit_returns_to_start_and_conserves_invariants(self):
        period = orbital_period_s(self.r1, EARTH.mu_m3_s2)
        trajectory = propagate_two_body(self.initial, period, EARTH)
        final = trajectory.state(-1)
        self.assertLess(np.linalg.norm(final.position_m - self.initial.position_m), 1.0)
        self.assertLess(np.linalg.norm(final.velocity_m_s - self.initial.velocity_m_s), 1e-3)
        radii = np.linalg.norm(trajectory.positions_m, axis=1)
        self.assertLess(np.max(np.abs(radii - self.r1)), 1.0)
        energies = np.sum(trajectory.velocities_m_s**2, axis=1)/2 - EARTH.mu_m3_s2/radii
        angular_momenta = np.cross(trajectory.positions_m, trajectory.velocities_m_s)
        self.assertLess(np.max(np.abs(energies - energies[0]))/abs(energies[0]), 1e-9)
        self.assertLess(np.max(np.linalg.norm(angular_momenta-angular_momenta[0], axis=1))
                        / np.linalg.norm(angular_momenta[0]), 1e-9)

    def test_hohmann_matches_independently_calculated_benchmark(self):
        solution = hohmann_transfer(self.r1, self.r2, EARTH.mu_m3_s2)
        self.assertAlmostEqual(solution.departure_delta_v_m_s, 28.025839314408586, delta=1e-9)
        self.assertAlmostEqual(solution.arrival_delta_v_m_s, 27.92341283132191, delta=1e-9)
        self.assertAlmostEqual(solution.total_delta_v_m_s, 55.9492521457305, delta=1e-9)
        self.assertAlmostEqual(solution.time_of_flight_s, 2807.5941199195818, delta=1e-8)

    def test_impulse_preserves_position_and_does_not_mutate_inputs(self):
        before = self.initial.vector.copy()
        command_array = np.array([0, 10.0, 0])
        command = Impulse(0, command_array, "ECI")
        command_array[1] = 99  # Input arrays must have been copied.
        after = apply_impulse(self.initial, command)
        np.testing.assert_array_equal(self.initial.vector, before)
        np.testing.assert_array_equal(after.position_m, self.initial.position_m)
        np.testing.assert_allclose(after.velocity_m_s - self.initial.velocity_m_s, [0, 10, 0], atol=1e-12)
        with self.assertRaises(ValueError):
            after.position_m[0] = 1

    def test_mismatched_frames_and_burn_epochs_are_rejected(self):
        for command in (Impulse(0, [0, 1, 0], "another_frame"), Impulse(1, [0, 1, 0], "ECI")):
            with self.subTest(command=command), self.assertRaises(ValueError):
                apply_impulse(self.initial, command)

    def test_full_level0_transfer_and_burn_events(self):
        result = self.run_transfer(self.request)
        self.assertEqual(len(result.burn_records), 2)
        self.assertGreater(result.burn_records[0].command.delta_v_m_s[1], 0)
        self.assertLess(result.burn_records[1].command.delta_v_m_s[1], 0)
        self.assertEqual(np.count_nonzero(np.diff(result.trajectory.epochs_s) == 0), 2)
        # A missing circularization burn should leave a distinctly elliptical orbit.
        before_second_burn = orbital_elements(result.burn_records[1].before, EARTH.mu_m3_s2)
        self.assertAlmostEqual(before_second_burn.eccentricity, (self.r2-self.r1)/(self.r2+self.r1), delta=1e-9)

    def test_lowering_transfer_uses_retrograde_burns(self):
        initial = circular_state(self.r2, EARTH, frame="ECI")
        result = self.run_transfer(TransferRequest(initial, EARTH, self.r1))
        for record in result.burn_records:
            self.assertLess(np.dot(record.command.delta_v_m_s, record.before.velocity_m_s), 0)
        forward = hohmann_transfer(self.r1, self.r2, EARTH.mu_m3_s2)
        backward = hohmann_transfer(self.r2, self.r1, EARTH.mu_m3_s2)
        self.assertAlmostEqual(forward.total_delta_v_m_s, backward.total_delta_v_m_s, delta=1e-10)

    def test_rotated_inclined_and_retrograde_transfers(self):
        for inclination in (0.7, math.pi/2, 2.6, math.pi):
            with self.subTest(inclination=inclination):
                initial = circular_state(self.r1, EARTH, epoch_s=321.25, inclination_rad=inclination,
                                         raan_rad=0.83, argument_of_latitude_rad=1.17, frame="custom_inertial")
                result = self.run_transfer(TransferRequest(initial, EARTH, self.r2))
                elements = orbital_elements(result.terminal_state, EARTH.mu_m3_s2)
                self.assertAlmostEqual(elements.inclination_rad, inclination, delta=1e-10)

    def test_non_earth_body_requires_no_algorithm_changes(self):
        body = CentralBody("Synthetic test body", mu_m3_s2=8e12, radius_m=1e6)
        initial = circular_state(2e6, body, epoch_s=123, inclination_rad=0.4, frame="test_body_inertial")
        self.run_transfer(TransferRequest(initial, body, 2.7e6))

    def test_gravitational_parameter_scaling_laws(self):
        baseline = hohmann_transfer(self.r1, self.r2, EARTH.mu_m3_s2)
        scaled = hohmann_transfer(self.r1, self.r2, 4*EARTH.mu_m3_s2)
        self.assertAlmostEqual(scaled.total_delta_v_m_s, 2*baseline.total_delta_v_m_s, delta=1e-10)
        self.assertAlmostEqual(scaled.time_of_flight_s, baseline.time_of_flight_s/2, delta=1e-10)

    def test_equal_radii_require_no_maneuver(self):
        rotated = circular_state(self.r1, EARTH, inclination_rad=0.7,
                                 raan_rad=0.83, argument_of_latitude_rad=1.17)
        for initial in (self.initial, rotated):
            with self.subTest(frame=initial.frame):
                result = self.run_transfer(TransferRequest(initial, EARTH, self.r1))
                self.assertEqual(result.candidate.burns, ())
                self.assertEqual(result.candidate.duration_s, 0)
                self.assertEqual(result.candidate.total_delta_v_m_s, 0)

    def test_eccentric_initial_orbit_is_rejected(self):
        eccentric = replace(self.initial, velocity_m_s=self.initial.velocity_m_s * 1.02)
        request = TransferRequest(eccentric, EARTH, self.r2)
        self.assertFalse(self.tool.check_applicability(request).applicable)
        with self.assertRaises(ValueError):
            self.tool.propose(request)

    def test_undefined_classical_angles_are_explicit(self):
        elements = orbital_elements(self.initial, EARTH.mu_m3_s2)
        self.assertIsNone(elements.raan_rad)
        self.assertIsNone(elements.argument_of_periapsis_rad)
        self.assertIsNone(elements.true_anomaly_rad)
        self.assertAlmostEqual(elements.true_longitude_rad, 0)
        inclined = circular_state(self.r1, EARTH, inclination_rad=0.5,
                                  raan_rad=0.7, argument_of_latitude_rad=1.2)
        elements = orbital_elements(inclined, EARTH.mu_m3_s2)
        self.assertAlmostEqual(elements.raan_rad, 0.7)
        self.assertAlmostEqual(elements.argument_of_latitude_rad, 1.2)
        self.assertIsNone(elements.argument_of_periapsis_rad)

    def test_invalid_radii_and_states_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hohmann_transfer(value, self.r2, EARTH.mu_m3_s2)
        with self.assertRaises(ValueError):
            TransferRequest(self.initial, EARTH, EARTH.radius_m - 1)
        with self.assertRaises(ValueError):
            CartesianState(0, [1, 2], [1, 2, 3], "ECI")
        with self.assertRaises(ValueError):
            propagate_two_body(self.initial, -1, EARTH)

    def test_executor_accepts_an_injected_propagator(self):
        calls = []

        def replacement_coast(initial, end, body):
            calls.append((initial.epoch_s, end, body.name))
            return propagate_two_body(initial, end, body, sample_step_s=41)

        candidate = self.tool.propose(self.request)
        result = execute_candidate(candidate, propagator=replacement_coast)
        self.assertEqual(len(calls), 1)
        self.assertTrue(evaluate_circular_transfer(result, self.r2).passed)

    def test_evaluator_catches_an_underperformed_burn(self):
        candidate = self.tool.propose(self.request)
        second = replace(candidate.burns[1], delta_v_m_s=candidate.burns[1].delta_v_m_s*0.95)
        faulty = replace(candidate, burns=(candidate.burns[0], second))
        result = execute_candidate(faulty)
        report = evaluate_circular_transfer(result, self.r2)
        self.assertFalse(report.passed)
        self.assertFalse(next(check for check in report.checks if check.name == "terminal velocity").passed)

    def test_tool_protocol_and_portable_command_export(self):
        self.assertIsInstance(self.tool, GncTool)
        before = self.initial.vector.copy()
        candidate = self.tool.propose(self.request)
        np.testing.assert_array_equal(self.initial.vector, before)
        payload = json.loads(json.dumps(candidate.to_dict(), allow_nan=False))
        self.assertEqual(payload["units"]["velocity"], "m/s")
        self.assertEqual(payload["frame"], "ECI")
        self.assertEqual(len(payload["burns"]), 2)


if __name__ == "__main__":
    unittest.main()
