"""CW mathematical checks, frame conversion, and raw/corrected Level 0 behavior."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from maneuver_planner.constants import EARTH
from maneuver_planner.cw_equations import (cw_derivative, cw_stm, cw_system_matrix,
                                           mean_motion, propagate_cw, target_cw)
from maneuver_planner.cw_transfer_tool import (CWLevel0Tool, circular_reference,
                                              correct_cw_candidate, hill_basis,
                                              relative_state_hill)
from maneuver_planner.evaluation import evaluate_target_orbit
from maneuver_planner.models import CartesianState, CentralBody, TransferRequest
from maneuver_planner.orbital_utils import circular_state, orbital_period_s
from maneuver_planner.physics import execute_candidate
from maneuver_planner.comparison import Level0Tool


class CWTransferTests(unittest.TestCase):
    def setUp(self):
        self.request = TransferRequest(circular_state(EARTH.radius_m+400000, EARTH, frame="ECI"),
                                       EARTH, EARTH.radius_m+500000)

    def test_cw_stm_agrees_with_independent_matrix_exponential(self):
        for n in (0.001, 0.003):
            for time in (0, 1e-6, 1200, 5000, -1200):
                with self.subTest(n=n, time=time):
                    np.testing.assert_allclose(cw_stm(time, n), expm(cw_system_matrix(n)*time), rtol=2e-11, atol=1e-9)

    def test_cw_targeting_matches_independent_ode_integration(self):
        initial = np.array([30., -1000., 50., .02, -.1, .01])
        n = mean_motion(EARTH.mu_m3_s2, self.request.target_radius_m)
        goal, velocity = [4., -3., 1.], [.01, -.02, .03]
        plan = target_cw(initial, goal, 1200, n, target_velocity_m_s=velocity)
        def derivative(t, state):
            x, y, z, vx, vy, vz = state
            return [vx, vy, vz, 3*n*n*x+2*n*vy, -2*n*vx, -n*n*z]
        final = solve_ivp(derivative, (0, 1200), plan.state_after_departure,
                          method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        final[3:] += plan.arrival_delta_v_m_s
        np.testing.assert_allclose(final, [*goal, *velocity], rtol=0, atol=1e-8)

    def test_frame_conversion_includes_rotation_velocity(self):
        reference = circular_reference(self.request, 0)
        n = mean_motion(EARTH.mu_m3_s2, self.request.target_radius_m)
        relative = relative_state_hill(self.request.initial_state, reference)
        dr = np.linalg.norm(self.request.initial_state.position_m)-self.request.target_radius_m
        dv = np.linalg.norm(self.request.initial_state.velocity_m_s)-np.linalg.norm(reference.velocity_m_s)
        np.testing.assert_allclose(relative, [dr, 0, 0, 0, dv-n*dr, 0], atol=1e-9)
        np.testing.assert_allclose(relative_state_hill(reference, reference), np.zeros(6), atol=0)
        np.testing.assert_allclose(hill_basis(reference).T@hill_basis(reference), np.eye(3), atol=1e-14)

    def test_raw_cw_is_executed_and_honestly_fails_the_strict_orbit_tolerance(self):
        tool = Level0Tool(3)
        candidate = tool.propose(self.request)
        result = execute_candidate(candidate)
        rr, vv = tool.terminal_reference(self.request)
        report = evaluate_target_orbit(result, self.request.target_radius_m,
                                       expected_position_m=rr, expected_velocity_m_s=vv)
        self.assertFalse(report.passed)
        self.assertGreater(np.linalg.norm(result.terminal_state.position_m-rr), 9000)
        self.assertLess(CWLevel0Tool().diagnostics(self.request)["cw_predicted_terminal_position_error_m"], 1e-6)

    def test_corrected_cw_passes_the_full_two_body_orbit_check(self):
        tool = CWLevel0Tool()
        seed = tool.propose(self.request)
        before = self.request.initial_state.vector.copy()
        corrected, trace = correct_cw_candidate(self.request, seed)
        result = execute_candidate(corrected, verification_duration_s=orbital_period_s(
            self.request.target_radius_m, EARTH.mu_m3_s2))
        target = circular_reference(self.request, corrected.duration_s)
        report = evaluate_target_orbit(result, self.request.target_radius_m,
                                       expected_position_m=target.position_m, expected_velocity_m_s=target.velocity_m_s)
        self.assertTrue(report.passed, report.format_text())
        self.assertTrue(trace["applied"])
        self.assertGreater(trace["iterations"], 0)
        self.assertLess(trace["position_errors_m"][-1], 1e-5)
        np.testing.assert_array_equal(self.request.initial_state.vector, before)
        self.assertEqual(seed.tool_name, "cw_linear_two_impulse")
        self.assertEqual(corrected.tool_name, "cw_with_two_body_correction")

    def test_cw_burns_are_transformed_at_their_own_epochs(self):
        tool = CWLevel0Tool(1200)
        plan = tool.linear_solution(self.request)
        candidate = tool.propose(self.request)
        initial_basis = hill_basis(circular_reference(self.request, 0))
        final_basis = hill_basis(circular_reference(self.request, 1200))
        np.testing.assert_allclose(candidate.burns[0].delta_v_m_s, initial_basis@plan.departure_delta_v_m_s, atol=1e-12)
        np.testing.assert_allclose(candidate.burns[1].delta_v_m_s, final_basis@plan.arrival_delta_v_m_s, atol=1e-12)
        self.assertGreater(np.linalg.norm(candidate.burns[1].delta_v_m_s-initial_basis@plan.arrival_delta_v_m_s), 1)

    def test_cw_correction_works_for_rotated_planes_lowering_and_another_body(self):
        body = CentralBody("Synthetic", 8e12, 1e6)
        cases = [
            TransferRequest(circular_state(EARTH.radius_m+400000, EARTH, inclination_rad=math.pi/2,
                                           raan_rad=.7, argument_of_latitude_rad=.8, epoch_s=321), EARTH, EARTH.radius_m+500000),
            TransferRequest(circular_state(EARTH.radius_m+500000, EARTH, inclination_rad=2.6,
                                           raan_rad=.2, argument_of_latitude_rad=.4), EARTH, EARTH.radius_m+400000),
            TransferRequest(circular_state(2e6, body, epoch_s=100, inclination_rad=.9), body, 2.03e6),
        ]
        for request in cases:
            with self.subTest(body=request.body.name, radius=request.target_radius_m):
                tool = CWLevel0Tool()
                candidate, _ = correct_cw_candidate(request, tool.propose(request))
                result = execute_candidate(candidate)
                target = circular_reference(request, candidate.duration_s)
                report = evaluate_target_orbit(result, request.target_radius_m,
                                               expected_position_m=target.position_m, expected_velocity_m_s=target.velocity_m_s)
                self.assertTrue(report.passed, report.format_text())

    def test_correction_does_not_call_the_hohmann_maneuver(self):
        # Patching the Hohmann proposal proves the common applicability check does
        # not obtain its burns from the baseline guidance implementation.
        with patch("maneuver_planner.hohmann_tool.HohmannTool.propose", side_effect=AssertionError("No fallback")):
            seed = CWLevel0Tool().propose(self.request)
            result, trace = correct_cw_candidate(self.request, seed)
        self.assertEqual(result.tool_name, "cw_with_two_body_correction")
        self.assertGreater(trace["iterations"], 0)

    def test_equal_orbits_need_no_cw_burn_or_correction(self):
        request = TransferRequest(self.request.initial_state, EARTH, EARTH.radius_m+400000)
        seed = CWLevel0Tool().propose(request)
        corrected, trace = correct_cw_candidate(request, seed)
        self.assertEqual(corrected.burns, ())
        self.assertFalse(trace["applied"])

    def test_singular_times_and_nonconvergence_are_reported(self):
        n = mean_motion(EARTH.mu_m3_s2, self.request.target_radius_m)
        with self.assertRaises(ValueError):
            CWLevel0Tool(np.pi/n).propose(self.request)
        with self.assertRaises(RuntimeError):
            correct_cw_candidate(self.request, CWLevel0Tool().propose(self.request), max_iterations=1)

    def test_method_3_cli_switches_between_corrected_and_raw_modes(self):
        root = Path(__file__).resolve().parents[1]
        for corrected in (True, False):
            with self.subTest(corrected=corrected), tempfile.TemporaryDirectory() as output:
                command = [sys.executable, str(root/"run_level0_comparison.py"), "--method-id", "3",
                           "--output-dir", output, "--no-plots"]
                if not corrected:
                    command.append("--no-cw-correction")
                process = subprocess.run(command, cwd=root, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0 if corrected else 1, process.stdout+process.stderr)
                report = json.loads((Path(output)/"method_3/summary.json").read_text())
                self.assertEqual(report["passed"], corrected)
                self.assertEqual(report["cw_diagnostics"]["nonlinear_correction"]["enabled"], corrected)


if __name__ == "__main__":
    unittest.main()
