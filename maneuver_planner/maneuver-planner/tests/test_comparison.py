"""Capability integration, reference equations, and current-run output regression tests."""

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import run_level0_comparison as runner
from maneuver_planner.bielliptic_tool import bielliptic_transfer
from maneuver_planner.comparison import METHODS, ComparisonSettings, Level0Tool
from maneuver_planner.constants import EARTH
from maneuver_planner.cw_transfer_tool import correct_cw_candidate
from maneuver_planner.evaluation import evaluate_target_orbit
from maneuver_planner.hohmann_tool import hohmann_transfer
from maneuver_planner.models import CentralBody, TransferRequest
from maneuver_planner.orbital_utils import circular_state, orbital_period_s
from maneuver_planner.physics import execute_candidate
from verification.reference_transfers import bielliptic_reference, hohmann_reference


class CapabilityTests(unittest.TestCase):
    def test_four_methods_reach_their_declared_endpoints(self):
        self.assertEqual([(m.method_id, m.name) for m in METHODS],
                         [(0, 'Hohmann'), (1, 'Bielliptic'), (2, 'Lambert'), (3, 'CW targeting')])
        for inclination, lower in ((0, False), (math.pi/2, False), (2.6, True)):
            r1, r2 = (EARTH.radius_m+500000, EARTH.radius_m+400000) if lower else (EARTH.radius_m+400000, EARTH.radius_m+500000)
            request = TransferRequest(circular_state(r1, EARTH, inclination_rad=inclination,
                                       raan_rad=.7, argument_of_latitude_rad=.8, epoch_s=123), EARTH, r2)
            for method in range(4):
                with self.subTest(inclination=inclination, lower=lower, method=method):
                    tool = Level0Tool(method, ComparisonSettings(EARTH.radius_m+2e6))
                    candidate = tool.propose(request)
                    if method == 3:
                        candidate, _ = correct_cw_candidate(request, candidate)
                    result = execute_candidate(candidate, verification_duration_s=orbital_period_s(r2, EARTH.mu_m3_s2))
                    rr, vv = tool.terminal_reference(request)
                    report = evaluate_target_orbit(result, r2, expected_position_m=rr, expected_velocity_m_s=vv)
                    self.assertTrue(report.passed, report.format_text())

    def test_bielliptic_matches_other_formulation_and_another_body(self):
        for r1, r2, rb, mu in ((6778137, 6878137, 8378137, EARTH.mu_m3_s2),
                               (6878137, 6778137, 8378137, EARTH.mu_m3_s2), (2e6, 2.7e6, 5e6, 8e12)):
            plan = bielliptic_transfer(r1, r2, rb, mu)
            dv, time = bielliptic_reference(r1, r2, rb, mu)
            self.assertAlmostEqual(plan.total_delta_v_m_s, dv, delta=1e-9)
            self.assertAlmostEqual(plan.time_of_flight_s, time, delta=1e-8)
            h = hohmann_transfer(r1, r2, mu)
            np.testing.assert_allclose([h.departure_delta_v_m_s, h.arrival_delta_v_m_s],
                                       hohmann_reference(r1, r2, mu), atol=1e-10, rtol=0)
        self.assertEqual(bielliptic_transfer(7e6, 7e6, 8e6, EARTH.mu_m3_s2).delta_v_m_s, ())
        with self.assertRaises(ValueError):
            bielliptic_transfer(7e6, 8e6, 7.5e6, EARTH.mu_m3_s2)

    def test_cw_and_lambert_match_when_given_identical_endpoint_and_time(self):
        request = TransferRequest(circular_state(EARTH.radius_m+400000, EARTH), EARTH, EARTH.radius_m+500000)
        cw = Level0Tool(3)
        candidate, _ = correct_cw_candidate(request, cw.propose(request))
        angle = cw.arrival_angle_deg(request, candidate)
        lambert = Level0Tool(2, ComparisonSettings(lambert_angle_deg=angle, lambert_tof_s=candidate.duration_s)).propose(request)
        for a, b in zip(candidate.burns, lambert.burns):
            np.testing.assert_allclose(a.delta_v_m_s, b.delta_v_m_s, atol=1e-6, rtol=0)


class OutputTests(unittest.TestCase):
    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main(args)

    def test_changed_parameters_replace_data_and_plot_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = ['--method-id', '2', '--output-dir', directory]
            self.assertEqual(self.run_cli(args), 0)
            first = json.loads((out/'comparison.json').read_text())
            image_before = hashlib.sha256((out/'comparison_metrics.png').read_bytes()).hexdigest()
            self.assertEqual(self.run_cli(args+['--target-altitude-km', '600', '--lambert-angle-deg', '175']), 0)
            second = json.loads((out/'comparison.json').read_text())
            self.assertNotEqual(first['run_id'], second['run_id'])
            self.assertNotEqual(first['generated_at_utc'], second['generated_at_utc'])
            self.assertEqual(second['target_altitude_km'], 600)
            self.assertEqual(second['methods'][0]['arrival_angle_deg'], 175)
            self.assertNotEqual(first['methods'][0]['total_delta_v_m_s'], second['methods'][0]['total_delta_v_m_s'])
            self.assertNotEqual(image_before, hashlib.sha256((out/'comparison_metrics.png').read_bytes()).hexdigest())
            summary = json.loads((out/'method_2/summary.json').read_text())
            self.assertEqual(summary['run_id'], second['run_id'])
            self.assertAlmostEqual(summary['metrics']['terminal_altitude_m'], 600000, delta=1)

    def test_selection_no_plots_and_failure_cannot_leave_old_success_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(self.run_cli(['--all', '--output-dir', directory, '--no-plots']), 0)
            (out/'comparison_metrics.png').write_bytes(b'old plot')
            (out/'method_1/level0_trajectory.png').write_bytes(b'old plot')
            (out/'method_7').mkdir()
            (out/'method_7/summary.json').write_text('{}')
            self.assertEqual(self.run_cli(['--method-id', '1', '--intermediate-altitude-km', '450',
                                          '--output-dir', directory, '--no-plots']), 1)
            data = json.loads((out/'comparison.json').read_text())
            self.assertEqual(len(data['methods']), 1)
            self.assertEqual(data['methods'][0]['status'], 'ERROR')
            self.assertTrue((out/'method_1/error.json').exists())
            for name in ('comparison_metrics.png', 'method_1/summary.json', 'method_1/level0_trajectory.png',
                         'method_0/summary.json', 'method_2/summary.json', 'method_7'):
                self.assertFalse((out/name).exists(), name)

    def test_default_output_is_beside_script_when_launched_from_another_directory(self):
        project = Path(runner.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            script_dir, launch_dir = work/'script', work/'launch'
            script_dir.mkdir()
            launch_dir.mkdir()
            script = script_dir/'run_level0_comparison.py'
            shutil.copy2(project/'run_level0_comparison.py', script)
            env = {**os.environ, 'PYTHONPATH': str(project)}
            process = subprocess.run([sys.executable, str(script), '--method-id', '2', '--no-plots'],
                                     cwd=launch_dir, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            self.assertTrue((script_dir/'comparison_results/comparison.json').exists())
            self.assertFalse((launch_dir/'comparison_results').exists())


if __name__ == '__main__':
    unittest.main()
