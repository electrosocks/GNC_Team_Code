"""Independent physical checks of the zero-revolution Lambert capability."""

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from maneuver_planner.constants import EARTH
from maneuver_planner.hohmann_tool import hohmann_transfer
from maneuver_planner.lambert import solve_lambert
from maneuver_planner.lambert_tool import LambertRequest, LambertTool
from maneuver_planner.models import CartesianState, CentralBody
from maneuver_planner.physics import execute_candidate
from verification.lambert_universal_reference import solve_lambert as reference_lambert


def independent_coast(r, v, duration, mu):
    # Dimensionless integration avoids changing physical tolerances with the body.
    scale = np.linalg.norm(r)
    speed = math.sqrt(mu/scale)
    clock = scale/speed
    end = duration/clock
    def derivative(_, y):
        return np.r_[y[3:], -y[:3]/np.linalg.norm(y[:3])**3]
    result = solve_ivp(derivative, (0, end), np.r_[np.array(r)/scale, np.array(v)/speed],
                       method='DOP853', rtol=3e-14, atol=1e-14, max_step=end/200)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:3, -1]*scale, result.y[3:, -1]*speed


class LambertTests(unittest.TestCase):
    def assert_arrives(self, r1, r2, duration, mu, normal=None):
        solution = solve_lambert(r1, r2, duration, mu, plane_normal=normal)
        rr, vv = independent_coast(r1, solution.departure_velocity_m_s, duration, mu)
        self.assertLess(np.linalg.norm(rr-r2), max(1e-3, np.linalg.norm(r1)*1e-12))
        self.assertLess(np.linalg.norm(vv-solution.arrival_velocity_m_s),
                        max(1e-7, np.linalg.norm(vv)*1e-11))
        return solution

    def test_textbook_and_independent_universal_variable_reference(self):
        r1, r2 = np.array([5000., 10000., 2100.])*1000, np.array([-14600., 2500., 7000.])*1000
        solution = self.assert_arrives(r1, r2, 3600, EARTH.mu_m3_s2)
        np.testing.assert_allclose(solution.departure_velocity_m_s, [-5992.5, 1925.4, 3245.6], atol=.1)
        expected = reference_lambert(r1/1000, r2/1000, 3600, EARTH.mu_m3_s2/1e9)
        np.testing.assert_allclose([solution.departure_velocity_m_s, solution.arrival_velocity_m_s],
                                   np.array(expected)*1000, rtol=0, atol=1e-6)

    def test_exact_180_matches_hohmann_raising_and_lowering_without_fallback(self):
        for r1, r2 in ((6778137., 6878137.), (6878137., 6778137.)):
            h = hohmann_transfer(r1, r2, EARTH.mu_m3_s2)
            with patch('maneuver_planner.hohmann_tool.hohmann_transfer', side_effect=AssertionError('No fallback')):
                solution = self.assert_arrives([r1, 0, 0], np.array([-r2, 0, 0]),
                                               h.time_of_flight_s, EARTH.mu_m3_s2, [0, 0, 1])
            dv = (np.linalg.norm(solution.departure_velocity_m_s-[0, math.sqrt(EARTH.mu_m3_s2/r1), 0])
                  +np.linalg.norm([0, -math.sqrt(EARTH.mu_m3_s2/r2), 0]-solution.arrival_velocity_m_s))
            self.assertAlmostEqual(dv, h.total_delta_v_m_s, delta=1e-8)

    def test_180_at_non_hohmann_times(self):
        r1, r2 = [6778137., 0, 0], np.array([-6878137., 0, 0])
        for duration in (1700., 4200., 8000.):
            with self.subTest(duration=duration):
                self.assert_arrives(r1, r2, duration, EARTH.mu_m3_s2, [0, 0, 1])

    def test_circular_orbits_across_near_zero_180_and_long_way_angles(self):
        radius, mu = 7e6, EARTH.mu_m3_s2
        speed, n = math.sqrt(mu/radius), math.sqrt(mu/radius**3)
        for degrees in (.001, 30., 170., 179.99, 180., 180.000001, 220., 300., 359.999):
            angle = np.deg2rad(degrees)
            r2 = radius*np.array([np.cos(angle), np.sin(angle), 0])
            with self.subTest(degrees=degrees):
                solution = self.assert_arrives([radius, 0, 0], r2, angle/n, mu, [0, 0, 1])
                np.testing.assert_allclose(solution.departure_velocity_m_s, [0, speed, 0], rtol=0, atol=1e-6)
                self.assertAlmostEqual(solution.transfer_angle_rad, angle, delta=1e-12)

    def test_inclined_polar_and_retrograde_180(self):
        for inclination in (.7, math.pi/2, 2.6):
            p = np.array([1., 0, 0])
            q = np.array([0, np.cos(inclination), np.sin(inclination)])
            normal = np.cross(p, q)
            radius, mu = 7e6, EARTH.mu_m3_s2
            duration = math.pi*math.sqrt(radius**3/mu)
            solution = self.assert_arrives(radius*p, -radius*p, duration, mu, normal)
            np.testing.assert_allclose(solution.departure_velocity_m_s, math.sqrt(mu/radius)*q, atol=1e-8)

    def test_short_long_hyperbolic_and_near_parabolic(self):
        mu, r1 = EARTH.mu_m3_s2, np.array([7e6, 0, 0])
        for degrees, duration in ((60, 300), (60, 1200), (170, 2800), (220, 4500), (300, 6000)):
            angle = np.deg2rad(degrees)
            r2 = 7.2e6*np.array([np.cos(angle), np.sin(angle), 0])
            with self.subTest(degrees=degrees, duration=duration):
                solution = self.assert_arrives(r1, r2, duration, mu, [0, 0, 1])
                expected = reference_lambert(r1/1000, r2/1000, duration, mu/1e9)
                np.testing.assert_allclose([solution.departure_velocity_m_s, solution.arrival_velocity_m_s],
                                           np.array(expected)*1000, atol=1e-6, rtol=0)
        r2 = 7.2e6*np.array([.5, math.sqrt(.75), 0])
        s = (np.linalg.norm(r1)+np.linalg.norm(r2)+np.linalg.norm(r2-r1))/2
        lam = math.sqrt(1-np.linalg.norm(r2-r1)/s)
        parabolic_time = (2/3*(1-lam**3))/math.sqrt(2*mu/s**3)
        for factor in (1-1e-8, 1., 1+1e-8):
            self.assert_arrives(r1, r2, parabolic_time*factor, mu, [0, 0, 1])

    def test_another_body_and_solar_system_scale(self):
        for radius, mu in ((2e6, 8e12), (1.495978707e11, 1.32712440018e20)):
            duration = .7*math.sqrt(radius**3/mu)
            self.assert_arrives([radius, 0, 0], radius*np.array([np.cos(.7), np.sin(.7), 0]),
                                duration, mu, [0, 0, 1])

    def test_180_agrees_with_independent_numerical_shooting(self):
        r1, r2, duration, mu = np.array([6778137., 0, 0]), np.array([-6878137., 0, 0]), 2100., EARTH.mu_m3_s2
        def residual(v):
            rr, _ = independent_coast(r1, np.r_[v*1000, 0], duration, mu)
            return (rr-r2)[:2]/1000
        targeted = root(residual, [0., 7.7], tol=1e-10)
        self.assertTrue(targeted.success)
        solution = solve_lambert(r1, r2, duration, mu, plane_normal=[0, 0, 1])
        np.testing.assert_allclose(solution.departure_velocity_m_s[:2], targeted.x*1000, atol=1e-6, rtol=0)

    def test_general_state_adapter_accepts_eccentric_initial_orbits(self):
        initial = CartesianState(123, [7e6, 0, 0], [250, 7000, 100], 'custom')
        target = CartesianState(1323, [3.6e6, 6.235382907e6, 0], [-6500, 3800, 20], 'custom')
        request = LambertRequest(initial, target, EARTH, [0, 0, 1])
        self.assertTrue(LambertTool().check_applicability(request).applicable)
        result = execute_candidate(LambertTool().propose(request))
        self.assertLess(np.linalg.norm(result.terminal_state.position_m-target.position_m), 1.)
        self.assertLess(np.linalg.norm(result.terminal_state.velocity_m_s-target.velocity_m_s), 1e-3)

    def test_invalid_and_explicitly_unsupported_inputs(self):
        base = ([7e6, 0, 0], [0, 7e6, 0], 1200, EARTH.mu_m3_s2)
        invalid = [([0, 0, 0], base[1], 1200, base[3]),
                   ([float('nan'), 0, 0], base[1], 1200, base[3]),
                   (base[0], base[1], 0, base[3]), (base[0], base[1], 1200, -1)]
        for args in invalid:
            with self.assertRaises(ValueError):
                solve_lambert(*args)
        for target in ([7e6, 0, 0], [8e6, 0, 0]):
            with self.assertRaisesRegex(ValueError, 'Same-ray'):
                solve_lambert(base[0], target, 1200, base[3], plane_normal=[0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'plane_normal'):
            solve_lambert(base[0], [-8e6, 0, 0], 1200, base[3])
        for kwargs in ({'plane_normal':[0, 0, 0]}, {'plane_normal':[1, 0, 0]}, {'revolutions':1}, {'max_iterations':0}):
            with self.assertRaises(ValueError):
                solve_lambert(*base, **kwargs)
        with self.assertRaises(RuntimeError):
            solve_lambert(*base, max_iterations=1)
        with self.assertRaises(ValueError):
            LambertRequest(CartesianState(0, base[0], [0, 7500, 0], 'A'),
                           CartesianState(1200, base[1], [-7500, 0, 0], 'B'), EARTH)


if __name__ == '__main__':
    unittest.main()
