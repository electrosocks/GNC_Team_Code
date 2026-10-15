import numpy as np
import pytest

from Lambert import solve_lambert
from Lambert_helpers import (
    stumpC as stump_c,
    stumpS as stump_s,
    y,
    F,
    solve_z,
)

MU_EARTH = 398600.4418


# =======================================================
#               Stumpff Function Tests
# =======================================================

class TestStumpffFunctions:

    def test_stumpff_c_at_zero(self):
        assert stump_c(0.0) == pytest.approx(0.5)

    def test_stumpff_s_at_zero(self):
        assert stump_s(0.0) == pytest.approx(1.0 / 6.0)

    def test_stumpff_c_positive_z(self):
        z = np.pi ** 2
        expected = 2.0 / (np.pi ** 2)

        assert stump_c(z) == pytest.approx(
            expected,
            rel=1e-10
        )

    def test_stumpff_s_positive_z(self):
        z = np.pi ** 2
        expected = 1.0 / (np.pi ** 2)

        assert stump_s(z) == pytest.approx(
            expected,
            rel=1e-10
        )

    def test_stumpff_c_negative_z(self):
        z = -1.0
        expected = np.cosh(1.0) - 1.0

        assert stump_c(z) == pytest.approx(
            expected,
            rel=1e-10
        )

    def test_stumpff_s_negative_z(self):
        z = -1.0
        expected = np.sinh(1.0) - 1.0

        assert stump_s(z) == pytest.approx(
            expected,
            rel=1e-10
        )


# =======================================================
#               Lambert Solver Tests
# =======================================================

class TestLambertSolver:

    def test_known_lambert_solution(self):
        R1 = np.array([
            5000.0,
            10000.0,
            2100.0
        ])

        R2 = np.array([
            -14600.0,
            2500.0,
            7000.0
        ])

        V1, V2 = solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH,
            orbit_type="pro"
        )

        expected_V1 = np.array([
            -5.9925,
            1.9254,
            3.2456
        ])

        expected_V2 = np.array([
            -3.3125,
            -4.1966,
            -0.3853
        ])

        assert np.allclose(
            V1,
            expected_V1,
            atol=1e-3
        )

        assert np.allclose(
            V2,
            expected_V2,
            atol=1e-3
        )

    def test_output_vectors_have_three_components(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        V1, V2 = solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH
        )

        assert V1.shape == (3,)
        assert V2.shape == (3,)

    def test_output_contains_finite_values(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        V1, V2 = solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH
        )

        assert np.all(np.isfinite(V1))
        assert np.all(np.isfinite(V2))

    def test_accepts_python_lists(self):
        V1, V2 = solve_lambert(
            [5000.0, 10000.0, 2100.0],
            [-14600.0, 2500.0, 7000.0],
            3600.0,
            MU_EARTH
        )

        assert isinstance(V1, np.ndarray)
        assert isinstance(V2, np.ndarray)

    def test_input_vectors_are_not_modified(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        original_R1 = R1.copy()
        original_R2 = R2.copy()

        solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH
        )

        assert np.allclose(R1, original_R1)
        assert np.allclose(R2, original_R2)


# =======================================================
#               Lambert Edge Case Tests
# =======================================================

class TestLambertEdgeCases:

    def test_negative_time_of_flight_raises_error(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                -3600.0,
                MU_EARTH
            )

    def test_zero_time_of_flight_raises_error(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                0.0,
                MU_EARTH
            )

    def test_negative_mu_raises_error(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                -MU_EARTH
            )

    def test_zero_mu_raises_error(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                0.0
            )

    def test_zero_initial_position_raises_error(self):
        R1 = np.array([0.0, 0.0, 0.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                MU_EARTH
            )

    def test_zero_final_position_raises_error(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                MU_EARTH
            )

    def test_invalid_orbit_type_defaults_to_prograde(self):
        R1 = np.array([5000.0, 10000.0, 2100.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        V1_invalid, V2_invalid = solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH,
            orbit_type="sideways"
        )

        V1_pro, V2_pro = solve_lambert(
            R1,
            R2,
            3600.0,
            MU_EARTH,
            orbit_type="pro"
        )

        assert np.allclose(V1_invalid, V1_pro)
        assert np.allclose(V2_invalid, V2_pro)

    def test_position_vectors_must_have_three_components(self):
        R1 = np.array([5000.0, 10000.0])
        R2 = np.array([-14600.0, 2500.0, 7000.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                MU_EARTH
            )

    def test_identical_positions_raise_error(self):
        R1 = np.array([7000.0, 0.0, 0.0])
        R2 = np.array([7000.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                MU_EARTH
            )

    @pytest.mark.xfail(
        reason=(
            "Current solver does not reliably reject exact "
            "180-degree transfer geometry."
        )
    )
    def test_180_degree_transfer_raises_error(self):
        R1 = np.array([7000.0, 0.0, 0.0])
        R2 = np.array([-7000.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            solve_lambert(
                R1,
                R2,
                3600.0,
                MU_EARTH
            )


# =======================================================
#         Two-Body Propagation Helper for Testing
# =======================================================

def two_body_derivative(state, mu):
    """
    Returns the time derivative of a spacecraft state.

    state = [x, y, z, vx, vy, vz]

    Position units: km
    Velocity units: km/s
    """

    position = state[:3]
    velocity = state[3:]

    r = np.linalg.norm(position)

    acceleration = -mu * position / (r ** 3)

    return np.concatenate((velocity, acceleration))


def propagate_two_body(R0, V0, total_time, mu, dt=1.0):
    """
    Propagates a spacecraft state using RK4 integration.
    """

    state = np.concatenate((
        np.asarray(R0, dtype=float),
        np.asarray(V0, dtype=float)
    ))

    time = 0.0

    while time < total_time:

        step = min(dt, total_time - time)

        k1 = two_body_derivative(
            state,
            mu
        )

        k2 = two_body_derivative(
            state + 0.5 * step * k1,
            mu
        )

        k3 = two_body_derivative(
            state + 0.5 * step * k2,
            mu
        )

        k4 = two_body_derivative(
            state + step * k3,
            mu
        )

        state = state + (
            step / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )

        time += step

    final_position = state[:3]
    final_velocity = state[3:]

    return final_position, final_velocity


# =======================================================
#               Lambert Integration Tests
# =======================================================

class TestLambertIntegration:

    def test_lambert_velocity_reaches_target_position(self):

        R1 = np.array([
            5000.0,
            10000.0,
            2100.0
        ])

        R2 = np.array([
            -14600.0,
            2500.0,
            7000.0
        ])

        time_of_flight = 3600.0

        V1, V2 = solve_lambert(
            R1,
            R2,
            time_of_flight,
            MU_EARTH,
            orbit_type="pro"
        )

        propagated_R2, propagated_V2 = propagate_two_body(
            R1,
            V1,
            time_of_flight,
            MU_EARTH,
            dt=1.0
        )

        position_error = np.linalg.norm(
            propagated_R2 - R2
        )

        velocity_error = np.linalg.norm(
            propagated_V2 - V2
        )

        assert position_error < 1.0
        assert velocity_error < 0.001