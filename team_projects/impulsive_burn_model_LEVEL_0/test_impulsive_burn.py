#---
# ===================
# ===================
# TESTS FOR IMPULSIVE_BURN PYTHON FILE
# ===================
# This file will help as:
# Each test below checks ONE specific claim about the burn model
# e.g
# "position doesn't change
# Last test class ties back to Level 0 specs
# RE-DRIVES the expected Delta-V values from vis-viva itself
# ===========================================================
# ===================

import numpy as np
import pytest

#pytest here is the test runner, 'pytest_approx(x)' below means "equal to x,
# allowing for tiny floating-point rounding error"

from impulsive_burn_model import StateVector, apply_impulsive_burn, prograde_delta_v

#-----------------------------------
# Physical constants
# impulsive burn model doesn't need to know mu or earth's radius
# as it only cares about velocities
# This keeps these constants local to the test file
# model is orbit-agnostic
# for this test it happens to use specific orbit scenario to exercise it
# --------------------------------------
MU_EARTH = 398600.4418 # km^3/s^2 ----------- Earth's gravitational parameter
R_EARTH = 6378.137     # km       ----------- Earth's mean equatorial radius

def circular_state(altitude_km: float, time: float = 0.0) -> StateVector:
    ##....
    # Build a StateVector for a circular orbit at the given altitude, using the same initial-condition convention given in the Level 0 SPEC:
    # Position of the +x axis, velocity in +y with circular-orbit magnitude.
    # --------
    # This is a stand in component
    #---------------
    # 
    # 
    # The Math:
    # for a circular orbit, gravity provides exactly the centripetal force needed to keep the
    # spacecraft moving in a circle at a constant radius
    # setting gravitational force equal to centripetal force
    # and solving for speed gives
    #           v_circular = sqrt(mu / r)
    # where r is the orbital radius (altitude + Earth's radius, not ALTITUDE ALONE
    # ------------------------------------
    
    r = R_EARTH + altitude_km
    v_circ = np.sqrt(MU_EARTH / r)
    position = np.array([r, 0.0, 0.0]) #----------- Spacecraft on the +x axis
    velocity = np.array([0.0, v_circ, 0.0]) #--------- Moving in +y, perpendicular to position, required for a circular orbit

    return StateVector(position=position, velocity=velocity, time=time)

def specific_energy (state: StateVector, mu: float = MU_EARTH) -> float:
    ###---
    # Compute specific orbital energy (energy per unit mass):
    #   epsilon = v^2 / 2 - mu / r
    # 
    # This part is useful as:
    # Specific energy depends ONLY on the orbits semi-major axis
    #       epsilon = -mu / (2*a)
    # So if we compute epsilon straight from a post-burn pair and it matches -mu/(2*a_transfer),
    # strong independent check that the burn put the spacecraft on the RIGHT transfer ellispe
    # not only that speed changed by the right amount, but that the resulting orbit
    # shape is correct
    # ----------

    r = np.linalg.norm(state.position)
    v = np.linalg.norm(state.velocity)
    return v**2 / 2 - mu / r

#============================
# Basic mechanics of the impulsive burn, independent of any
# specific orbit scenario. Meaning this would still make sense if 
# Level 0 used compleltey different altitudes inputs
# ===============================================

class TestBasicBurnMechanics:

    def test_zero_delta_v_leaves_state_unchanged(self):
        #sanity check, if fails something is fundementally wrong
        # i.e.
        # accidently adding a constant, not the delta_v
        # ---------------------
        state = circular_state(400.0)
        result = apply_impulsive_burn(state, delta_v=np.zeros(3))

        assert np.allclose(result.state.velocity, state.velocity)
        assert np.allclose(result.state.position, state.position)
        assert result.delta_v_magnitude == pytest.approx(0.0)
       

    def test_position_and_time_unchanged_by_burn(self):
        #defining property of an "impulsive" burn as opposed
        # to a finite-duration one
        # if test ever fails, the burn
        # model has stopped being "impulsive"
        # such as introducing a time-dependent position update#
        state = circular_state(400.0, time=123.4)
        dv = prograde_delta_v(state.velocity, 0.5)
        result = apply_impulsive_burn(state,dv)

        assert np.allclose(result.state.position, state.position)
        assert result.state.time == state.time

    def test_speed_change_matches_delta_v_magnitude(self):
        #----------------------
        # For a PROGRADE burn specifically (dv parallel to v), the new
        # speed should be exactly old_speed + dv_magnitdue
        # the vectors head-to-tail along the same line, so their magnitudes add directly
        # WOULD NOT HOLD as some other angle, this test uses
        # prograde_delta_v rather than arbitrary direciton
        # --------------------------
        state = circular_state(400.0)
        dv_mag = 0.5 # km/s
        dv = prograde_delta_v(state.velocity, dv_mag)
        result = apply_impulsive_burn(state,dv)

        speed_before = np.linalg.norm(state.velocity)
        speed_after = np.linalg.norm(result.state.velocity)
        assert speed_after - speed_before == pytest.approx(dv_mag, rel=1e-9)

    def test_prograde_burn_is_aligned_with_velocity(self):
        #This CHECKS THE DIRECTION, not just magnitdue:
        #The delta_v vector that prograde_delta_v()
        #built should point exactly along the pre-burn
        # velocity unit vector
        # Confirms prograde_delta_v()
        # isnt accidenally intro any off-axis componenet#

        state=circular_state(400.0)
        dv = prograde_delta_v(state.velocity, 0.5)
        result = apply_impulsive_burn(state, dv)

        v_before_dir = state.velocity / np.linalg.norm(state.velocity)
        dv_dir = result.delta_v_vector / np.linalg.norm(result.delta_v_vector)
        assert np.allclose(v_before_dir, dv_dir)

    def test_retrograde_burn_reduces_speed(self):
        #A negative magntidue should slow the spacecraft down
        # point downard relative to velocity, not used in this level, but 
        # worth noting now since other level (deorbit, phasing) will need retrograde burns too
        # ------------------------
        state = circular_state(400.0)
        dv = prograde_delta_v(state.velocity, -0.3)
        result = apply_impulsive_burn(state, dv)

        assert np.linalg.norm(result.state.velocity) < np.linalg.norm(state.velocity)

#====================================================
# Actual Level 0 Scenario (400 km --->>> 500 km Hohmann transfer
# ====================================================

class TestHohmannLevel0Scenario:
    ###
    #Sanity check against SPEC in LV 0
    # DELTA_V valeus are derived from vis-viva 
    # this class works in two ways:
    # 1. test of the burn model
    # 2. independent check on the Delta_v numbers used elsewhere for design
    ###

    def setup_method(self):
        #pytest auto calls setup_method() before EACH test
        #method in this class, so every test gets a fresh, indepent cpy of these values
        #---> no state leaking b/w tests

        self.r1 = R_EARTH + 400.0 # Starting circular orbit radius [km]
        self.r2 = R_EARTH + 500.0 # Target circular orbit radius [km]

        #SEmi_major axis of the transfer ellipse
        #by definition, a Hohmann transfer's ellipse has its periapsis at the start
        # orbit's radius and its apoapsis at the target orbit's
        # radius. 
        # Semi-major axis of an ellilpse just the average of its closest
        # and farthest distance from the focus
        self.a_transfer = (self.r1 + self.r2) / 2

        #Circular-orbit speeds at each radius: v = sqrt(mu / r)
        self.v1_circular = np.sqrt(MU_EARTH / self.r1)
        self.v2_circular = np.sqrt(MU_EARTH / self.r2)

        #Speed ON THE TRANSFER ELLIPSE AT EACH RADIUS
        # from vis - viva: -------> 
        # v^2 = mu * (2/r - 1/a)
        # Vis-viva relates speed to position for ANY orbit
        # Given that orbit's semi-major axis 'a' 
        # 'a' here is the TRANSFER orbit's semi major axis, and we
        # evaluate it at r1 (departure point) and r2 (arrival point)
        # ---------------------------
        self.v_transfer_at_r1 = np.sqrt(MU_EARTH * (2 / self.r1 - 1 / self.a_transfer))
        self.v_transfer_at_r2 = np.sqrt(MU_EARTH * (2 / self.r2 - 1 / self.a_transfer))

        #BURN 1 (departure): SPACECRAFT currently moving at 
        # v1_circular. To get onto the transfer ellipse, it needs to be 
        # moving at v_transfer_at_r1 instead
        # burn makes up the difference:
        self.dv1 = self.v_transfer_at_r1 - self.v1_circular

        #BURN 2 (circularization) at r2, the spacecraft arrives moving
        # at v_transfer_at_r2 (slower than circular speed there, because
        # it's decelerated climbing out of the gravity well on the way up
        # to circularize, needs to speed back up the v2_circular
        self.dv2 = self.v2_circular - self.v_transfer_at_r2

    def test_departure_burn_places_state_on_transfer_ellipse(self):
        #Checks ORBIT SHAPE via specific energy
        # -----------------#
        state0 = circular_state(400.0)
        dv = prograde_delta_v(state0.velocity, self.dv1)
        result = apply_impulsive_burn(state0, dv)

        expected_energy = -MU_EARTH / (2 * self.a_transfer)
        assert specific_energy(result.state) == pytest.approx(expected_energy, rel=1e-9)

    def test_departure_burn_speed_matches_transfer_vis_viva(self):
        #MORE DIRECT CHECK
        # DOES THE resutling SPEED match what VIS_VIVA PREDICTS
        # AT r1 on the transfer ellipse?????
        # Paried with ENERGY CHECK above
        # THIS CONFIRMS both the magntidue of the burn 
        # and resulting orbit are correct
        state0 = circular_state(400.0)
        dv = prograde_delta_v(state0.velocity, self.dv1)
        result = apply_impulsive_burn(state0, dv)

        assert np.linalg.norm(result.state.velocity) == pytest.approx(
            self.v_transfer_at_r1, rel=1e-9
        )


    def test_total_delta_v_is_small_and_positive(self):
        # A coarse "did I mess up a sign or a unit somewhere" guard rail.
        # Raising a 100 km altitude should cost on the order of TENS of
        # m/s, not, say, several km/s (which would indicate something
        # like forgetting to add Earth's radius to the altitude, or a
        # sign error making one of the burns subtract instead of add).
        total_dv = self.dv1 + self.dv2
        assert 0.0 < total_dv < 1.0  # km/s, i.e. under 1000 m/s
 
    def test_circularization_burn_reaches_target_circular_speed(self):
        # This test checks burn #2 IN ISOLATION: rather than propagating
        # burn #1's result all the way around the transfer ellipse (that
        # would require the two-body propagator, which isn't this
        # module's job to test), we construct the "arrival at apoapsis"
        # state analytically and confirm burn #2 alone finishes the job.
        #
        # Position is placed at -r2 on the x-axis (opposite side of the
        # orbit from where burn 1 happened, i.e. half an orbit later) and
        # velocity is set to the transfer-ellipse speed at r2, pointed
        # the correct way for a spacecraft coasting past apoapsis on this
        # ellipse.
        state_at_apoapsis = StateVector(
            position=np.array([-self.r2, 0.0, 0.0]),
            velocity=np.array([0.0, -self.v_transfer_at_r2, 0.0]),
            time=0.0,
        )
        dv = prograde_delta_v(state_at_apoapsis.velocity, self.dv2)
        result = apply_impulsive_burn(state_at_apoapsis, dv)
 
        assert np.linalg.norm(result.state.velocity) == pytest.approx(
            self.v2_circular, rel=1e-9
        )


    
    
    