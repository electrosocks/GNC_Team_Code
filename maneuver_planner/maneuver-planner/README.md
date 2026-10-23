# Spacecraft Maneuver Planner: Level 0

Version 0.4 provides four reusable capabilities and a manual comparison runner.
The original Hohmann benchmark remains available through `run_level0.py`.
The autonomous planner is a later development step.

## Install and run

Requires Python 3.10 or newer. Extract this revision into a fresh folder rather
than overlaying version 0.3, whose obsolete files would otherwise remain.
From this project directory:

```bash
python -m pip install -e ".[simulation]"
python run_level0_comparison.py
```

Edit the settings at the top of **`run_level0_comparison.py`**:

```python
METHOD_ID = "all"  # Or one number from the table below.
INITIAL_ALTITUDE_KM = 400.0
TARGET_ALTITUDE_KM = 500.0
LAMBERT_ARRIVAL_ANGLE_DEG = 180.0
CW_NONLINEAR_CORRECTION = True
```

**The IDs changed in v0.4. Old IDs 4 through 7 no longer exist.**

| ID | Capability | Configuration |
|---:|---|---|
| 0 | Hohmann | Two tangential burns between circular coplanar orbits |
| 1 | Bielliptic | Three burns; specify intermediate apoapsis altitude |
| 2 | Lambert | Specify arrival angle and flight time for the Level 0 demo |
| 3 | CW | Target a circular reference; optional two-body correction |

Command-line arguments override those settings:

```bash
python run_level0_comparison.py --method-id 2
python run_level0_comparison.py --all --target-altitude-km 600
python run_level0_comparison.py --method-id 2 --lambert-angle-deg 175 --lambert-tof-s 3000
python run_level0_comparison.py --method-id 3 --no-cw-correction
python run_level0_comparison.py --list
```

## Where the current results are

The default output is **`comparison_results/` beside the comparison script**,
regardless of the directory from which you launch it. The runner prints the
absolute location, actual settings, UTC creation time, and a fresh run ID.

- `comparison.txt`, `comparison.csv`, `comparison.json`: current selected methods.
- `comparison_trajectories.png`, `comparison_metrics.png`: current plots.
- `method_0/` through `method_3/`: commands, burn vectors, sampled trajectory,
  validation report, and individual plot for each selected method.
- CW also saves its initial prediction and correction history in `cw_diagnostics.json`.

Each run replaces these generated outputs. Unselected methods' old generated files
are removed. `--no-plots` removes old plots instead of leaving stale images.
A failed method writes `error.json`; a previous successful trajectory cannot remain
beside that error. Run IDs appear in the JSON, text reports, and plots.
Unrelated user files in the output directory are preserved.

**`results/` belongs to `run_level0.py` and is a separate Hohmann-only result set.**
Running the comparison updates `comparison_results/`, not that older result set.
If a viewer already has an image open, reopen it after running and check its
embedded run ID. To choose another output location, pass `--output-dir PATH`;
explicit relative paths are relative to your launch directory.

## What is being compared

With the supplied Earth configuration, initial altitude 400 km, target altitude
500 km, intermediate bielliptic apoapsis 2000 km, and correction enabled:

| ID | Capability | Delta-v (m/s) | Transfer (min) | Arrival angle | Burns |
|---:|---|---:|---:|---:|---:|
| 0 | Hohmann | 55.949252 | 46.793235 | 180 degrees | 2 |
| 1 | Bielliptic | 786.455823 | 109.964047 | 360 degrees | 3 |
| 2 | Lambert | 55.949252 | 46.793235 | 180 degrees | 2 |
| 3 | CW + two-body correction | 143.329389 | 46.793235 | 178.040831 degrees | 2 |

The Lambert default now uses Hohmann's endpoint and supplied duration, so their
costs agree. No Hohmann calculation or fallback is called inside the Lambert solver.
`LAMBERT_TIME_OF_FLIGHT_S = None` supplies the half-period at the mean radius;
it does not optimize time. `CW_TIME_OF_FLIGHT_S = None` supplies the same clock.

CW's virtual reference starts on the target circular orbit at the same angular
position as the initial spacecraft. It travels about 178.04 degrees during the
default interval. This is a different endpoint from the 180-degree transfer.
Giving Lambert that same endpoint/time reproduces the corrected CW burns.
Bielliptic also uses a different route and duration.

**PASS validates the assigned trajectory, not minimum fuel or identical constraints.**
The evaluator checks the target orbit, declared endpoint, burn execution,
conservation, a complete verification revolution, and sampled body clearance.
The verification revolution is excluded from transfer time and maneuver cost.
A FAIL or ERROR returns exit code 1; successful runs return 0.

## Reusable calculations

All public calculation interfaces use meters, seconds, m/s, and m^3/s^2.
Radii are measured from the body's center. Earth constants live in `constants.py`
and the example runners, not in the maneuver equations. Frame/epoch conventions
are explicit in `models.py`; frame conversions are the caller's responsibility.

- `hohmann_tool.py`: pure `hohmann_transfer(r1, r2, mu)` plus a state adapter.
- `bielliptic_tool.py`: pure `bielliptic_transfer(r1, r2, rb, mu)` plus a state adapter.
- `lambert.py`: pure endpoint targeting, with a separate `LambertTool` state adapter.
- `cw_equations.py`: reusable CW derivative, state-transition matrix, propagation,
  and linear targeting. `cw_transfer_tool.py` contains the Level 0 adapter/corrector.
- `physics.py`: replaceable two-body propagation and ideal impulse execution.
- `evaluation.py`: checks the executed result independently of the maneuver tool.
- `comparison.py`: adapts this circular-orbit example to the four capabilities.

For general Lambert targeting:

```python
from maneuver_planner.lambert import solve_lambert

solution = solve_lambert(
    [6_778_137, 0, 0], [-6_878_137, 0, 0],
    time_of_flight_s=2807.5941199195818,
    mu_m3_s2=3.986004418e14,
    plane_normal=[0, 0, 1],
)
v_departure = solution.departure_velocity_m_s
v_arrival = solution.arrival_velocity_m_s
```

`plane_normal` is the desired angular-momentum direction. It supports arbitrary
orbital planes and sets the short/long-way direction consistently. It must be
perpendicular to both endpoint positions. Without it, noncollinear endpoints
select the short path. Exactly antipodal endpoints require an explicit normal.
`LambertRequest(initial_state, target_state, body, plane_normal)` also allows an
eccentric initial orbit and a noncircular target state. It does not inherit
Hohmann's circular-orbit restriction.

The solver uses the published Lancaster-Blanchard/Izzo time equation with a
bracketed scalar root solve, stable treatment near parabolic energy, scaling, and
algebraic velocity reconstruction at 180 degrees. Reference: D. Izzo (2015),
[Revisiting Lambert's problem](https://doi.org/10.1007/s10569-014-9587-y),
Eqs. 18, 20 and Algorithm 1. This is not the full Householder algorithm in that paper.

Scope is **zero revolutions**. Elliptic, hyperbolic, near-parabolic, short/long-way,
and exact/near-180-degree transfers are tested. Multiple revolutions, same-ray
radial transfers, and coincident endpoints are explicitly unsupported. Extremely
ill-conditioned cases may raise a convergence error. Valid mathematical solutions
still need independent body-clearance and mission-constraint evaluation.

## CW correction

Pure CW is a linear approximation around a circular reference. In the default
400-to-500 km case its burns miss the target position by about 9.39 km when
executed under full two-body gravity. Disabling correction preserves that result
and its failed strict validation; the report does not hide the error.

When enabled, the separate corrector propagates the CW departure burn, measures
the terminal position miss, and uses CW's velocity-to-position sensitivity to
adjust the departure velocity. It repeats, then computes the arrival burn from
the nonlinear predicted arrival velocity. There are still two actual burns;
iterations are planning calculations. The current correction model is two-body
gravity, not drag, J2, navigation uncertainty, or finite-duration thrust.

## Verification

```bash
python -m unittest discover -s tests -v
python run_level0.py
```

All **44 tests** pass. Checks include the analytic Hohmann benchmark, an alternate
bielliptic formulation, independent universal-variable Lambert comparisons,
independent numerical propagation and shooting at 180 degrees, non-Hohmann flight
times, varied orbital orientations, another body and solar-system scale,
near-parabolic/hyperbolic trajectories, raw/corrected CW, and changed-parameter
reruns that replace both data and plot contents.

`verification/` holds reference equations only. It is excluded from the installed
package and is never imported by production tools. The former team menu, MATLAB
port, and full team source collection have been removed from the deliverable.
The original submissions remain available in the earlier project version and
original uploaded files. See `CHANGELOG.md` for this revision's migration notes.
