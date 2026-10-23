# Level 0 Requirements

## Mission and environment

Execute a manually specified, two-impulse Hohmann transfer from a 400 km circular
Earth orbit to a 500 km circular Earth orbit. Verify the resulting orbit for one
additional revolution. The transfer ends immediately after circularization;
the extra revolution is validation time.

| Parameter | Value |
|---|---:|
| Gravitational parameter | 3.986004418e14 m^3/s^2 |
| Spherical reference radius | 6378137 m |
| Departure radius | 6778137 m |
| Target radius | 6878137 m |
| Initial epoch | 0 s |
| Initial position | [6778137, 0, 0] m |
| Initial velocity | [0, sqrt(mu / 6778137), 0] m/s |
| Inclination | 0 rad |
| Frame | ECI, fixed inertial axes |
| Forces | Point-mass Earth gravity |
| Maneuvers | Instantaneous inertial velocity increments |

The Earth equatorial radius and gravitational parameter use the WGS 84 values
listed by [NGA](https://earth-info.nga.mil/index.php?action=wgs84&dir=wgs84).
Altitude is radial distance minus the chosen spherical reference radius.

## Module contracts

- Use SI units internally and identify the inertial frame explicitly.
- Use a common epoch origin for all states and commands.
- Accept the central-body parameters as data; only the scenario runner selects Earth.
- The tool proposes burn commands without modifying its input state.
- The mathematical Hohmann function accepts two radii and mu, without a simulator.
- The executor stops at every burn epoch and records both sides of the impulse.
- Evaluate the numerically propagated state independently of tool-reported predictions.
- Check conservation separately for each coast, because impulses change energy and angular momentum.
- Keep plotting and file output outside guidance and dynamics.

## Analytical benchmark

| Quantity | Reference |
|---|---:|
| Departure delta-v magnitude | 28.025839314408586 m/s |
| Arrival delta-v magnitude | 27.92341283132191 m/s |
| Total delta-v | 55.9492521457305 m/s |
| Transfer time | 2807.5941199195818 s |
| Target circular speed | 7612.608173223869 m/s |
| Verification orbit period | 5676.9780285258585 s |

At transfer completion, the reference position is [-6878137, 0, 0] m and the
reference velocity is [0, -sqrt(mu / 6878137), 0] m/s. The first burn points +y;
the second points -y. Both are prograde in their local direction of motion.

## Acceptance criteria

| Check | Maximum allowed error |
|---|---:|
| Initial unforced orbit closure, position | 1 m |
| Initial unforced orbit closure, velocity | 0.001 m/s |
| Initial unforced orbit radius | 1 m |
| Transfer endpoint, position | 1 m |
| Transfer endpoint, velocity | 0.001 m/s |
| Final semimajor axis relative to target radius | 1 m |
| Final periapsis and apoapsis relative to target radius | 1 m |
| Altitude at saved verification points | 1 m |
| Final eccentricity | 1e-7 |
| Change in orbital-plane normal direction | 1e-8 rad |
| Relative specific-energy drift within each coast | 1e-9 |
| Relative angular-momentum-vector drift within each coast | 1e-9 |
| Total applied delta-v relative to reference | 1e-6 m/s |
| Transfer duration relative to reference | 1e-6 s |
| Position jump at an impulse | 1 m (implementation preserves it exactly) |
| Applied velocity jump relative to command | 0.001 m/s |

Use DOP853 with relative tolerance 1e-11, position absolute tolerance 1e-5 m,
and velocity absolute tolerance 1e-8 m/s. Save coast samples at intervals no
larger than 30 seconds, including exact endpoints. Solver tolerances control
local numerical error; the physical checks above assess the resulting trajectory.

## Reuse and negative checks

The same tool must work without code changes for an orbit lowering, an inclined
or retrograde orbit, another central body's mu, and a nonzero initial epoch.
An already satisfied circular-orbit request returns no maneuvers. Invalid radii,
noncircular initial orbits, frame mismatches, and incorrect burn epochs must be
rejected. An intentionally underperformed burn must fail the independent
evaluation. Passing these checks does not validate higher-fidelity flight dynamics.
