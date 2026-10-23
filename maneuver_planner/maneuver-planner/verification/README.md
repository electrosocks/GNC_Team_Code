# Verification references

These modules are never imported by `maneuver_planner` or its runners and are
excluded from the installed package. Tests deliberately compare different
formulations and independently propagate the returned velocities.

- `reference_transfers.py`: speed-ratio Hohmann and helper-based vis-viva
  bielliptic equations, adapted from the earlier team comparisons. Production
  uses one Hohmann and one signed bielliptic implementation.
- `lambert_universal_reference.py` and `lambert_helpers.py`: the earlier team's
  universal-variable reference, in km, km/s, seconds and km^3/s^2. Its equations
  use Stumpff functions and differ from the production Lancaster-Blanchard/Izzo
  formulation. This reference rejects collinear geometry and has a bounded root
  scan; it is not evidence of general edge-case coverage. Exact 180-degree tests
  instead use analytic circular/Hohmann cases and independent numerical shooting.

The original source collection and reviewed adaptations were retained in the
previous project version. These few references are the only comparison-only
implementations distributed with version 0.4.
