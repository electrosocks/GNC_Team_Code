# Version 0.4.0

- Consolidated the runtime menu to 0 Hohmann, 1 bielliptic, 2 Lambert, 3 CW.
  IDs from 0.3 are not aliases; update scripts that used the old menu.
- Kept the baseline Hohmann equations and original Hohmann-only runner.
- Added one signed SI bielliptic capability with an explicit apoapsis parameter.
- Replaced the three runtime Lambert choices with one scaled zero-revolution
  Lancaster-Blanchard/Izzo formulation and bracketed root finding. Exact 180-degree
  geometry uses an explicit orbital normal and algebraic velocity reconstruction.
- Added a general Lambert state request that accepts eccentric initial orbits
  and arbitrary target states; the Level 0 example retains circular target setup.
- Set the demo Lambert endpoint to 180 degrees, with the Hohmann duration supplied
  as its default input. Reports display arrival angles to expose unequal constraints.
- Retained the reusable CW equations and explicit nonlinear-correction toggle.
- Removed team component switches and the full team source collection from the
  final package. A few independent references remain under verification/ only.
- Made default output directories relative to the runner file rather than the
  launch directory. Every comparison run clears known generated outputs before
  writing, so no-plots, method selection, and errors cannot leave old success plots.
- Added UTC timestamps/run IDs to reports and images; prints resolved parameters
  and absolute output location. The separate results/ directory is still produced
  only by run_level0.py; comparison_results/ belongs to run_level0_comparison.py.
- 44 tests pass, including numerical shooting/propagation, matched endpoint/time
  comparisons, and repeated changed-parameter runs that replace data and images.

Supported dynamics remain two-body gravity and ideal impulses. Multiple-revolution
Lambert, coincident/same-ray radial endpoint solves, finite burns, perturbations,
and autonomous mission-level tool selection are outside this revision.
