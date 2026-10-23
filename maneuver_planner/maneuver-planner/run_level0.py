"""Run the manually specified Level 0 scenario and write reproducible results."""

import argparse
import csv
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from maneuver_planner.constants import EARTH
from maneuver_planner.evaluation import ValidationTolerances, evaluate_circular_transfer
from maneuver_planner.hohmann_tool import HohmannTool
from maneuver_planner.models import TransferRequest
from maneuver_planner.orbital_utils import circular_state, orbital_period_s
from maneuver_planner.physics import IntegratorSettings, execute_candidate, propagate_two_body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-altitude-km", type=float, default=400.0)
    parser.add_argument("--target-altitude-km", type=float, default=500.0)
    parser.add_argument("--inclination-deg", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--show", action="store_true", help="Also open the figure interactively.")
    args = parser.parse_args()

    # These are scenario choices. The reusable modules do not import EARTH.
    body = EARTH
    initial = circular_state(body.radius_m + args.initial_altitude_km * 1000, body,
                             inclination_rad=np.deg2rad(args.inclination_deg), frame="ECI")
    target_radius = body.radius_m + args.target_altitude_km * 1000
    request = TransferRequest(initial, body, target_radius)
    candidate = HohmannTool().propose(request)
    integrator = IntegratorSettings()
    tolerances = ValidationTolerances()
    coast = partial(propagate_two_body, settings=integrator, sample_step_s=30.0)
    result = execute_candidate(candidate, propagator=coast,
                               verification_duration_s=orbital_period_s(target_radius, body.mu_m3_s2))
    report = evaluate_circular_transfer(result, target_radius, tolerances=tolerances)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "maneuver.json").write_text(json.dumps(candidate.to_dict(), indent=2, allow_nan=False) + "\n")
    settings_for_json = asdict(integrator)
    # Infinity is valid to SciPy but not standard JSON.
    if np.isinf(settings_for_json["max_step_s"]):
        settings_for_json["max_step_s"] = None
    summary = {"scenario": "circular Hohmann transfer", "central_body": asdict(body),
               "initial_radius_m": float(np.linalg.norm(initial.position_m)),
               "target_radius_m": target_radius, "frame": initial.frame,
               "integrator": {"method": "DOP853", **settings_for_json,
                              "max_step_null_means": "unbounded", "sample_step_s": 30.0},
               "tolerances": asdict(tolerances),
               "verification_duration_s": result.trajectory.epochs_s[-1] - candidate.end_epoch_s,
               **report.to_dict()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "summary.txt").write_text(report.format_text() + "\n")
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        for epoch, vector in zip(result.trajectory.epochs_s, result.trajectory.states):
            writer.writerow([epoch, *vector])
    with (output / "burns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch_s", "frame", "dv_x_m_s", "dv_y_m_s", "dv_z_m_s", "magnitude_m_s"])
        for record in result.burn_records:
            burn = record.command
            writer.writerow([burn.epoch_s, burn.frame, *burn.delta_v_m_s, burn.magnitude_m_s])

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    from maneuver_planner.plotting import plot_transfer
    import matplotlib.pyplot as plt
    figure = plot_transfer(result, target_radius)
    figure.savefig(output / "level0_trajectory.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(figure)
    print(report.format_text())
    print(f"\nResults: {output.resolve()}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
