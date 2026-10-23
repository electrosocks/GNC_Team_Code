"""Execute one or all four Level 0 capabilities and replace the current run outputs."""

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from maneuver_planner.comparison import METHODS, ComparisonSettings, Level0Tool, method_info
from maneuver_planner.constants import EARTH
from maneuver_planner.cw_transfer_tool import CWLevel0Tool, correct_cw_candidate
from maneuver_planner.evaluation import evaluate_target_orbit
from maneuver_planner.models import TransferRequest
from maneuver_planner.orbital_utils import circular_state, orbital_period_s
from maneuver_planner.physics import execute_candidate


# IDs changed in v0.4: one capability per family, independent of its reference code.
METHOD_ID = "all"  # 0 Hohmann; 1 bielliptic; 2 Lambert; 3 CW.
INITIAL_ALTITUDE_KM = 400.0
TARGET_ALTITUDE_KM = 500.0                          # Bielliptical wins at 120_000.0; standard is 500.0 km
INCLINATION_DEG = 0.0
BIELLIPTIC_APOAPSIS_ALTITUDE_KM = 2000.0            # Bielliptical wins at 200_000.0; standard is 2000.0 km
LAMBERT_ARRIVAL_ANGLE_DEG = 90.0
LAMBERT_TIME_OF_FLIGHT_S = 30 * 60  # None supplies the Hohmann reference duration; not optimized. This is in seconds.
CW_TIME_OF_FLIGHT_S = None  # Same default comparison clock; CW targets its own reference spacecraft.
CW_NONLINEAR_CORRECTION = True
OUTPUT_DIR = Path(__file__).resolve().parent/"comparison_results"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def clear_generated_outputs(output):
    """Remove only known generated files, including obsolete method IDs and plots."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "comparison.csv", "comparison.txt",
                 "comparison_trajectories.png", "comparison_metrics.png"):
        (output/name).unlink(missing_ok=True)
    for directory in output.glob("method_*"):
        if not directory.is_dir() or not directory.name.removeprefix("method_").isdigit():
            continue
        for name in ("maneuver.json", "summary.json", "summary.txt", "trajectory.csv",
                     "burns.csv", "cw_diagnostics.json", "level0_trajectory.png", "error.json"):
            (directory/name).unlink(missing_ok=True)
        if not any(directory.iterdir()):
            directory.rmdir()


def save_run(output, info, result, report, metadata, angle):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"maneuver.json", result.candidate.to_dict())
    write_json(output/"summary.json", {**metadata, "method": asdict(info),
                                      "arrival_angle_deg": angle, **report.to_dict()})
    mode = result.candidate.tool_name if info.method_id == 3 else info.name
    heading = (f"Run {metadata['run_id']} | {metadata['generated_at_utc']}\n"
               f"Method {info.method_id}: {mode}; arrival angle {angle:.6f} deg\n")
    (output/"summary.txt").write_text(heading+report.format_text()+"\n", encoding="utf-8")
    with (output/"trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        writer.writerows([t, *v] for t, v in zip(result.trajectory.epochs_s, result.trajectory.states))
    with (output/"burns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch_s", "frame", "dv_x_m_s", "dv_y_m_s", "dv_z_m_s", "magnitude_m_s"])
        writer.writerows([r.command.epoch_s, r.command.frame, *r.command.delta_v_m_s,
                          r.command.magnitude_m_s] for r in result.burn_records)
    if "cw_diagnostics" in metadata:
        write_json(output/"cw_diagnostics.json", metadata["cw_diagnostics"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--method-id", type=int, choices=range(4))
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--initial-altitude-km", type=float, default=INITIAL_ALTITUDE_KM)
    parser.add_argument("--target-altitude-km", type=float, default=TARGET_ALTITUDE_KM)
    parser.add_argument("--inclination-deg", type=float, default=INCLINATION_DEG)
    parser.add_argument("--intermediate-altitude-km", type=float, default=BIELLIPTIC_APOAPSIS_ALTITUDE_KM)
    parser.add_argument("--lambert-angle-deg", type=float, default=LAMBERT_ARRIVAL_ANGLE_DEG)
    parser.add_argument("--lambert-tof-s", type=float, default=LAMBERT_TIME_OF_FLIGHT_S)
    parser.add_argument("--cw-tof-s", type=float, default=CW_TIME_OF_FLIGHT_S)
    parser.add_argument("--cw-correction", action=argparse.BooleanOptionalAction, default=CW_NONLINEAR_CORRECTION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for info in METHODS:
            print(f"{info.method_id}: {info.name} [{info.source}]")
        return 0
    choice = "all" if args.all else args.method_id if args.method_id is not None else METHOD_ID
    infos = list(METHODS) if choice == "all" else [method_info(choice)]
    body = EARTH
    request = TransferRequest(circular_state(body.radius_m+args.initial_altitude_km*1000, body,
                              inclination_rad=np.deg2rad(args.inclination_deg), frame="ECI"),
                              body, body.radius_m+args.target_altitude_km*1000)
    settings = ComparisonSettings(body.radius_m+args.intermediate_altitude_km*1000,
                                  args.lambert_angle_deg, args.lambert_tof_s, args.cw_tof_s)
    output = args.output_dir.resolve()
    metadata = {"schema_version": 2, "run_id": uuid4().hex[:12],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "scenario": "Level 0 manual capability comparison", "output_directory": str(output),
                "central_body": asdict(body), "frame": request.initial_state.frame,
                "initial_altitude_km": args.initial_altitude_km, "target_altitude_km": args.target_altitude_km,
                "initial_radius_m": float(np.linalg.norm(request.initial_state.position_m)),
                "target_radius_m": request.target_radius_m, "inclination_deg": args.inclination_deg,
                "settings": asdict(settings), "cw_nonlinear_correction": args.cw_correction,
                "verification_duration_s": orbital_period_s(request.target_radius_m, body.mu_m3_s2)}
    clear_generated_outputs(output)
    print(f"Run {metadata['run_id']} | {metadata['generated_at_utc']}")
    print(f"Scenario: {args.initial_altitude_km:g} -> {args.target_altitude_km:g} km; inclination {args.inclination_deg:g} deg")
    print(f"Lambert: arrival {args.lambert_angle_deg:g} deg; time "
          +(f"{args.lambert_tof_s:g} s" if args.lambert_tof_s is not None else "Hohmann reference duration (supplied input)"))
    print("CW: "+("CW + two-body correction" if args.cw_correction else "pure CW")
          +"; initially aligned reference on the target orbit")
    heading = "ID  Method          Status   Delta-v (m/s)  Time (min)  Arrival (deg)  Burns"
    print(heading)
    rows, completed, lines = [], [], [heading]
    for info in infos:
        row = {"method_id": info.method_id, "method": info.name, "status": "ERROR",
               "total_delta_v_m_s": None, "transfer_duration_s": None, "arrival_angle_deg": None,
               "burn_count": None, "max_altitude_error_m": None, "note": ""}
        try:
            tool = Level0Tool(info.method_id, settings)
            candidate = tool.propose(request)
            run_metadata = dict(metadata)
            if info.method_id == 3:
                diagnostics = CWLevel0Tool(settings.cw_tof_s).diagnostics(request)
                if args.cw_correction:
                    candidate, trace = correct_cw_candidate(request, candidate)
                else:
                    trace = {"enabled": False, "applied": False, "iterations": 0}
                diagnostics["nonlinear_correction"] = trace
                run_metadata["cw_diagnostics"] = diagnostics
                row["note"] = "CW + two-body correction" if args.cw_correction else "Pure CW; approximation error evaluated under two-body gravity"
            angle = tool.arrival_angle_deg(request, candidate)
            result = execute_candidate(candidate, verification_duration_s=metadata["verification_duration_s"])
            rr, vv = tool.terminal_reference(request)
            report = evaluate_target_orbit(result, request.target_radius_m,
                                           expected_position_m=rr, expected_velocity_m_s=vv)
            row.update(status="PASS" if report.passed else "FAIL",
                       total_delta_v_m_s=report.metrics["total_delta_v_m_s"],
                       transfer_duration_s=candidate.duration_s, arrival_angle_deg=angle,
                       burn_count=len(candidate.burns), max_altitude_error_m=report.metrics["max_altitude_error_m"])
            save_run(output/f"method_{info.method_id}", info, result, report, run_metadata, angle)
            completed.append((info, result, report))
            line = (f"{info.method_id:<3} {info.name:<15} {row['status']:<7} {row['total_delta_v_m_s']:>13.6f} "
                    f"{candidate.duration_s/60:>11.6f} {angle:>14.6f} {len(candidate.burns):>6}")
        except Exception as error:
            row["note"] = f"{type(error).__name__}: {error}"
            directory = output/f"method_{info.method_id}"
            directory.mkdir(exist_ok=True)
            write_json(directory/"error.json", {**metadata, "method": asdict(info), "error": row["note"]})
            line = f"{info.method_id:<3} {info.name:<15} ERROR   {row['note']}"
        rows.append(row)
        lines.append(line)
        print(line)
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_json(output/"comparison.json", {**metadata, "status": status, "methods": rows})
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"comparison.txt").write_text(
        f"Run {metadata['run_id']} | {metadata['generated_at_utc']}\n"
        f"{args.initial_altitude_km:g} -> {args.target_altitude_km:g} km\n"
        +"\n".join(lines)+"\n\n"+json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    if completed and not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from maneuver_planner.plotting import plot_method_comparison, plot_transfer
        stamp = f"Run {metadata['run_id']} | {metadata['generated_at_utc']}"
        for info, result, report in completed:
            figure = plot_transfer(result, request.target_radius_m)
            mode = ("CW + correction" if args.cw_correction else "Pure CW") if info.method_id == 3 else info.name
            figure.suptitle(f"{mode}: {args.initial_altitude_km:g} -> {args.target_altitude_km:g} km | "
                            f"{'PASS' if report.passed else 'FAIL'}\n{stamp}", fontsize=12)
            figure.savefig(output/f"method_{info.method_id}"/"level0_trajectory.png", dpi=150)
            plt.close(figure)
        paths, metrics = plot_method_comparison(completed, request.target_radius_m)
        for figure, name in ((paths, "comparison_trajectories.png"), (metrics, "comparison_metrics.png")):
            figure.suptitle(figure._suptitle.get_text()+"\n"+stamp, fontsize=12)
            figure.savefig(output/name, dpi=160)
            plt.close(figure)
    print(f"\nCurrent results: {output}")
    print("PASS validates each assigned trajectory; it does not establish minimum cost.")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
