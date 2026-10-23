"""Optional plots of already executed results; guidance and dynamics never call this."""

import numpy as np

from .models import ExecutionResult


def plot_transfer(result: ExecutionResult, target_radius_m: float):
    """Return a Matplotlib figure. Saving or displaying it is the caller's choice."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    initial, body = result.candidate.initial_state, result.candidate.body
    times = result.trajectory.epochs_s
    positions = result.trajectory.positions_m
    r1 = np.linalg.norm(initial.position_m)
    p = initial.position_m / r1
    h = np.cross(initial.position_m, initial.velocity_m_s)
    q = np.cross(h / np.linalg.norm(h), p)
    mission = times <= result.terminal_state.epoch_s
    verification = result.verification_trajectory

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        fig, (orbit, altitude) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        fig.suptitle(f"{body.name} circular orbit transfer", fontsize=17, fontweight="bold")
        theta = np.linspace(0, 2*np.pi, 500)
        if body.radius_m > 0:
            orbit.add_patch(Circle((0, 0), body.radius_m / 1000, facecolor="#e7edf4",
                                   edgecolor="#b0becd", linewidth=0.8))
            orbit.text(0, 0, body.name, ha="center", va="center", color="#587088")
        orbit.plot(r1/1000 * np.cos(theta), r1/1000 * np.sin(theta), "--",
                   color="#8997a6", linewidth=1, label="Initial orbit")
        orbit.plot(target_radius_m/1000 * np.cos(theta), target_radius_m/1000 * np.sin(theta),
                   ":", color="#32928a", linewidth=1.6, label="Target orbit")
        orbit.plot(positions[mission] @ p / 1000, positions[mission] @ q / 1000,
                   color="#253f68", linewidth=2, label="Transfer")
        if verification is not None:
            orbit.plot(verification.positions_m @ p / 1000, verification.positions_m @ q / 1000,
                       color="#32928a", alpha=0.7, linewidth=1, label="Verification coast")
        for i, record in enumerate(result.burn_records, 1):
            xy = (np.dot(record.before.position_m, p)/1000, np.dot(record.before.position_m, q)/1000)
            orbit.scatter(*xy, color="#c66032", zorder=5, s=35)
            orbit.annotate(f"Burn {i}", xy, xytext=(0, 12), textcoords="offset points",
                           ha="center", fontsize=9)
        orbit.set_aspect("equal")
        orbit.set_xlabel("Position along initial radial axis (km)")
        orbit.set_ylabel("Position along initial tangential axis (km)")
        orbit.legend(loc="lower left", fontsize=8, framealpha=0.95)
        orbit.grid(alpha=0.15)

        elapsed_minutes = (times - initial.epoch_s) / 60
        height_km = (np.linalg.norm(positions, axis=1) - body.radius_m) / 1000
        altitude.plot(elapsed_minutes, height_km, color="#253f68", linewidth=2)
        altitude.axhline((target_radius_m-body.radius_m)/1000, color="#32928a",
                         linestyle="--", linewidth=1, label="Target altitude")
        arrival_min = (result.terminal_state.epoch_s - initial.epoch_s) / 60
        altitude.axvline(arrival_min, color="#c66032", linestyle=":", linewidth=1.3,
                         label="Transfer complete")
        if verification is not None:
            altitude.axvspan(arrival_min, elapsed_minutes[-1], color="#32928a", alpha=0.06)
        altitude.set_xlabel("Elapsed time (min)")
        altitude.set_ylabel("Altitude above reference radius (km)")
        altitude.set_title(f"Total delta-v: {result.candidate.total_delta_v_m_s:.3f} m/s   |   "
                           f"Transfer: {result.candidate.duration_s/60:.3f} min", fontsize=10)
        altitude.legend(loc="lower right", fontsize=9)
        altitude.grid(alpha=0.2)
    return fig


def plot_method_comparison(runs, target_radius_m: float):
    """Return trajectory and metric figures for successfully executed manual runs.

    runs is a sequence of (MethodInfo, ExecutionResult, EvaluationReport). IDs keep
    their input order; plots do not rank methods or select an objective winner.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if not runs:
        raise ValueError("At least one completed run is needed for a comparison plot.")
    colors = ["#223d65", "#76589a", "#217b78", "#c66032"]
    styles = ["-", "-", "--", "-."]
    short = {0: "Hohmann", 1: "Bielliptic", 2: "Lambert", 3: "CW"}
    first = runs[0][1].candidate.initial_state
    body = runs[0][1].candidate.body
    radial = first.position_m/np.linalg.norm(first.position_m)
    normal = np.cross(first.position_m, first.velocity_m_s)
    tangent = np.cross(normal/np.linalg.norm(normal), radial)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        paths, (orbit, altitude) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        start_altitude = (np.linalg.norm(first.position_m)-body.radius_m)/1000
        target_altitude = (target_radius_m-body.radius_m)/1000
        paths.suptitle(f"{start_altitude:g} km to {target_altitude:g} km: configured capabilities",
                       fontsize=16, fontweight="bold")
        if body.radius_m:
            orbit.add_patch(Circle((0, 0), body.radius_m/1000, color="#e8edf3"))
        angle = np.linspace(0, 2*np.pi, 600)
        orbit.plot(target_radius_m/1000*np.cos(angle), target_radius_m/1000*np.sin(angle),
                   color="#9aabb7", linestyle=":", linewidth=1)
        for info, result, report in runs:
            mission = result.trajectory.epochs_s <= result.terminal_state.epoch_s
            positions = result.trajectory.positions_m[mission]
            elapsed = (result.trajectory.epochs_s[mission]-first.epoch_s)/60
            method_label = short.get(info.method_id, info.name)
            if info.method_id == 3:
                method_label = "CW + correction" if "correction" in result.candidate.tool_name else "Pure CW"
            label = f"{info.method_id}: {method_label}"
            if not report.passed:
                label += " (FAIL)"
            color = colors[info.method_id % len(colors)]
            style = styles[info.method_id % len(styles)]
            orbit.plot(positions@radial/1000, positions@tangent/1000, color=color, linestyle=style,
                       linewidth=2, label=label)
            altitude.plot(elapsed, (np.linalg.norm(positions, axis=1)-body.radius_m)/1000,
                          color=color, linestyle=style, linewidth=2, label=label)
        orbit.set_aspect("equal")
        orbit.set_xlabel("Initial radial axis (km)")
        orbit.set_ylabel("Initial tangential axis (km)")
        orbit.grid(alpha=0.15)
        orbit.set_title("Transfer arcs; endpoint constraints are configurable", fontsize=10)
        altitude.axhline((target_radius_m-body.radius_m)/1000, color="#8a969d", linestyle=":", linewidth=1)
        altitude.set_xlabel("Elapsed transfer time (min)")
        altitude.set_ylabel("Altitude (km)")
        altitude.grid(alpha=0.2)
        altitude.legend(loc="upper right", fontsize=8, framealpha=0.95)
        altitude.set_title("Same target orbit; configured endpoints and times may differ", fontsize=10)

        metrics, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        metrics.suptitle(f"{start_altitude:g} km to {target_altitude:g} km: configured cost and duration (not optimized)",
                         fontsize=16, fontweight="bold")
        labels = []
        for info, result, report in runs:
            name = short[info.method_id]
            if info.method_id == 3:
                name = "CW + corr." if "correction" in result.candidate.tool_name else "Pure CW"
            labels.append(f"{info.method_id}\n{name}"+("" if report.passed else "\nFAIL"))
        bar_colors = [colors[info.method_id % len(colors)] for info, _, _ in runs]
        values = ([report.metrics["total_delta_v_m_s"] for _, _, report in runs],
                  [report.metrics["transfer_duration_s"]/60 for _, _, report in runs])
        for axis, data, ylabel in zip(axes, values, ["Total applied delta-v (m/s)", "Transfer duration (min)"]):
            bars = axis.bar(labels, data, color=bar_colors, width=0.68)
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            axis.set_ylim(0, max(data)*1.18 if max(data) else 1)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
    return paths, metrics
