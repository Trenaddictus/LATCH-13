"""Post-run inspection using saved numeric data; no acados dependency."""
import json
from pathlib import Path

import numpy as np

from latch13.tracks.f1tenth import Raceline


def load_predictions(path):
    """Read the stream of individual .npy records written during the run."""
    snapshots = []
    with open(path, "rb") as handle:
        while handle.read(1):
            handle.seek(-1, 1)
            snapshots.append(np.load(handle, allow_pickle=False))
    return snapshots


def visualize_run(directory, *, show=True):
    """Save overview.png and open a prediction slider after computation ends.

    Blue is the complete measured path; orange is the saved prediction at the
    selected solve time. Red marks the measured pose at that time. Failed solve
    predictions are explicitly labelled as untrusted solver iterates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    directory = Path(directory)
    data = np.atleast_1d(np.genfromtxt(directory / "trajectory.csv", delimiter=",", names=True))
    summary = json.loads((directory / "summary.json").read_text())
    track = Raceline.from_npz(directory / "track.npz")
    diagnostics = [json.loads(line) for line in (directory / "diagnostics.jsonl").read_text().splitlines()]
    saved = [item for item in diagnostics if item.get("prediction_saved")]
    snapshots = load_predictions(directory / "predictions.npy")
    if len(saved) != len(snapshots):
        raise ValueError("Prediction records and diagnostics do not match")

    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    path, errors, speed, acceleration, controls, timing, margin, slack = axes.flat
    for edge in (track.left, track.right):
        path.plot(*np.vstack((edge, edge[:1])).T, color="0.7", linewidth=0.7)
    path.plot(*np.vstack((track.points, track.points[:1])).T, "k--", linewidth=0.6, label="Raceline")
    path.plot(data["X_m"], data["Y_m"], label="Measured path")
    predicted, = path.plot([], [], color="tab:orange", label="Prediction")
    measured, = path.plot([], [], "ro", markersize=4, label="State at solve")
    path.set(xlabel="X [m]", ylabel="Y [m]", aspect="equal")
    path.legend(fontsize="small")
    times = data["time_s"]
    errors.plot(times, data["ey_m"], label="Lateral error [m]")
    errors.plot(times, data["epsi_rad"], label="Heading error [rad]")
    speed.plot(times, data["vx_mps"], label="Longitudinal speed [m/s]")
    if "reference_speed_mps" in summary:  # runs before ROADMAP step 3 tracked a constant v_ref
        speed.axhline(summary["reference_speed_mps"], color="0.5", linestyle="--", label="Target speed")
    if "speed_limit_mps" in summary:
        speed.axhline(summary["speed_limit_mps"], color="r", linestyle="--", linewidth=0.8,
                      label="Speed limit")
    acceleration.plot(times, data["acceleration_mps2"], label="Acceleration state [m/s²]")
    controls.step(times[1:], data["jerk_mps3"][1:], where="pre", label="Jerk [m/s³]")
    controls.step(times[1:], data["steering_rate_radps"][1:], where="pre", label="Steering rate [rad/s]")
    timing.plot(times[1:], data["solver_time_ms"][1:], label="acados solve [ms]")
    timing.plot(times[1:], data["controller_wall_ms"][1:], label="Controller wall time [ms]")
    if "track_margin_m" in data.dtype.names:  # absent in runs before track-edge constraints
        margin.plot(times, data["track_margin_m"], label="Vehicle body to nearest edge [m]")
        margin.axhline(summary["track_margin_option_m"], color="0.5", linestyle="--",
                       label="MPC safety margin")
        margin.axhline(0, color="r", linewidth=0.8, label="Track edge")
        slack.plot(times[1:], data["max_slack"][1:], label="Max soft-constraint slack over horizon")
    else:
        for ax in (margin, slack):
            ax.text(0.5, 0.5, "not recorded in this run", ha="center", transform=ax.transAxes)
    for ax in (errors, speed, acceleration, controls, timing, margin, slack):
        ax.set_xlabel("Simulation time [s]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")
    fig.suptitle(f"{summary['track']} | N={summary['horizon_N']} | {summary['stop_reason']}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    def update(index):
        index = min(int(index), len(saved) - 1)
        info, snapshot = saved[index], snapshots[index]
        xy = track.cartesian(snapshot[:, 0], snapshot[:, 1])
        predicted.set_data(xy[:, 0], xy[:, 1])
        row = int(np.argmin(abs(times - info["time_s"])))
        measured.set_data([data["X_m"][row]], [data["Y_m"][row]])
        label = "optimized" if info["status"] == 0 else "FAILED: untrusted iterate"
        path.set_title(f"Step {info['step']}, t={info['time_s']:.2f}s | {label}")
        fig.canvas.draw_idle()

    if saved:
        slider = Slider(fig.add_axes((0.18, 0.015, 0.65, 0.02)), "Prediction", 0,
                        max(1, len(saved) - 1), valinit=len(saved) - 1, valstep=1)
        slider.on_changed(update)
        fig._prediction_slider = slider  # Retain widget callbacks for the figure lifetime.
        update(len(saved) - 1)
    fig.savefig(directory / "overview.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
