"""LATCH-13: Frenet MPC with the independent TUM vehicle simulator.

State conventions
-----------------
MPC (Frenet, 7): [s, e_y, e_psi, v_x, v_y, r, delta]
Plant (Cartesian, 7): [X, Y, psi, v_x, v_y, r, delta]
Controls (2): [longitudinal_acceleration, steering_rate]

Visualization uses the original TUM-CONTROL Mode 1 plotting functions,
without instantiating TUM's tightly-coupled Logger class.

Run from the LATCH-13 repository root, after activating the Conda environment
and configuring ACADOS_SOURCE_DIR, LD_LIBRARY_PATH and PYTHONPATH.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from latch13.controllers.ocp import build_mpc
from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator

ROOT = Path(__file__).resolve().parents[1]
TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"


def numeric_csv(path, min_cols):
    """Read numeric CSV rows; tolerate comments, headers, comma/semicolon."""
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = ";" if ";" in line else ","
            parts = [part.strip() for part in line.split(sep)]
            try:
                values = [float(item) for item in parts[:min_cols]]
            except ValueError:
                continue
            if len(values) == min_cols and np.all(np.isfinite(values)):
                rows.append(values)
    if len(rows) < 5:
        raise ValueError(f"Not enough numeric rows in {path}")
    return np.asarray(rows, dtype=float)


def load_track(name, scale):
    """Read the raceline, then construct closed-loop segment geometry."""
    path = TRACK_ROOT / name / f"{name}_raceline.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raceline: {path}")
    data = numeric_csv(path, 7)
    points = data[:, 1:3] * scale
    if np.linalg.norm(points[-1] - points[0]) < 1e-6:
        points = points[:-1]

    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-7):
        raise ValueError("Raceline has duplicate adjacent coordinates")

    s = np.r_[0.0, np.cumsum(lengths[:-1])]
    lap_length = float(lengths.sum())
    dx = np.gradient(points[:, 0], s)
    dy = np.gradient(points[:, 1], s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    kappa = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1e-8
    )
    kappa = gaussian_filter1d(kappa, sigma=3.0, mode="wrap")

    return {
        "points": points,
        "x": points[:, 0],
        "y": points[:, 1],
        "segments": segments,
        "segment_lengths": lengths,
        "s": s,
        "kappa": kappa,
        "L": lap_length,
    }


def curvature(track, progress):
    return float(np.interp(
        progress % track["L"],
        np.r_[track["s"], track["L"]],
        np.r_[track["kappa"], track["kappa"][0]],
    ))


def reference_pose(track, progress):
    """Piecewise-linear Frenet reference X, Y, heading."""
    progress = float(progress % track["L"])
    index = int(np.searchsorted(track["s"], progress, side="right") - 1)
    index = max(0, min(index, len(track["s"]) - 1))
    length = track["segment_lengths"][index]
    alpha = np.clip((progress - track["s"][index]) / length, 0.0, 1.0)
    segment = track["segments"][index]
    xy = track["points"][index] + alpha * segment
    heading = np.arctan2(segment[1], segment[0])
    return float(xy[0]), float(xy[1]), float(heading)


def project(track, X, Y, heading):
    """Project a Cartesian pose onto the closed raceline."""
    p = np.array([X, Y], dtype=float)
    start = track["points"]
    tangent = track["segments"]
    squared = np.maximum(np.einsum("ij,ij->i", tangent, tangent), 1e-12)
    alpha = np.clip(
        np.einsum("ij,ij->i", p - start, tangent) / squared, 0.0, 1.0
    )
    candidates = start + alpha[:, None] * tangent
    index = int(np.argmin(np.sum((candidates - p) ** 2, axis=1)))
    d = tangent[index]
    offset = p - candidates[index]
    length = track["segment_lengths"][index]
    ey = (d[0] * offset[1] - d[1] * offset[0]) / length
    psi_ref = np.arctan2(d[1], d[0])
    epsi = np.arctan2(np.sin(heading - psi_ref), np.cos(heading - psi_ref))
    s = (track["s"][index] + alpha[index] * length) % track["L"]
    return float(s), float(ey), float(epsi)


def load_centerline_boundaries(name, scale):
    """F1TENTH centerline: x, y, width_right, width_left."""
    path = TRACK_ROOT / name / f"{name}_centerline.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing centerline: {path}")
    data = numeric_csv(path, 4)
    center = data[:, :2] * scale
    if np.linalg.norm(center[-1] - center[0]) < 1e-6:
        center = center[:-1]
        data = data[:-1]
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    right = center - (data[:, 2] * scale)[:, None] * normal
    left = center + (data[:, 3] * scale)[:, None] * normal
    return center, left, right


def create_simulator(mpc_dt):
    with open(TUM_CONFIG / "sim_main_params.yaml", encoding="utf-8") as f:
        sim_config = yaml.safe_load(f)
    sim_dt = float(sim_config["Ts"])
    ratio = mpc_dt / sim_dt
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, atol=1e-10):
        raise ValueError(f"MPC dt={mpc_dt} is not a multiple of sim dt={sim_dt}")

    # TUM YAML paths include 'EDGAR/' already; provide parent Config/.
    vehicle = PassengerVehicleSimulator(
        config_path=str(TUM_CONFIG.parent) + "/",
        sim_main_params=sim_config,
        Ts=sim_dt,
    )
    return vehicle, sim_dt, substeps


class TUMLiveMode1:
    """Invoke TUM's original Mode 1 plot functions with LATCH-13 data."""

    def __init__(self, track, track_name, scale, initial_state,
                 simulator, N, dt, speed):
        # TUM exposes these functions in Logging_Plotting's module scope.
        from Utils.Logging_Plotting import initLiveVisuMode1, updateLiveVisuMode1
        import matplotlib.pyplot as plt

        self.plt = plt
        self.tum_update = updateLiveVisuMode1
        self.track, self.N, self.dt, self.speed = track, N, dt, speed
        self.t0 = time.perf_counter()
        self.length = float(simulator.params.veh_length)
        self.width = float(simulator.params.veh_width)
        self.xwidth = 100.0
        self.ywidth = 100.0
        self.capacity = 2048
        self.CiLX = np.zeros((self.capacity, 7))
        self.DisturbedX = np.zeros((self.capacity, 7))
        self.simU = np.zeros((self.capacity, 2))
        self.CiLX[0] = initial_state
        self.DisturbedX[0] = initial_state

        center, left, right = load_centerline_boundaries(track_name, scale)
        pose = np.array([
            initial_state[0], initial_state[1], initial_state[2],
            np.hypot(initial_state[3], initial_state[4]),
        ])
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        (
            self.MPC_pred_line,
            self.Ref_traj_line,
            self.Disturbed_traj_line,
            self.ax,
            self.center_line,
            self.inner_line,
            self.outer_line,
            self.line,
            self.dot,
            self.text,
        ) = initLiveVisuMode1(
            self.ax, self.CiLX, pose,
            self.xwidth, self.ywidth,
            center[:, 0], left[:, 0], right[:, 0],
            center[:, 1], left[:, 1], right[:, 1],
            self.length, self.width,
        )
        self.fig.suptitle("LATCH-13 | TUM-CONTROL Mode 1 | Frenet MPC")
        plt.show(block=False)

    def store(self, step, cartesian_state, control):
        """Record EVERY step, even if we only redraw occasionally."""
        required = step + 2
        if required > self.capacity:
            new_capacity = max(required, self.capacity * 2)
            for name in ("CiLX", "DisturbedX", "simU"):
                old = getattr(self, name)
                new = np.zeros((new_capacity, old.shape[1]))
                new[:len(old)] = old
                setattr(self, name, new)
            self.capacity = new_capacity
        self.CiLX[step + 1] = cartesian_state
        self.DisturbedX[step + 1] = cartesian_state
        self.simU[step] = control

    def redraw(self, step, solver, progress):
        prediction = np.zeros((self.N + 1, 8))
        reference = np.zeros((self.N + 1, 4))
        for j in range(self.N + 1):
            state = np.asarray(solver.get(j, "x"), dtype=float).reshape(-1)
            s, ey, epsi, vx, vy, yaw_rate, steering = state
            Xref, Yref, psi_ref = reference_pose(self.track, s)
            prediction[j, :7] = [
                Xref - ey * np.sin(psi_ref),
                Yref + ey * np.cos(psi_ref),
                psi_ref + epsi, vx, vy, yaw_rate, steering,
            ]
            if j < self.N:
                prediction[j, 7] = float(np.asarray(
                    solver.get(j, "u"), dtype=float
                ).reshape(-1)[0])
            elif self.N > 0:
                prediction[j, 7] = prediction[j - 1, 7]
            xr, yr, yaw_ref = reference_pose(
                self.track, progress + j * self.dt * self.speed
            )
            reference[j] = [xr, yr, yaw_ref, self.speed]

        ref_index = int(np.searchsorted(
            self.track["s"], progress % self.track["L"], side="right"
        ) - 1)
        self.tum_update(
            step + 1,
            time.perf_counter() - self.t0,
            self.CiLX,
            self.simU,
            self.DisturbedX,
            {
                "pos_x": reference[:, 0],
                "pos_y": reference[:, 1],
                "ref_yaw": reference[:, 2],
                "ref_v": reference[:, 3],
            },
            prediction,
            self.Ref_traj_line,
            self.MPC_pred_line,
            self.Disturbed_traj_line,
            self.ax,
            self.xwidth,
            self.ywidth,
            self.line,
            self.dot,
            self.text,
            self.length,
            self.width,
            current_ref_idx=ref_index,
        )
        self.plt.pause(0.001)
        return self.plt.fignum_exists(self.fig.number)

    def save_and_close(self, path):
        if self.plt.fignum_exists(self.fig.number):
            self.fig.savefig(path, dpi=150)
            self.plt.close(self.fig)


def run(args):
    track = load_track(args.track, args.scale)
    output = ROOT / "results" / args.track / f"N{args.N}"
    output.mkdir(parents=True, exist_ok=True)

    first_segment = track["segments"][0]
    yaw0 = float(np.arctan2(first_segment[1], first_segment[0]))
    # Physical plant: [X, Y, psi, vx, vy, r, delta]
    xc = np.array([
        track["x"][0], track["y"][0], yaw0, args.speed, 0, 0, 0
    ], dtype=float)
    # Frenet MPC: [s, ey, epsi, vx, vy, r, delta]
    xf = np.array([0, 0, 0, args.speed, 0, 0, 0], dtype=float)

    solver = build_mpc(
        TUM_CONFIG, xf, N=args.N, dt=args.dt,
        v_ref=args.speed, output_dir=output / "solver",
    )
    for j in range(args.N + 1):
        guess = xf.copy()
        guess[0] += j * args.dt * args.speed
        solver.set(j, "x", guess)
        if j < args.N:
            solver.set(j, "u", np.zeros(2))

    simulator, sim_dt, substeps = create_simulator(args.dt)
    print("Plant: TUM-CONTROL PassengerVehicleSimulator (7-state Cartesian)")
    print("Controller: LATCH-13 Frenet MPC (7-state, acceleration input)")
    print(f"Track: {args.track}, length: {track['L']:.1f} m")
    print(f"MPC dt: {args.dt}, plant dt: {sim_dt}, substeps: {substeps}")

    viewer = None
    if not args.no_plot:
        viewer = TUMLiveMode1(
            track, args.track, args.scale, xc, simulator,
            args.N, args.dt, args.speed,
        )

    keys = [
        "step", "time_s", "lap", "s_m", "X_m", "Y_m", "yaw_rad",
        "ey_m", "epsi_rad", "vx_mps", "vy_mps", "yaw_rate_radps",
        "steering_angle_rad", "acceleration_mps2", "steering_rate_radps",
        "solver_time_ms", "controller_wall_ms", "solver_status",
    ]
    samples = 0
    progress = 0.0
    previous_wrapped_s = 0.0
    ey_sq_sum = 0.0
    max_ey = 0.0
    solve_times = []
    failures = 0
    stop_reason = "step_limit"
    t0 = time.perf_counter()
    csv_path = output / "trajectory.csv"

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            iteration = 0
            while args.steps == 0 or iteration < args.steps:
                loop_start = time.perf_counter()
                solver.constraints_set(0, "lbx", xf)
                solver.constraints_set(0, "ubx", xf)

                # Initial approximation: track curvature at expected progress.
                for j in range(args.N + 1):
                    expected_s = xf[0] + j * args.dt * args.speed
                    solver.set(j, "p", np.array([curvature(track, expected_s)]))

                status = int(solver.solve())
                wall_ms = (time.perf_counter() - loop_start) * 1000.0
                if status != 0:
                    failures += 1
                    stop_reason = f"solver_status_{status}"
                    print(f"Solver failed at step {iteration}: status {status}")
                    try:
                        print("Residuals:", solver.get_residuals())
                        print("NLP iterations:", solver.get_stats("nlp_iter"))
                    except Exception:
                        pass
                    break

                u0 = np.asarray(solver.get(0, "u"), dtype=float).reshape(-1)
                if u0.size != 2:
                    raise ValueError("Expected [acceleration, steering_rate]")
                solve_ms = float(np.asarray(
                    solver.get_stats("time_tot")
                ).reshape(-1)[0]) * 1000.0

                # Original TUM simulator: same input ordering [a_x, delta_dot].
                sim_input = u0.copy()
                for _ in range(substeps):
                    xc = np.asarray(
                        simulator.simulator_step(xc, sim_input), dtype=float
                    ).reshape(-1)
                if xc.size != 7 or not np.all(np.isfinite(xc)):
                    stop_reason = "invalid_plant_state"
                    print("Invalid vehicle state; stopping")
                    break

                wrapped_s, ey, epsi = project(track, xc[0], xc[1], xc[2])
                L = track["L"]
                ds = ((wrapped_s - previous_wrapped_s + L / 2) % L) - L / 2
                progress += ds
                previous_wrapped_s = wrapped_s
                xf = np.array([progress, ey, epsi, *xc[3:]], dtype=float)

                samples += 1
                ey_sq_sum += ey * ey
                max_ey = max(max_ey, abs(ey))
                solve_times.append(solve_ms)
                lap = max(0, int(np.floor(progress / L)))

                writer.writerow({
                    "step": iteration,
                    "time_s": samples * args.dt,
                    "lap": lap,
                    "s_m": progress,
                    "X_m": xc[0],
                    "Y_m": xc[1],
                    "yaw_rad": xc[2],
                    "ey_m": ey,
                    "epsi_rad": epsi,
                    "vx_mps": xc[3],
                    "vy_mps": xc[4],
                    "yaw_rate_radps": xc[5],
                    "steering_angle_rad": xc[6],
                    "acceleration_mps2": u0[0],
                    "steering_rate_radps": u0[1],
                    "solver_time_ms": solve_ms,
                    "controller_wall_ms": wall_ms,
                    "solver_status": status,
                })
                if samples % 50 == 0:
                    handle.flush()
                    print(f"step={iteration} lap={lap} ey={ey:.3f}m "
                          f"solve={solve_ms:.2f}ms")

                if viewer is not None:
                    viewer.store(iteration, xc, u0)
                    if iteration % args.plot_every == 0:
                        if not viewer.redraw(iteration, solver, progress):
                            stop_reason = "plot_closed"
                            break

                if abs(ey) > args.max_error:
                    stop_reason = "excessive_tracking_error"
                    print(f"Lateral error {ey:.2f} m exceeds {args.max_error} m")
                    break

                if args.realtime:
                    elapsed = time.perf_counter() - loop_start
                    time.sleep(max(0.0, args.dt - elapsed))

                iteration += 1
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; saving results")
    finally:
        solve_times = np.asarray(solve_times, dtype=float)
        summary = {
            "track": args.track,
            "horizon_N": args.N,
            "mpc_dt_s": args.dt,
            "sim_dt_s": sim_dt,
            "reference_speed_mps": args.speed,
            "track_scale": args.scale,
            "track_length_m": track["L"],
            "simulation_type": "TUM-CONTROL PassengerVehicleSimulator",
            "plotter": "TUM-CONTROL Mode 1" if viewer else "disabled",
            "samples": samples,
            "simulated_time_s": samples * args.dt,
            "completed_laps": max(0, int(np.floor(progress / track["L"]))),
            "stop_reason": stop_reason,
            "lateral_rmse_m": float(np.sqrt(ey_sq_sum / samples)) if samples else None,
            "max_abs_lateral_error_m": float(max_ey),
            "mean_solver_ms": float(np.mean(solve_times)) if samples else None,
            "p95_solver_ms": float(np.percentile(solve_times, 95)) if samples else None,
            "max_solver_ms": float(np.max(solve_times)) if samples else None,
            "solver_failures": failures,
            "wall_elapsed_s": time.perf_counter() - t0,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if viewer is not None:
            viewer.save_and_close(output / "trajectory.png")
        print(json.dumps(summary, indent=2))
        print(f"Results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--N", type=int, default=38)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=0,
                        help="0 means run until interrupted")
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--max-error", type=float, default=8.0)
    args = parser.parse_args()
    if args.N < 1 or args.dt <= 0 or args.speed <= 0 or args.scale <= 0:
        parser.error("N, dt, speed and scale must be positive")
    if args.steps < 0 or args.plot_every < 1:
        parser.error("steps must be >= 0; plot-every must be >= 1")
    run(args)
