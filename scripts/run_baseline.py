"""Run acados Frenet MPC, save results, then inspect them offline.

MPC state: [s, ey, epsi, vx, vy, r, delta, a]. Inputs: [jerk, steering_rate].
The independent TUM plant uses seven Cartesian states and [a, steering_rate].
No plotting occurs during optimization or plant integration.
Use --replay RUN_DIRECTORY to inspect a saved run without rebuilding acados.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"


def configure_acados_environment():
    """Discover installed acados and restart once if its loader path is missing.

    Linux reads LD_LIBRARY_PATH at process startup, so changing os.environ
    alone does not fix ctypes loading transitive acados dependencies. Explicit
    ACADOS_SOURCE_DIR takes precedence over discovery from acados_template.
    Replay never calls this function.
    """
    from importlib.util import find_spec

    configured = os.environ.get("ACADOS_SOURCE_DIR")
    if configured:
        source = Path(configured).expanduser().resolve()
    else:
        spec = find_spec("acados_template")
        if spec is None or spec.origin is None:
            raise RuntimeError("Install acados_template in the active Python environment")
        source = next((parent for parent in Path(spec.origin).resolve().parents
                       if (parent / "lib" / "libacados.so").is_file()), None)
        if source is None:
            raise RuntimeError("Cannot discover acados; set ACADOS_SOURCE_DIR to its installation directory")
    library_dir = source / "lib"
    if not (library_dir / "libacados.so").is_file():
        raise FileNotFoundError(f"Missing acados shared library: {library_dir / 'libacados.so'}")

    os.environ["ACADOS_SOURCE_DIR"] = str(source)
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if str(library_dir) not in paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(library_dir)] + [path for path in paths if path]
        )
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


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
    from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator

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


def advance_plant(simulator, cartesian_state, acceleration, control, sim_dt, substeps):
    """Apply constant jerk using a midpoint acceleration at each plant substep.

    Acceleration is an ideal commanded actuator state: a_dot = jerk. TUM's
    seven-state plant has no acceleration actuator state of its own. Midpoint
    sampling approximates the ramp inside each substep; no clipping changes the
    optimized input. Return the endpoint acceleration for the next MPC state.
    """
    jerk, steering_rate = np.asarray(control, dtype=float).reshape(2)
    state = np.asarray(cartesian_state, dtype=float).copy()
    for _ in range(substeps):
        midpoint_acceleration = acceleration + 0.5 * sim_dt * jerk
        state = np.asarray(simulator.simulator_step(
            state, np.array([midpoint_acceleration, steering_rate])
        ), dtype=float).reshape(-1)
        acceleration += sim_dt * jerk
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError("Invalid Cartesian plant state")
    return state, float(acceleration)


def initialize_solver(solver, xf, N, dt, speed):
    for j in range(N + 1):
        guess = xf.copy()
        guess[0] += j * dt * speed
        solver.set(j, "x", guess)
        if j < N:
            solver.set(j, "u", np.zeros(2))


def shift_warm_start(solver, N):
    """Move the last optimized trajectory forward by one controller interval."""
    states = [solver.get(j, "x") for j in range(1, N + 1)]
    controls = [solver.get(j, "u") for j in range(1, N)]
    controls.append(solver.get(N - 1, "u"))
    for j in range(N):
        solver.set(j, "x", states[j])
        solver.set(j, "u", controls[j])
    solver.set(N, "x", states[-1])


def prediction_snapshot(solver, N, parameters):
    """Rows contain eight states, two controls, and curvature; terminal u is NaN."""
    snapshot = np.full((N + 1, 11), np.nan)
    for j in range(N + 1):
        snapshot[j, :8] = np.asarray(solver.get(j, "x")).reshape(8)
        snapshot[j, 10] = parameters[j]
        if j < N:
            snapshot[j, 8:10] = np.asarray(solver.get(j, "u")).reshape(2)
    return snapshot


def solver_diagnostics(solver):
    """Best-effort diagnostics must not hide the original failure."""
    result = {}
    for name in ("time_tot", "nlp_iter", "residuals"):
        try:
            value = solver.get_residuals() if name == "residuals" else solver.get_stats(name)
            result[name] = np.asarray(value).tolist()
        except Exception as error:
            result[name] = {"unavailable": str(error)}
    return result


def solve_with_recovery(solver, xf, parameters, N, dt, speed):
    """Solve once, then try one cold restart at the SAME measured state.

    Record rejected iterates before reset. No failed solution is applied to the
    plant. This follows TUM's reinitialization approach, but requires a successful
    retry rather than continuing with the failed solve's first control.
    """
    attempts = []
    total_solver_ms = 0.0
    for attempt in range(2):
        if attempt:
            solver.reset()
            initialize_solver(solver, xf, N, dt, speed)
        solver.constraints_set(0, "lbx", xf)
        solver.constraints_set(0, "ubx", xf)
        for j, kappa in enumerate(parameters):
            solver.set(j, "p", np.array([kappa]))
        status = int(solver.solve())
        details = solver_diagnostics(solver)
        total_solver_ms += float(np.asarray(details["time_tot"]).reshape(-1)[0]) * 1000
        details["status"] = status
        if status != 0:
            details["rejected_prediction"] = prediction_snapshot(solver, N, parameters).tolist()
        attempts.append(details)
        if status == 0:
            break
    return status, attempts, total_solver_ms


def run(args):
    from dataclasses import asdict
    from datetime import datetime

    # TUM-CONTROL uses top-level imports such as Prediction_Models and
    # Vehicle_Simulator. Resolve its root relative to this script so running
    # the baseline does not depend on a shell-specific PYTHONPATH setting.
    tum_root = ROOT / "external" / "TUM-CONTROL"
    if not (tum_root / "Prediction_Models").is_dir():
        raise FileNotFoundError(f"Missing TUM-CONTROL prediction models: {tum_root}")
    if str(tum_root) not in sys.path:
        sys.path.insert(0, str(tum_root))

    from latch13.controllers.ocp import (
        ConstraintOptions, SolverOptions, StageCostWeights, build_mpc, zero_terminal_cost,
    )

    track = load_track(args.track, args.scale)
    output = Path(args.output).resolve() if args.output else (
        ROOT / "results" / args.track / f"N{args.N}" /
        datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output.mkdir(parents=True, exist_ok=True)
    # Save geometry so replay needs neither track CSVs nor the solver.
    center, left, right = load_centerline_boundaries(args.track, args.scale)
    np.savez(output / "track.npz", **track, center=center, left=left, right=right)
    (output / "config.json").write_text(json.dumps({
        **vars(args), "solver_options": asdict(SolverOptions()),
        "stage_weights": asdict(StageCostWeights()),
        "constraint_options": asdict(ConstraintOptions()),
    }, indent=2), encoding="utf-8")

    yaw0 = float(np.arctan2(*track["segments"][0][::-1]))
    xc = np.array([track["x"][0], track["y"][0], yaw0, args.speed, 0, 0, 0.])
    xf = np.array([0, 0, 0, args.speed, 0, 0, 0, 0.])
    terminal = zero_terminal_cost if args.terminal_cost == "zero" else None
    solver = build_mpc(
        TUM_CONFIG, xf, N=args.N, dt=args.dt, v_ref=args.speed,
        output_dir=output / "solver", terminal_cost=terminal,
    )
    initialize_solver(solver, xf, args.N, args.dt, args.speed)
    simulator, sim_dt, substeps = create_simulator(args.dt)
    print(f"Eight-state Frenet MPC: N={args.N}, horizon={args.N * args.dt:.2f}s")
    print(f"Plant dt={sim_dt}s, MPC dt={args.dt}s; plotting after simulation")
    print(f"Results: {output}")

    fields = [
        "step", "time_s", "lap", "s_m", "X_m", "Y_m", "yaw_rad", "ey_m", "epsi_rad",
        "vx_mps", "vy_mps", "yaw_rate_radps", "steering_angle_rad", "acceleration_mps2",
        "jerk_mps3", "steering_rate_radps", "solver_time_ms", "controller_wall_ms", "solver_status",
    ]
    samples, progress, previous_s, failures = 0, 0.0, 0.0, 0
    recovered_solves = 0
    ey_sq_sum, max_ey = 0.0, 0.0
    solve_times = []
    stop_reason = "step_limit"
    t0 = time.perf_counter()
    try:
        with open(output / "trajectory.csv", "w", newline="", encoding="utf-8") as handle, \
             open(output / "predictions.npy", "wb") as predictions, \
             open(output / "diagnostics.jsonl", "w", encoding="utf-8") as diagnostics:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(zip(fields, [
                -1, 0, 0, 0, *xc[:3], 0, 0, *xc[3:], 0,
                0, 0, 0, 0, -1,
            ])))
            while args.steps == 0 or samples < args.steps:
                loop_start = time.perf_counter()
                parameters = [curvature(track, xf[0] + j * args.dt * args.speed)
                              for j in range(args.N + 1)]
                status, attempts, solve_ms = solve_with_recovery(
                    solver, xf, parameters, args.N, args.dt, args.speed,
                )
                if status == 0 and len(attempts) > 1:
                    recovered_solves += 1
                    print(f"Recovered solver at step {samples} after cold restart")
                wall_ms = (time.perf_counter() - loop_start) * 1000
                info = {"step": samples, "time_s": samples * args.dt,
                        "status": status, "frenet_state": xf.tolist(),
                        "attempts": attempts, "total_solver_ms": solve_ms,
                        **solver_diagnostics(solver)}
                if status != 0 or samples % args.plot_every == 0 or samples + 1 == args.steps:
                    np.save(predictions, prediction_snapshot(solver, args.N, parameters), allow_pickle=False)
                    info["prediction_saved"] = True
                diagnostics.write(json.dumps(info) + "\n")
                diagnostics.flush()
                if status != 0:
                    failures += 1
                    stop_reason = f"solver_status_{status}"
                    print(f"Solver failed at step {samples}: status {status}")
                    break

                u0 = np.asarray(solver.get(0, "u"), dtype=float).reshape(2)
                if not np.all(np.isfinite(u0)):
                    raise ValueError("Nonfinite optimized control")
                xc, acceleration = advance_plant(simulator, xc, xf[7], u0, sim_dt, substeps)
                wrapped_s, ey, epsi = project(track, *xc[:3])
                L = track["L"]
                progress += ((wrapped_s - previous_s + L / 2) % L) - L / 2
                previous_s = wrapped_s
                xf = np.array([progress, ey, epsi, *xc[3:], acceleration])
                shift_warm_start(solver, args.N)
                samples += 1
                ey_sq_sum += ey**2
                max_ey = max(max_ey, abs(ey))
                solve_times.append(solve_ms)
                writer.writerow(dict(zip(fields, [
                    samples - 1, samples * args.dt, max(0, int(progress // L)), progress,
                    *xc[:3], ey, epsi, *xc[3:], acceleration, *u0, solve_ms, wall_ms, status,
                ])))
                if samples % 50 == 0:
                    handle.flush()
                    print(f"step={samples} ey={ey:.3f}m solve={solve_ms:.2f}ms")
                if abs(ey) > args.max_error:
                    stop_reason = "excessive_tracking_error"
                    break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; saving results before visualization")
    except Exception as error:
        stop_reason = "exception"
        (output / "failure.json").write_text(json.dumps({
            "step": samples, "error": repr(error), "frenet_state": xf.tolist(),
            "cartesian_state": xc.tolist(), **solver_diagnostics(solver),
        }, indent=2), encoding="utf-8")
        raise
    finally:
        summary = {
            "track": args.track, "horizon_N": args.N, "mpc_dt_s": args.dt,
            "sim_dt_s": sim_dt, "reference_speed_mps": args.speed,
            "terminal_cost": args.terminal_cost, "track_scale": args.scale,
            "track_length_m": track["L"], "samples": samples,
            "simulated_time_s": samples * args.dt,
            "completed_laps": max(0, int(progress // track["L"])),
            "stop_reason": stop_reason, "solver_failures": failures,
            "recovered_solves": recovered_solves,
            "lateral_rmse_m": float(np.sqrt(ey_sq_sum / samples)) if samples else None,
            "max_abs_lateral_error_m": max_ey,
            "mean_solver_ms": float(np.mean(solve_times)) if samples else None,
            "p95_solver_ms": float(np.percentile(solve_times, 95)) if samples else None,
            "max_solver_ms": float(np.max(solve_times)) if samples else None,
            "wall_elapsed_s": time.perf_counter() - t0,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Results saved to: {output}")
    if not args.no_plot:
        from latch13.visualization.offline import visualize_run
        visualize_run(output)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--N", type=int, default=38)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1000,
                        help="simulation steps; 0 runs until Ctrl+C, then opens visualization")
    parser.add_argument("--plot-every", "--prediction-every", dest="plot_every", type=int, default=5,
                        help="save a prediction snapshot every N solves; failures always saved")
    parser.add_argument("--no-plot", action="store_true", help="save data without opening post-run plots")
    parser.add_argument("--max-error", type=float, default=8.0)
    parser.add_argument("--terminal-cost", choices=("quadratic", "zero"), default="quadratic")
    parser.add_argument("--output", help="explicit output directory; files inside are overwritten")
    parser.add_argument("--replay", help="visualize a saved run directory without running MPC")
    args = parser.parse_args()
    if args.N < 1 or not np.all(np.isfinite([args.dt, args.speed, args.scale, args.max_error])) or min(args.dt, args.speed, args.scale, args.max_error) <= 0:
        parser.error("N, dt, speed, scale and max-error must be finite and positive")
    if args.steps < 0 or args.plot_every < 1:
        parser.error("steps must be >= 0; prediction-every must be >= 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.replay:
        from latch13.visualization.offline import visualize_run
        visualize_run(Path(args.replay))
    else:
        configure_acados_environment()
        run(args)
