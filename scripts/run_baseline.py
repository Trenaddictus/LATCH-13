"""Run acados Frenet MPC, save results, then inspect them offline.

MPC state: [s, ey, epsi, vx, vy, r, delta, a]. Inputs: [jerk, steering_rate].
The independent TUM plant uses seven Cartesian states and [a, steering_rate].
There is no reference speed: the MPC rewards progress (--progress-weight), and
--v-init only sets the start speed and the cold-start guess. No plotting occurs during optimization or plant integration. --live [1|2] opts
in to TUM-CONTROL's Mode 1 or Mode 2 live view, drawn outside solver timing.
--controller leapc runs the same loop with the Leap-C MPC at the fixed Phase 1
terminal cost; leapc-unshifted also uses the warm start of the RL training.
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


def max_slack(solver, N):
    """Largest soft-constraint slack over nodes 1..N (node 0 has none)."""
    values = [np.r_[solver.get(j, "sl"), solver.get(j, "su")] for j in range(1, N + 1)]
    return float(max(np.max(v) if v.size else 0.0 for v in values))


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
    from latch13.simulation.closed_loop import (
        constant_speed_progress, initialize_solver, prediction_snapshot, shift_warm_start,
        shifted_progress, solve_with_recovery, solver_diagnostics, stage_parameters,
    )
    from latch13.simulation.plant import ClosedLoopPlant
    from latch13.tracks.f1tenth import load_track

    track = load_track(TRACK_ROOT, args.track, args.scale)
    output = Path(args.output).resolve() if args.output else (
        ROOT / "results" / args.track / f"N{args.N}" /
        datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output.mkdir(parents=True, exist_ok=True)
    # Save geometry so replay needs neither track CSVs nor the solver.
    track.to_npz(output / "track.npz")
    constraints = ConstraintOptions(track_margin=args.track_margin)
    stage_weights = StageCostWeights(progress=args.progress_weight)
    (output / "config.json").write_text(json.dumps({
        **vars(args), "solver_options": asdict(SolverOptions()),
        "stage_weights": asdict(stage_weights),
        "constraint_options": asdict(constraints),
    }, indent=2), encoding="utf-8")

    plant = ClosedLoopPlant(TUM_CONFIG, track, args.dt)
    plant.reset([0, 0, 0, args.v_init, 0, 0, 0, 0])
    xc, xf = plant.cartesian, plant.frenet
    terminal = zero_terminal_cost if args.terminal_cost == "zero" else None
    if args.controller == "acados":
        solver = build_mpc(
            TUM_CONFIG, xf, N=args.N, dt=args.dt, output_dir=output / "solver",
            stage_weights=stage_weights, terminal_cost=terminal, constraints=constraints,
        )
    else:
        from latch13.controllers.leapc_fixed import build_fixed_theta_solver
        # Stable directory, so compiled code is reused; acados rebuilds it if the OCP changes.
        generated = (ROOT / "results" / "generated" / "leapc" / "closed_loop" /
                     f"N{args.N}_dt{args.dt}_q{args.progress_weight}_m{args.track_margin}")
        solver = build_fixed_theta_solver(
            TUM_CONFIG, generated, N=args.N, dt=args.dt, shift=args.controller == "leapc",
            stage_weights=stage_weights, constraints=constraints,
        )
    initialize_solver(solver, xf, args.N, args.dt)
    simulator, sim_dt = plant.simulator, plant.sim_dt
    print(f"Eight-state Frenet MPC: N={args.N}, horizon={args.N * args.dt:.2f}s")
    print(f"Plant dt={sim_dt}s, MPC dt={args.dt}s; plotting after simulation")
    print(f"Results: {output}")
    live = None
    if args.live:
        from latch13.visualization.live import LIVE_VIEWS
        # Same vehicle file as the OCP bounds; Mode 2 normalises by these limits.
        with open(TUM_CONFIG / "veh_params_pred.yaml", encoding="utf-8") as f:
            limits = yaml.safe_load(f)
        live = LIVE_VIEWS[args.live](
            track, xc, xf, simulator.params, limits, dt=args.dt, rate=args.live_rate,
            title=f"{args.track} | N={args.N} | progress weight {args.progress_weight}",
        )

    fields = [
        "step", "time_s", "lap", "s_m", "X_m", "Y_m", "yaw_rad", "ey_m", "epsi_rad",
        "vx_mps", "vy_mps", "yaw_rate_radps", "steering_angle_rad", "acceleration_mps2",
        "jerk_mps3", "steering_rate_radps", "solver_time_ms", "controller_wall_ms", "solver_status",
        "track_margin_m", "max_slack",
    ]
    margins, slacks, speeds = [], [], []
    samples, progress, failures = 0, 0.0, 0
    lap_time = None
    # Stage progress for the parameter lookup: constant-speed guess first,
    # then the previous solution's predicted s, shifted by one stage.
    horizon_s = constant_speed_progress(xf[0], args.N, args.dt, args.v_init)
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
                0, 0, 0, 0, -1, plant.body_margin(), 0,
            ])))
            while args.steps == 0 or samples < args.steps:
                loop_start = time.perf_counter()
                parameters = stage_parameters(track, horizon_s)
                status, attempts, solve_ms = solve_with_recovery(
                    solver, xf, parameters, args.N, args.dt,
                )
                if status == 0 and len(attempts) > 1:
                    recovered_solves += 1
                    print(f"Recovered solver at step {samples} after cold restart")
                wall_ms = (time.perf_counter() - loop_start) * 1000
                slack = max_slack(solver, args.N)
                info = {"step": samples, "time_s": samples * args.dt,
                        "status": status, "frenet_state": xf.tolist(), "max_slack": slack,
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
                if live is not None and samples % args.live_every == 0:
                    # Before the warm-start shift, so the prediction matches xc.
                    snapshot = prediction_snapshot(solver, args.N, parameters)
                    if not live.draw(samples * args.dt, xf, snapshot, solve_ms):
                        stop_reason = "live_view_closed"
                        break

                u0 = np.asarray(solver.get(0, "u"), dtype=float).reshape(2)
                if not np.all(np.isfinite(u0)):
                    raise ValueError("Nonfinite optimized control")
                plant.step(u0)
                xc, xf = plant.cartesian, plant.frenet
                progress, ey, epsi = (float(value) for value in xf[:3])
                acceleration, L = float(xf[7]), track.length
                predicted_s = np.array([solver.get(j, "x")[0] for j in range(args.N + 1)])
                horizon_s = shifted_progress(predicted_s, xf[0])
                if live is not None:
                    live.record(xc, xf, planned_speed=float(solver.get(1, "x")[3]))
                shift_warm_start(solver, args.N)
                samples += 1
                speeds.append(xc[3])
                if lap_time is None and progress >= L:
                    lap_time = samples * args.dt
                ey_sq_sum += ey**2
                max_ey = max(max_ey, abs(ey))
                solve_times.append(solve_ms)
                margins.append(plant.body_margin())
                slacks.append(slack)
                writer.writerow(dict(zip(fields, [
                    samples - 1, samples * args.dt, max(0, int(progress // L)), progress,
                    *xc[:3], ey, epsi, *xc[3:], acceleration, *u0, solve_ms, wall_ms, status,
                    margins[-1], slack,
                ])))
                if samples % 50 == 0:
                    handle.flush()
                    print(f"step={samples} s={progress:.0f}m vx={xc[3]:.1f}m/s ey={ey:.3f}m "
                          f"solve={solve_ms:.2f}ms")
                if margins[-1] < 0 and not args.allow_off_track:
                    # Vehicle body crossed a track edge (ROADMAP D3).
                    stop_reason = "left_track"
                    print(f"Left the track at step {samples - 1}, s={progress:.1f} m: "
                          f"body {-margins[-1]:.2f} m beyond the edge")
                    break
                if args.max_error is not None and abs(ey) > args.max_error:
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
        if live is not None:
            live.close()
        summary = {
            "track": args.track, "horizon_N": args.N, "mpc_dt_s": args.dt,
            "sim_dt_s": sim_dt, "progress_weight": args.progress_weight,
            "initial_speed_mps": args.v_init, "speed_limit_mps": constraints.speed_max,
            "controller": args.controller, "leapc_retries": getattr(solver, "retries", None),
            "terminal_cost": args.terminal_cost, "track_scale": args.scale,
            "track_length_m": track.length, "samples": samples,
            "simulated_time_s": samples * args.dt,
            "completed_laps": max(0, int(progress // track.length)),
            "lap_time_s": lap_time,
            "mean_speed_mps": float(np.mean(speeds)) if samples else None,
            "max_speed_mps": float(np.max(speeds)) if samples else None,
            "stop_reason": stop_reason, "solver_failures": failures,
            "recovered_solves": recovered_solves,
            "lateral_rmse_m": float(np.sqrt(ey_sq_sum / samples)) if samples else None,
            "max_abs_lateral_error_m": max_ey,
            "mean_solver_ms": float(np.mean(solve_times)) if samples else None,
            "p95_solver_ms": float(np.percentile(solve_times, 95)) if samples else None,
            "max_solver_ms": float(np.max(solve_times)) if samples else None,
            "track_margin_option_m": constraints.track_margin,
            "min_track_margin_m": float(np.min(margins)) if samples else None,
            "track_violation_steps": int(np.sum(np.asarray(margins) < 0)),
            "max_slack": float(np.max(slacks)) if samples else None,
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
    parser.add_argument("--v-init", type=float, default=10.0,
                        help="initial speed of the car and the first guess [m/s]; not a reference")
    parser.add_argument("--progress-weight", type=float, default=1.0,
                        help="stage-cost weight q_s on the progress reward -s_dot")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1000,
                        help="simulation steps; 0 runs until Ctrl+C, then opens visualization")
    parser.add_argument("--plot-every", "--prediction-every", dest="plot_every", type=int, default=5,
                        help="save a prediction snapshot every N solves; failures always saved")
    parser.add_argument("--no-plot", action="store_true", help="save data without opening post-run plots")
    parser.add_argument("--live", nargs="?", type=int, choices=(1, 2), const=1, metavar="MODE",
                        help="live TUM-CONTROL view during simulation: 1 = track (default), "
                             "2 = track, speed, deviation and g-g; closing it stops the run")
    parser.add_argument("--live-every", type=int, default=1,
                        help="redraw the live view every N solves")
    parser.add_argument("--live-rate", type=float,
                        help="pace the live view to this multiple of real time (1 = real time); "
                             "default runs as fast as possible")
    parser.add_argument("--max-error", type=float,
                        help="optional stop when |ey| from the raceline exceeds this [m]; "
                             "off by default, the track-edge stop applies")
    parser.add_argument("--track-margin", type=float, default=0.3,
                        help="safety margin between vehicle body and track edge in the MPC [m]")
    parser.add_argument("--allow-off-track", action="store_true",
                        help="keep running after the vehicle body crosses a track edge")
    parser.add_argument("--terminal-cost", choices=("quadratic", "zero"), default="quadratic")
    parser.add_argument("--controller", choices=("acados", "leapc", "leapc-unshifted"), default="acados",
                        help="acados = Phase 1 solver; leapc = Leap-C MPC, fixed theta, shifted warm "
                             "start; leapc-unshifted = same with the warm start used in RL training")
    parser.add_argument("--output", help="explicit output directory; files inside are overwritten")
    parser.add_argument("--replay", help="visualize a saved run directory without running MPC")
    args = parser.parse_args()
    positive = [args.dt, args.v_init, args.scale] + ([args.max_error] if args.max_error is not None else [])
    if args.N < 1 or not np.all(np.isfinite(positive)) or min(positive) <= 0:
        parser.error("N, dt, v-init, scale and max-error must be finite and positive")
    if not (np.isfinite(args.progress_weight) and args.progress_weight >= 0):
        parser.error("progress-weight must be finite and nonnegative")
    if args.controller != "acados" and args.terminal_cost != "quadratic":
        parser.error("--terminal-cost zero is only available with --controller acados")
    if args.steps < 0 or args.plot_every < 1 or args.live_every < 1:
        parser.error("steps must be >= 0; prediction-every and live-every must be >= 1")
    if not (np.isfinite(args.track_margin) and args.track_margin >= 0):
        parser.error("track-margin must be finite and nonnegative")
    if args.live_rate is not None and not (np.isfinite(args.live_rate) and args.live_rate > 0):
        parser.error("live-rate must be finite and positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.replay:
        from latch13.visualization.offline import visualize_run
        visualize_run(Path(args.replay))
    else:
        configure_acados_environment()
        run(args)
