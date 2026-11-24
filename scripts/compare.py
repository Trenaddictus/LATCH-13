"""Compare N = 38, N = 10 with the default terminal cost, and learned N = 10 controllers, from the same start.

The controllers drive one after another through the same environment as
RL training (same plant, exit check and endings), optionally with the live view
(--live 1|2). Afterwards it saves a comparison plot, a summary table and one CSV
per run in results/compare/<time>/.

* N = 38: the Leap-C MPC at the default terminal cost, with the warm start of
  RL training (``--controller leapc-unshifted`` in run_baseline.py).
* N = 10 default: the RL planner at the initial theta.
* N = 10 learned: the RL planner, with theta from a trained policy (checkpoint),
  one per run given to --learned (constant or road-dependent theta).

    python scripts/compare.py --learned results/rl/step12_constant_seed2 --live
    python scripts/compare.py --start lap --live \
        --learned results/rl/step12b_constant_seed3 results/rl/step13b_road_seed0

--start test (default) drives the held-out test section (corners 5+6) from
s = 1300 m at 20 m/s; --start lap a full lap from s = 0 at 10 m/s. Each learned
run uses its best validation checkpoint (val_log.csv), or RUN:STEP for another.
Run with the venv activated.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from latch13.controllers.leapc_fixed import build_fixed_theta_solver  # noqa: E402
from latch13.controllers.ocp import ConstraintOptions, default_stage_weights, weights_tag  # noqa: E402
from latch13.controllers.planner import RacingPlanner, RacingPlannerConfig  # noqa: E402
from latch13.learning.policy import best_checkpoint, load_actor, theta_function  # noqa: E402
from latch13.simulation.closed_loop import (  # noqa: E402
    constant_speed_progress, initialize_solver, shifted_progress, solve_with_recovery, stage_parameters,
)
from latch13.simulation.env import EnvOptions, RacingEnv  # noqa: E402
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
GENERATED = ROOT / "results" / "generated" / "leapc"
STARTS = {"test": ("test", 1300.0, 20.0, 450), "lap": ("full", 0.0, 10.0, 3200)}  # mode, s, v, steps


class SolverController:
    """Leap-C MPC at a fixed theta, driven like run_baseline.py --controller leapc-unshifted."""

    def __init__(self, track, N, dt=0.08):
        weights = default_stage_weights(N)
        folder = GENERATED / "closed_loop" / f"N{N}_dt{dt}_m{ConstraintOptions().track_margin}_w{weights_tag(weights)}"
        self.solver = build_fixed_theta_solver(TUM_CONFIG, folder, N=N, dt=dt, shift=False)
        self.track, self.N, self.dt = track, N, dt

    def reset(self, state):
        self.solver.reset()
        initialize_solver(self.solver, state, self.N, self.dt)
        self.horizon = constant_speed_progress(state[0], self.N, self.dt, state[3])

    def act(self, observation, state):
        status, _, _ = solve_with_recovery(self.solver, state, stage_parameters(self.track, self.horizon),
                                           self.N, self.dt)
        plan = np.array([self.solver.get(j, "x") for j in range(self.N + 1)])
        self.next_plan = plan[:, 0]
        return (self.solver.get(0, "u") if status == 0 else np.full(2, np.nan)), plan

    def moved(self, state):
        self.horizon = shifted_progress(self.next_plan, state[0])


class PlannerController:
    """The RL planner, with theta from ``theta(observation)``."""

    def __init__(self, planner, theta):
        self.planner, self.theta = planner, theta

    def reset(self, state):
        self.ctx = None

    def act(self, observation, state):
        with torch.no_grad():
            self.ctx, u0, x, _, _ = self.planner(torch.as_tensor(observation[None]),
                                                 params=self.theta(observation), ctx=self.ctx)
        plan = x[0].numpy().copy()
        plan[:, 0] += state[0]  # the planner works with s = 0 at the car
        return u0[0].numpy(), plan

    def moved(self, state):
        pass


def drive(name, controller, env, start, steps, lap, args):
    observation, info = env.reset(seed=0, options={"state": start})
    state = info["frenet_state"]
    start_s = state[0]
    controller.reset(state)
    live = None
    if args.live:
        from latch13.visualization.live import LIVE_VIEWS
        with open(TUM_CONFIG / "veh_params_pred.yaml", encoding="utf-8") as handle:
            limits = yaml.safe_load(handle)
        live = LIVE_VIEWS[args.live](
            env.track, info["cartesian_state"], state, env.plant.simulator.params, limits,
            dt=env.dt, rate=args.live_rate, title=f"{name} | {args.start}",
        )
    rows, end = [], "step_limit"
    try:
        for step in range(steps):
            begin = time.perf_counter()
            u, plan = controller.act(observation, state)
            milliseconds = (time.perf_counter() - begin) * 1000
            if live is not None and step % args.live_every == 0:
                if not live.draw(step * env.dt, state, plan, milliseconds):
                    end = "live_view_closed"
                    break
            observation, reward, terminated, truncated, info = env.step(u)
            state = info["frenet_state"]
            if live is not None:
                live.record(info["cartesian_state"], state, planned_speed=float(plan[1, 3]))
            rows.append({"step": step, "time_s": (step + 1) * env.dt, "s_m": state[0], "ey_m": state[1],
                         "epsi_rad": state[2], "vx_mps": state[3], "jerk": u[0], "steering_rate": u[1],
                         "reward": reward, "margin_m": env.plant.body_margin(), "controller_ms": milliseconds})
            if terminated or truncated:
                end = info["termination"]
                break
            if lap and state[0] - start[0] >= env.track.length:
                end = "lap"
                break
            controller.moved(state)
    finally:
        if live is not None:
            live.close()
    run = pd.DataFrame(rows)
    run.attrs["end"], run.attrs["start_s"] = end, float(start_s)
    return run


def summary(name, run):
    milliseconds = run.controller_ms.to_numpy()
    return {
        "controller": name, "end": run.attrs["end"], "steps": len(run), "time_s": round(len(run) * 0.08, 2),
        "distance_m": round(run.s_m.iloc[-1] - run.attrs["start_s"], 1),
        "lap_time_s": round(len(run) * 0.08, 2) if run.attrs["end"] == "lap" else None,
        "mean_speed_mps": round(run.vx_mps.mean(), 2), "max_speed_mps": round(run.vx_mps.max(), 2),
        "rms_ey_m": round(float(np.sqrt(np.mean(run.ey_m**2))), 4), "max_abs_ey_m": round(run.ey_m.abs().max(), 3),
        "min_margin_m": round(run.margin_m.min(), 3), "return": round(run.reward.sum(), 1),
        "median_ms": round(float(np.median(milliseconds)), 2), "p95_ms": round(float(np.percentile(milliseconds, 95)), 2),
    }


def plot(runs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for name, run in runs.items():
        label = f"{name} ({run.attrs['end']})"
        axes[0].plot(run.s_m, run.vx_mps, label=label)
        axes[1].plot(run.s_m, run.ey_m, label=label)
        axes[2].plot(run.s_m, run.time_s, label=label)
        axes[3].plot(run.s_m, run.margin_m, label=label)
    for ax, ylabel in zip(axes, ("speed [m/s]", "ey [m]", "time [s]", "body to edge [m]")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[3].axhline(0, color="r", linewidth=0.8)
    axes[-1].set_xlabel("s [m] (lap position for --start test)")
    axes[0].legend(fontsize="small")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--learned", nargs="+", metavar="RUN[:STEP]",
                        help="one or more training run directories (results/rl/...), each driven as its own "
                             "controller; RUN:STEP picks a checkpoint (default: best validation). Omit to "
                             "compare the two fixed controllers only")
    parser.add_argument("--step", type=int, help="checkpoint step, with a single --learned run")
    parser.add_argument("--start", choices=sorted(STARTS), default="test")
    parser.add_argument("--steps", type=int, help="step limit; default 450 (test) or 3200 (lap)")
    parser.add_argument("--live", nargs="?", type=int, choices=(1, 2), const=1, metavar="MODE",
                        help="live TUM-CONTROL view of each run: 1 = track (default), 2 = with panels")
    parser.add_argument("--live-every", type=int, default=1, help="redraw the live view every N steps")
    parser.add_argument("--live-rate", type=float, help="pace the live view to this multiple of real time")
    parser.add_argument("--output", help="output directory; default results/compare/<time>")
    args = parser.parse_args()

    mode, s0, v0, default_steps = STARTS[args.start]
    steps = args.steps or default_steps
    track = load_track(TRACK_ROOT, "Silverstone", 10.0)
    env = RacingEnv(TUM_CONFIG, track, mode=mode, options=EnvOptions(max_steps=steps))
    with open(TUM_CONFIG / "veh_params_pred.yaml", encoding="utf-8") as handle:
        wheelbase = float(yaml.safe_load(handle)["l"])
    kappa = float(track.curvature(s0))  # steady cornering at the start, as the env's own starts
    start = [s0, 0.0, 0.0, v0, 0.0, v0 * kappa, wheelbase * kappa, 0.0]
    output = Path(args.output) if args.output else ROOT / "results" / "compare" / datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    planner = RacingPlanner(TUM_CONFIG, GENERATED / "planner" / f"N10_w{weights_tag(default_stage_weights(10))}",
                            RacingPlannerConfig())
    initial = {name: torch.as_tensor(value[None]) for name, value in planner.default_param(None).items()}
    controllers = {"N38 default": SolverController(track, 38), "N10 default": PlannerController(
        planner, lambda observation: initial)}
    notes = []
    if args.step is not None and len(args.learned or []) != 1:
        parser.error("--step needs exactly one --learned run; use RUN:STEP for several")
    if args.learned:
        from leapc_lab.planner import ControllerFromPlanner
    for entry in args.learned or []:
        path, _, step_text = entry.rpartition(":") if entry.rpartition(":")[2].isdigit() else (entry, "", "")
        run_dir = Path(path).resolve()
        if not (run_dir / "run_config.json").is_file():
            parser.error(f"{path} is not a training run (no run_config.json)")
        step = int(step_text) if step_text else args.step
        step, score = (step, None) if step else best_checkpoint(run_dir)
        actor = load_actor(run_dir, step, env.observation_space, env.action_space,
                           ControllerFromPlanner(planner))
        name = f"N10 {run_dir.name}" if len(args.learned) > 1 else "N10 learned"
        controllers[name] = PlannerController(planner, theta_function(actor))
        notes.append(f"{run_dir.name} at step {step}" + (f" (validation {score:.1f})" if score else ""))
        print(f"{name}: {notes[-1]}")
    note = ("learned: " + "; ".join(notes)) if notes else ""

    runs = {}
    for name, controller in controllers.items():
        print(f"Driving {name} ...", flush=True)
        runs[name] = drive(name, controller, env, start, steps, args.start == "lap", args)
        runs[name].to_csv(output / f"{name.replace(' ', '_')}.csv", index=False)
    table = pd.DataFrame([summary(name, run) for name, run in runs.items()])
    table.to_csv(output / "summary.csv", index=False)
    (output / "config.json").write_text(json.dumps({**vars(args), "start_state": start, "note": note},
                                                   indent=2), encoding="utf-8")
    plot(runs, output / "comparison.png", f"Silverstone, start {args.start} (s = {s0:g} m, {v0:g} m/s). {note}")
    print(table.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
