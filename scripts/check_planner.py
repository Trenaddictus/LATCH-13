"""Acceptance test for ROADMAP step 11, planner part.

1. Driven only from env observations, the planner at the default theta acts
   like ``run_baseline.py --controller leapc-unshifted``: the same controls in
   the runner's own states, and in its own closed loop the same ending,
   distance and RMS ey.
2. Through mpc-sac's validation rollout, which keeps the planner context from
   one episode to the next, every episode starts cold and gives the same
   result as with a fresh context.
3. A failed solve gives a non-finite action, which ends the episode as an MPC
   failure; a batch of observations gives the same actions as one at a time.
   N = 38 is refused, because the observation describes only 30 m of road.

The training run itself is tested with ``scripts/train_rl.py --steps 3000``.
Run with the venv activated. Exit code 0 only if everything passes.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from leapc_lab.planner import ControllerFromPlanner  # noqa: E402
from mpc_sac.utils.gym import wrap_env  # noqa: E402
from mpc_sac.utils.rollout import episode_rollout  # noqa: E402

from latch13.controllers.leapc_fixed import phase1_theta  # noqa: E402
from latch13.controllers.planner import RacingPlanner, RacingPlannerConfig  # noqa: E402
from latch13.simulation.env import EnvOptions, RacingEnv  # noqa: E402
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
COLUMNS = ["s_m", "ey_m", "epsi_rad", "vx_mps", "vy_mps", "yaw_rate_radps", "steering_angle_rad",
           "acceleration_mps2"]
failed = []


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)
    if not passed:
        failed.append(name)


def theta_tensors(theta):
    return {name: torch.as_tensor(np.asarray(value)[None], dtype=torch.float64)
            for name, value in theta.items()}


def runner_run(N, v_init, steps):
    """States and applied controls of ``run_baseline.py --controller leapc-unshifted``."""
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_baseline.py"), "--controller", "leapc-unshifted",
            "--N", str(N), "--v-init", str(v_init), "--steps", str(steps), "--no-plot", "--output", folder,
        ], check=True, capture_output=True)
        with open(Path(folder) / "trajectory.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    states = np.array([[float(row[name]) for name in COLUMNS] for row in rows])
    controls = np.array([[float(row["jerk_mps3"]), float(row["steering_rate_radps"])] for row in rows[1:]])
    return states, controls  # states[k] is the state before controls[k]; row 0 is the start


def check_against_runner(track, planner, N, v_init, steps):
    states, controls = runner_run(N, v_init, steps)
    theta = theta_tensors(phase1_theta())
    env = RacingEnv(TUM_CONFIG, track, mode="full", options=EnvOptions(max_steps=None, stop_speed=None))
    label = f"N={N} v_init={v_init}"

    # 1. Same situations: the runner's own states, the planner's own warm start carried along.
    # The closed loop cannot amplify differences here, so this isolates the planner.
    ctx, differences = None, []
    with torch.no_grad():
        for state, control in zip(states[:-1], controls):
            observation, _ = env.reset(seed=0, options={"state": state})
            ctx, u0, _, _, _ = planner(torch.as_tensor(observation[None]), params=theta, ctx=ctx)
            differences.append(np.abs(u0[0].numpy() - control))
    # Relative to the control range. The SQP stops at a stationarity tolerance of 1e-3, so a
    # warm start that differs slightly can end a little elsewhere; seen at single steps in braking
    # zones (largest 1.4e-4 of the jerk range). These limits were set after seeing that.
    relative = np.array(differences) / (env.action_space.high - env.action_space.low)
    typical, worst = np.percentile(relative, 99.9, axis=0), relative.max(axis=0)
    check(f"{label}: same controls as the runner along its trajectory ({len(controls)} steps)",
          np.all(typical < 1e-4) and np.all(worst < 1e-3),
          f"99.9 % of steps within {typical.max():.1e} of the range, worst {worst.max():.1e}")

    # 2. Its own closed loop. Small differences (warm start rebased by the predicted instead of
    # the measured progress) grow over a long run, as differences in the start state do
    # (ROADMAP Step 9), so compare the whole run by its metrics.
    observation, _ = env.reset(seed=0, options={"state": [0, 0, 0, v_init, 0, 0, 0, 0]})
    ctx, driven, reason = None, [], None
    with torch.no_grad():
        for _ in range(steps):
            ctx, u0, _, _, _ = planner(torch.as_tensor(observation[None]), params=theta, ctx=ctx)
            observation, _, terminated, truncated, info = env.step(u0[0].numpy())
            if not np.all(np.isfinite(u0.numpy())):
                reason = "mpc_failure"
                break
            driven.append(info["frenet_state"])
            if terminated or truncated:
                reason = info["termination"]
                break
    actual, expected = np.array(driven), states[1:]
    distance = [run[-1, 0] - run[0, 0] for run in (actual, expected)]
    rms = [np.sqrt(np.mean(run[:, 1] ** 2)) for run in (actual, expected)]
    check(f"{label}: own closed loop like the runner's",
          len(actual) == len(expected) and abs(distance[0] / distance[1] - 1) < 5e-3
          and abs(rms[0] / rms[1] - 1) < 0.05,
          f"{len(actual)} vs {len(expected)} steps, end {reason}; distance {distance[0]:.1f} vs "
          f"{distance[1]:.1f} m, RMS ey {rms[0]:.4f} vs {rms[1]:.4f} m")


def check_validation(track, planner, episodes):
    """mpc-sac's validate(): one policy closure, its context kept across episodes."""
    controller = ControllerFromPlanner(planner)
    theta = theta_tensors(planner.default_param(None))

    def run(fresh_per_episode):
        env = wrap_env(RacingEnv(TUM_CONFIG, track, mode="val"))
        state = {"ctx": None}

        def policy(observation):
            with torch.no_grad():
                ctx, action = controller(torch.as_tensor(observation[None]), theta, ctx=state["ctx"])
            state["ctx"] = ctx
            return action[0].numpy(), ctx, None

        results = []
        before = planner.cold_starts
        for episode in range(episodes):
            if fresh_per_episode:
                state["ctx"] = None
            rollout = episode_rollout(policy, env, episodes=1, rng=episode)
            stats, _ = next(rollout)
            results.append((stats["score"], stats["length"]))
        return results, planner.cold_starts - before

    kept, kept_cold = run(fresh_per_episode=False)
    fresh, fresh_cold = run(fresh_per_episode=True)
    check("validation: context kept across episodes starts every episode cold",
          kept_cold == episodes, f"{kept_cold} cold starts in {episodes} episodes")
    check("validation: same results as a fresh context per episode", kept == fresh,
          f"scores {[round(score, 1) for score, _ in kept]}")


def check_failure_and_batch(track, planner):
    env = RacingEnv(TUM_CONFIG, track, mode="full")
    # Braking at the hard 2 m/s floor: the jerk limit cannot stop the deceleration in time.
    observation, _ = env.reset(seed=0, options={"state": [100, 0, 0, 2.05, 0, 0, 0, -3.5]})
    theta = theta_tensors(planner.default_param(None))
    with torch.no_grad():
        ctx, u0, _, _, _ = planner(torch.as_tensor(observation[None]), params=theta)
    _, _, terminated, _, info = env.step(u0[0].numpy())
    check("failed solve ends the episode as an MPC failure",
          int(ctx.status[0]) != 0 and terminated and info["termination"] == "mpc_failure",
          f"status {int(ctx.status[0])}, end {info['termination']}")

    starts = [[s, 0.2, 0.01, v, 0, 0, 0, 0] for s, v in ((300, 24), (700, 15), (2000, 20))]
    observations = []
    for start in starts:
        observation, _ = env.reset(seed=0, options={"state": start})
        observations.append(observation)
    batch = torch.as_tensor(np.array(observations))
    with torch.no_grad():
        theta_batch = {name: value.expand(len(starts), -1) for name, value in theta.items()}
        _, together, _, _, _ = planner(batch, params=theta_batch)
        alone = torch.cat([planner(batch[i:i + 1], params=theta)[1] for i in range(len(starts))])
    difference = float((together - alone).abs().max())
    check("a batch of observations gives the same actions as one at a time", difference < 1e-9,
          f"max difference {difference:.1e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3, help="validation episodes")
    args = parser.parse_args()

    track = load_track(TRACK_ROOT, "Silverstone", 10.0)
    generated = ROOT / "results" / "generated" / "leapc" / "check_planner"
    planner = RacingPlanner(TUM_CONFIG, generated / "N10", RacingPlannerConfig(N=10, n_batch_init=4))
    check_failure_and_batch(track, planner)
    check_validation(track, planner, args.episodes)
    for v_init in (10.0, 12.0, 15.0):
        check_against_runner(track, planner, 10, v_init, 3000)
    try:
        RacingPlanner(TUM_CONFIG, generated / "N38", RacingPlannerConfig(N=38))
        check("N = 38 is refused (its horizon is longer than the observed road)", False)
    except ValueError as error:
        check("N = 38 is refused (its horizon is longer than the observed road)", True, str(error))
    print(f"{len(failed)} failed" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
