"""Acceptance test for ROADMAP step 10: the racing environment.

1. The fixed-theta Leap-C MPC driven through the env gives exactly the
   trajectory of ``run_baseline.py --controller leapc-unshifted``; the road
   decoded from the observation equals the MPC's road data along the way.
2. The exit check agrees with a brute-force test on random poses.
3. Gymnasium's checker, the splits, every way an episode ends, and the calls
   mpc-sac makes (train reset, validation rollout, float32 buffer).

Run with the venv activated. Exit code 0 only if everything passes.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from latch13.controllers.leapc_fixed import build_fixed_theta_solver  # noqa: E402
from latch13.simulation.closed_loop import (  # noqa: E402
    constant_speed_progress, initialize_solver, shift_warm_start, shifted_progress,
    solve_with_recovery, stage_parameters,
)
from latch13.simulation.env import (  # noqa: E402
    EnvOptions, RacingEnv, planner_road, planner_state,
)
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
HELD_OUT = ((1300.0, 1850.0), (4140.0, 4391.0))  # test section, validation corners
failed = []


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)
    if not passed:
        failed.append(name)


def drive(env, N, v_init, steps):
    """Fixed-theta MPC through env.step, with the runner's loop functions."""
    generated = ROOT / "results" / "generated" / "leapc" / "check_env" / f"N{N}"
    solver = build_fixed_theta_solver(TUM_CONFIG, generated, N=N, dt=env.dt, shift=False)
    observation, info = env.reset(seed=0, options={"state": [0, 0, 0, v_init, 0, 0, 0, 0]})
    x = info["frenet_state"]
    initialize_solver(solver, x, N, env.dt)
    horizon = constant_speed_progress(x[0], N, env.dt, v_init)
    rows, state_error, road_error = [], 0.0, 0.0
    for _ in range(steps):
        parameters = stage_parameters(env.track, horizon)
        ahead = horizon - x[0]
        if ahead.max() <= env.options.near_distance:  # the observation serves N = 10
            error = np.abs(planner_road(observation, ahead) - parameters).max()
            road_error = max(road_error, float(error))
            state_error = max(state_error, float(np.abs(planner_state(observation)[1:] - x[1:]).max()))
        status, _, _ = solve_with_recovery(solver, x, parameters, N, env.dt)
        u = solver.get(0, "u") if status == 0 else np.full(2, np.nan)
        observation, _, terminated, truncated, info = env.step(u)
        if status != 0:
            break
        x = info["frenet_state"]
        rows.append(np.r_[x[0], info["cartesian_state"], x[1:3], x[7], u])
        if terminated or truncated:
            break
        horizon = shifted_progress(np.array([solver.get(j, "x")[0] for j in range(N + 1)]), x[0])
        shift_warm_start(solver, N)
    return np.array(rows), info["termination"], state_error, road_error


def check_trajectory(track, N, v_init, steps):
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_baseline.py"), "--controller", "leapc-unshifted",
            "--N", str(N), "--v-init", str(v_init), "--steps", str(steps), "--no-plot", "--output", folder,
        ], check=True, capture_output=True)
        with open(Path(folder) / "trajectory.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[1:]
    columns = ["s_m", "X_m", "Y_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_radps", "steering_angle_rad",
               "ey_m", "epsi_rad", "acceleration_mps2", "jerk_mps3", "steering_rate_radps"]
    expected = np.array([[float(row[name]) for name in columns] for row in rows])

    options = EnvOptions(max_steps=None, stop_speed=None)  # the runner has neither
    env = RacingEnv(TUM_CONFIG, track, mode="full", options=options)
    actual, reason, state_error, road_error = drive(env, N, v_init, steps)
    label = f"N={N} v_init={v_init}"
    same = actual.shape == expected.shape
    difference = float(np.abs(actual - expected).max()) if same else np.inf
    check(f"{label}: same trajectory as the runner", same and difference == 0,
          f"{len(actual)} steps, max difference {difference:.1e}, end: {reason}")
    if N == 10:  # the observed horizon is too short for N = 38
        check(f"{label}: observation gives the MPC's state and road", max(state_error, road_error) < 1e-9,
              f"state {state_error:.1e}, road {road_error:.1e}")
        defaults = RacingEnv(TUM_CONFIG, track, mode="full", options=EnvOptions(max_steps=None))
        stopped, reason, _, _ = drive(defaults, N, v_init, steps)
        prefix = np.array_equal(stopped, expected[:len(stopped)])
        check(f"{label}: the 3 m/s stop changes only where the episode ends", prefix,
              f"{len(stopped)} steps, end: {reason}")


def on_track(track, points):
    start = np.concatenate((track.left, track.right))
    end = np.concatenate((np.roll(track.left, -1, axis=0), np.roll(track.right, -1, axis=0)))
    crosses = (start[:, 1] > points[:, 1:]) != (end[:, 1] > points[:, 1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = start[:, 0] + (points[:, 1:] - start[:, 1]) / (end - start)[:, 1] * (end - start)[:, 0]
    return np.count_nonzero(crosses & (x > points[:, :1]), axis=1) % 2 == 1


def check_exit(track, params, count, rng):
    length, width = float(params.veh_length), float(params.veh_width)
    grid = np.linspace(-0.5, 0.5, 41)
    body = np.array([(a * length, b * width) for a in grid for b in (-0.5, 0.5)]
                    + [(a * length, b * width) for a in (-0.5, 0.5) for b in grid])
    tested = wrong = 0
    for _ in range(count):
        # Half the poses in the tight corners, all close to an edge.
        s = rng.uniform(600, 950) if rng.random() < 0.5 else rng.uniform(0, track.length)
        edge = track.edges(s)[rng.integers(2)]
        xy, yaw, _ = track.pose(s)
        x, y = track.cartesian(s, edge + rng.uniform(-3, 3))
        heading = float(yaw) + rng.uniform(-1.5, 1.5)
        margin = track.body_margin(x, y, heading, length, width)
        if abs(margin) < 0.05:
            continue  # finer than the sampling of the brute-force test
        cos, sin = np.cos(heading), np.sin(heading)
        inside = on_track(track, np.array([x, y]) + body @ np.array([[cos, sin], [-sin, cos]])).all()
        tested += 1
        wrong += inside != (margin > 0)
    check("exit check against brute force", tested > count / 2 and wrong == 0,
          f"{wrong} wrong of {tested}")


def check_splits(track, count):
    sections = EnvOptions().sections
    # Starts must stay out of what is held out from them. The validation hairpin overlaps the
    # training split on purpose; the test section is held out from both.
    forbidden = {"train": sections["test"] + sections["val"][:1], "val": sections["test"],
                 "test": sections["train"] + sections["val"]}
    for mode in ("train", "val", "test"):
        env = RacingEnv(TUM_CONFIG, track, mode=mode)
        others = forbidden[mode]
        env.reset(seed=0)
        bad = 0
        for _ in range(count):
            observation, info = env.reset()
            s = info["frenet_state"][0] % track.length
            leaked = any((s - o.start) % track.length < (o.end - o.start) % track.length for o in others)
            bad += (leaked or not env.observation_space.contains(observation)
                    or env.plant.body_margin() < 0 or info["frenet_state"][3] <= env.options.stop_speed)
        check(f"{mode}: valid starts inside the split", bad == 0, f"{bad} bad of {count}")


def check_preview(track):
    env = RacingEnv(TUM_CONFIG, track, mode="train")
    reach = max(env.options.preview) + track.segment_lengths.max()  # interpolation reads one vertex on
    reach += env.speed_max * env.dt  # the last observation lies one step past the section
    seen = 0
    for section in env._sections:
        road = (section.end - section.start) % track.length
        s = (section.start + np.arange(0.0, road + reach, 0.5)) % track.length
        seen += sum(int(np.sum((s >= start) & (s < end))) for start, end in HELD_OUT)
    check("train: preview never reaches a held-out corner", seen == 0)


def check_score(track):
    bonus = {}
    for mode, s in (("train", 3929), ("val", 4429), ("test", 1849)):
        rewards = []
        for limit in (100, 300):
            env = RacingEnv(TUM_CONFIG, track, mode=mode, options=EnvOptions(max_steps=limit))
            env.reset(seed=0, options={"state": [s, 0, 0, 20, 0, 0, 0, 0]})
            rewards.append(env.step(np.zeros(2))[1])
        bonus[mode] = round(rewards[1] - rewards[0], 9)
    check("time bonus only in val and test", bonus == {"train": 0, "val": 200, "test": 200}, str(bonus))


def check_endings(track):
    cruise = [2200, 0, 0, 20, 0, 0, 0, 0]
    still = np.zeros(2)

    def end(state, action=still, mode="full", **options):
        env = RacingEnv(TUM_CONFIG, track, mode=mode, options=EnvOptions(**options))
        env.reset(seed=0, options={"state": state})
        observation, reward, terminated, truncated, info = env.step(action)
        flags = {key: value for key, value in info["task"].items() if key.startswith("end_")}
        flagged = flags.get(f"end_{info['termination']}") == 1 and sum(flags.values()) == 1
        finite = np.isfinite(reward) and env.observation_space.contains(observation) and flagged
        return info["termination"], terminated, truncated, finite

    edge = float(track.edges(2200)[1])
    widest = float(track.s[np.argmax(np.minimum(track.ey_max, -track.ey_min))])  # room to stand sideways
    cases = {
        "mpc_failure": (end(cruise, np.full(2, np.nan)), True),
        "non_finite_state": (end(cruise, np.array([1e200, 1e200])), True),
        "left_track": (end([2200, edge, 0, 20, 0, 0, 0, 0]), True),
        "spun": (end([widest, 0, 1.65, 5, 0, 0, 0, 0]), True),
        "low_speed": (end([2200, 0, 0, 2.9, 0, 0, 0, 0]), True),
        "state_out_of_bounds": (end([2200, 0, 0, 31, 0, 0, 0, 0]), True),
        "section_end": (end([3929, 0, 0, 20, 0, 0, 0, 0], mode="train"), False),
        "time_limit": (end(cruise, max_steps=1), False),
    }
    for expected, ((reason, terminated, truncated, finite), terminal) in cases.items():
        passed = reason == expected and terminated == terminal and truncated != terminal and finite
        check(f"ending {expected}", passed, f"got {reason}, {terminated=}, {truncated=}, {finite=}")


def check_mpc_sac(track):
    from mpc_sac.rl.buffer import ReplayBuffer
    from mpc_sac.utils.gym import seed_env, wrap_env
    from mpc_sac.utils.rollout import episode_rollout

    env = wrap_env(RacingEnv(TUM_CONFIG, track, mode="train"))
    first, _ = seed_env(env, 5, {"mode": "train"})
    second, _ = seed_env(env, 5, {"mode": "train"})
    check("same seed, same start", np.array_equal(first, second))
    buffer = ReplayBuffer(8, "cpu")
    next_observation, reward, terminated, _, _ = env.step(np.zeros(2))
    buffer.put((first, np.zeros(4, dtype=np.float32), reward, next_observation, terminated))
    check("float32 replay buffer", buffer.sample(1)[0].shape == (1, *env.observation_space.shape))

    validation, visited = RacingEnv(TUM_CONFIG, track, mode="val"), []
    sections = validation.options.sections["val"]

    def inside(s):
        return any((s - section.start + 5) % track.length <= (section.end - section.start + 5) % track.length
                   for section in sections)

    def policy(observation):
        visited.append(validation.plant.wrapped_s)
        return np.zeros(2), None, None

    episodes = list(episode_rollout(policy, wrap_env(validation), episodes=5, rng=3))
    check("mpc-sac validation stays in its split",
          len(episodes) == 5 and all(inside(s) for s in visited),
          f"{len(visited)} steps, all inside the validation sections: {all(inside(s) for s in visited)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-long", type=int, default=2500, help="steps of the N = 38 comparison")
    parser.add_argument("--samples", type=int, default=3000, help="poses and starts to sample")
    args = parser.parse_args()

    track = load_track(TRACK_ROOT, "Silverstone", 10.0)
    env = RacingEnv(TUM_CONFIG, track)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_env(env, skip_render_check=True)
            check("Gymnasium env checker", True, "; ".join(sorted({str(w.message) for w in caught})))
        except Exception as error:
            check("Gymnasium env checker", False, repr(error))
    check_exit(track, env.plant.simulator.params, args.samples, np.random.default_rng(0))
    check_splits(track, args.samples)
    check_preview(track)
    check_score(track)
    check_endings(track)
    check_mpc_sac(track)
    for N, v_init, steps in ((10, 10.0, 3000), (10, 12.0, 3000), (10, 15.0, 3000), (38, 10.0, args.steps_long)):
        check_trajectory(track, N, v_init, steps)
    print(f"{len(failed)} failed" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
