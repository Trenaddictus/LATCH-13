"""Gymnasium racing environment for learning the terminal cost (ROADMAP step 10).

The action is the MPC's [jerk, steering_rate]. One step is one MPC interval of
the TUM plant, with the same plant, projection and exit check as the runner.

Observation, with distances measured from the car (no absolute s):

* state (7): ey, epsi, vx, vy, r, delta, a
* road in the horizon: the raceline's own vertices, starting just behind the
  car, as [distance, curvature, ey_min, ey_max]. Interpolating them gives
  exactly the MPC's road data, which a regular grid does not.
* road preview: [curvature, ey_min, ey_max] at fixed distances beyond that.

``planner_state`` and ``planner_road`` decode what an MPC planner needs.

The trainer passes only the action to ``step``, so a failed MPC solve has to be
signalled with a non-finite action; the episode then ends with ``mpc_failure``.
The split is chosen in the constructor, because mpc-sac passes no mode when it
resets the validation environment. In val and test, finishing a section adds
the time left before the time limit as progress at full speed, so that the
summed score sees time. The training reward has no such bonus.

Reward weights, failure penalty (100), time limit, split and validation score
were decided on 2026-11-24 (docs/PROBLEM_SPEC.md section 10). The start
distribution is still a proposal.

As with ``ocp.py``, external/TUM-CONTROL must be on sys.path.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from latch13.controllers.ocp import ConstraintOptions
from latch13.simulation.plant import ClosedLoopPlant, PlantStateError

MODES = ("train", "val", "test", "full")
# Every way an episode ends; info["task"] has one 0/1 flag per entry ("end_<reason>"), which mpc-sac
# averages into the share of episodes that ended that way.
TERMINATIONS = ("left_track", "spun", "low_speed", "state_out_of_bounds", "non_finite_state", "mpc_failure")
TRUNCATIONS = ("section_end", "time_limit")
# ey, epsi, vx, vy, r, delta, a. Wide enough for every state that may continue:
# a state outside ends the episode, so clipping never alters the MPC's x0.
STATE_LOW = np.array([-22.0, -np.pi / 2, 0.0, -10.0, -3.0, -0.7, -4.0])
STATE_HIGH = np.array([22.0, np.pi / 2, 30.0, 10.0, 3.0, 0.7, 3.0])


@dataclass(frozen=True)
class RewardOptions:
    progress: float = 1.0
    # 1.0 since 2026-11-24 (was 0.1): with 0.1, 3 m/s more speed outweighed 1 m of tracking error
    # about 15:1 and the learned controllers drove up to 10 m wide (ROADMAP Step 13).
    tracking: float = 1.0
    heading: float = 100.0  # 0.1 rad counts like 1 m
    lateral_excess: float = 1.0
    # 100 since 2026-11-24 (was 10): continuing is worth about 85 in discounted reward, so 10 added
    # little to a crash, and the pilot traded crashes for speed (ROADMAP Step 12).
    failure_penalty: float = 100.0


@dataclass(frozen=True)
class StartOptions:
    """Random start states, slowed where a corner ahead requires it.

    The speed cap uses only the track and the acceleration limit, not the
    N = 38 speed profile (D1). It is never seen by the MPC or the reward (D2).
    """

    ey: float = 0.5
    epsi: float = 0.05
    speed_min: float = 10.0
    speed_max: float = 25.0
    lateral_fraction: float = 0.85  # share of the lateral limit usable at the start
    braking: float = 1.5  # deceleration assumed available before a corner


@dataclass(frozen=True)
class Section:
    """Track section from ``start`` to ``end``; may wrap around the lap."""

    start: float
    end: float
    start_window: float  # episodes start within this distance after ``start``


def silverstone_sections():
    """Split of docs/PROBLEM_SPEC.md section 7, for Silverstone scaled x10.

    Test is corners 5+6 (held out). Validation is corners 12+13 (held out from
    training) and, since 2026-11-24, also the hairpin (corners 3+4, s = 450-1000 m),
    so that checkpoint selection sees the hardest corner. The hairpin is also
    in the training split; only the test section is strictly held out.
    Training ends far enough before the held-out corners that its 200 m
    preview never reaches them.
    """
    return {
        "train": (Section(1850.0, 3930.0, 1980.0), Section(4430.0, 1095.0, 1030.0)),
        "val": (Section(3930.0, 4430.0, 60.0), Section(450.0, 1000.0, 60.0)),
        "test": (Section(1300.0, 1850.0, 120.0),),
    }


@dataclass(frozen=True)
class EnvOptions:
    max_steps: int | None = 375  # 30 s; None for no time limit
    # Env only. The MPC's model is unusable below about 1.86 m/s and its hard
    # limit is 2 m/s; stop before solves fail there. None switches it off.
    stop_speed: float | None = 3.0
    near_vertices: int = 20
    near_distance: float = 30.0  # must cover the planner's horizon
    preview: tuple = tuple(float(d) for d in range(40, 201, 10))
    reward: RewardOptions = field(default_factory=RewardOptions)
    start: StartOptions = field(default_factory=StartOptions)
    sections: dict = field(default_factory=silverstone_sections)


def planner_state(observation):
    """MPC state with s = 0 at the car."""
    return np.r_[0.0, observation[:7]]


def planner_road(observation, distance, near_vertices=EnvOptions.near_vertices):
    """[curvature, ey_min, ey_max] at distances ahead of the car."""
    near = np.reshape(observation[7:7 + 4 * near_vertices], (near_vertices, 4))
    return np.stack([np.interp(distance, near[:, 0], near[:, i]) for i in (1, 2, 3)], axis=-1)


class RacingEnv(gym.Env):
    """Closed-loop racing episodes on one split of one track.

    ``reset(options={"state": x})`` starts at a given MPC state. In mode
    "full" episodes start anywhere and end only by failure or the time limit.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_dir, track, mode="train", dt=0.08, options=None, constraints=None,
                 episode_log=None):
        """``episode_log``: optional CSV file that gets one row per finished episode (where it
        started and ended, how it ended, its length and return)."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
        self.track, self.mode, self.dt = track, mode, float(dt)
        self.options = options if options is not None else EnvOptions()
        constraints = constraints if constraints is not None else ConstraintOptions()
        self.speed_max = constraints.speed_max
        with open(Path(config_dir) / "veh_params_pred.yaml", encoding="utf-8") as handle:
            self._vehicle = yaml.safe_load(handle)

        self.plant = ClosedLoopPlant(config_dir, track, dt)
        self._sections = self._validated_sections()
        self.action_space = gym.spaces.Box(
            np.array([self._vehicle["jerk_min"], self._vehicle["delta_f_dot_min"]], dtype=float),
            np.array([self._vehicle["jerk_max"], self._vehicle["delta_f_dot_max"]], dtype=float),
            dtype=np.float64,
        )
        self.observation_space = self._build_observation_space()
        self.render_mode = None
        self._steps, self._section_left = 0, np.inf
        self._observation = None
        self._episode_log = Path(episode_log) if episode_log is not None else None
        self._start, self._return = None, 0.0

    def _validated_sections(self):
        length = self.track.length
        if self.mode == "full":
            return (Section(0.0, length, length),)
        sections = tuple(self.options.sections[self.mode])
        for section in sections:
            if not (0 <= section.start < length and 0 <= section.end < length):
                raise ValueError(f"{section} is not on this {length:.0f} m track")
            if not 0 < section.start_window < (section.end - section.start) % length:
                raise ValueError(f"Start window of {section} must lie inside the section")
        return sections

    def _build_observation_space(self):
        track, count = self.track, self.options.near_vertices
        spacing = np.tile(track.segment_lengths, 2)
        covered = np.convolve(spacing, np.ones(count - 2), mode="valid").min()
        if covered < self.options.near_distance:
            raise ValueError(f"{count} vertices cover only {covered:.1f} m; increase near_vertices")
        road_low = np.array([track.kappa.min(), track.ey_min.min(), track.ey_max.min()])
        road_high = np.array([track.kappa.max(), track.ey_min.max(), track.ey_max.max()])
        # The same bounds on every track of this scale, so networks carry over.
        limit_low, limit_high = np.array([-0.05, -22.0, 0.0]), np.array([0.05, 0.0, 22.0])
        if np.any(road_low < limit_low) or np.any(road_high > limit_high):
            raise ValueError("Track curvature or edges exceed the observation bounds")
        longest = track.segment_lengths.max()
        near_low = np.r_[-longest, limit_low]
        near_high = np.r_[longest * count, limit_high]
        far = len(self.options.preview)
        low = np.r_[STATE_LOW, np.tile(near_low, count), np.tile(limit_low, far)]
        high = np.r_[STATE_HIGH, np.tile(near_high, count), np.tile(limit_high, far)]
        return gym.spaces.Box(low, high, dtype=np.float64)

    def _observe(self):
        track, s = self.track, self.plant.wrapped_s % self.track.length
        first = np.searchsorted(track.s, s, side="right") - 1
        laps, index = np.divmod(first + np.arange(self.options.near_vertices), len(track.s))
        near = np.column_stack((
            track.s[index] + laps * track.length - s,
            track.kappa[index], track.ey_min[index], track.ey_max[index],
        ))
        ahead = s + np.asarray(self.options.preview)
        far = np.column_stack((track.curvature(ahead), *track.edges(ahead)))
        observation = np.r_[self.plant.frenet[1:], near.ravel(), far.ravel()]
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _sample_start(self):
        start, vehicle = self.options.start, self._vehicle
        windows = np.array([section.start_window for section in self._sections])
        ahead = np.arange(0.0, 200.0, 2.0)
        for _ in range(100):
            section = self._sections[self.np_random.choice(len(windows), p=windows / windows.sum())]
            s = (section.start + self.np_random.uniform(0, section.start_window)) % self.track.length
            ey = self.np_random.uniform(-start.ey, start.ey)
            epsi = self.np_random.uniform(-start.epsi, start.epsi)
            curvature = np.maximum(np.abs(self.track.curvature(s + ahead)), 1e-6)
            corner_speed_sq = start.lateral_fraction * vehicle["lat_acc_max"] / curvature
            cap = np.sqrt(np.min(corner_speed_sq + 2 * start.braking * ahead))
            vx = min(self.np_random.uniform(start.speed_min, start.speed_max), cap)
            kappa = float(self.track.curvature(s))
            # Yaw rate and steering of steady cornering, so corner starts don't begin with a skid.
            self.plant.reset([s, ey, epsi, vx, 0.0, vx * kappa, vehicle["l"] * kappa, 0.0])
            if self.plant.body_margin() >= 0:
                return section
        raise RuntimeError("Found no start state on the track")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mode = options.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(f"This is a {self.mode!r} environment, not {mode!r}; build one per split")
        if "state" in options:
            self.plant.reset(options["state"])
            section = self._section_at(self.plant.wrapped_s)
        else:
            section = self._sample_start()
        self._steps, self._section_left = 0, np.inf
        self._start, self._return = self.plant.frenet.copy(), 0.0
        if self.mode != "full":
            self._section_left = (section.end - self.plant.wrapped_s) % self.track.length
        self._observation = self._observe()
        return self._observation.copy(), self._state_info()

    def _section_at(self, s):
        length = self.track.length
        for section in self._sections:
            if (s - section.start) % length < ((section.end - section.start) % length or length):
                return section
        raise ValueError(f"s = {s:.1f} m is outside the {self.mode!r} sections")

    def _state_info(self):
        return {"frenet_state": self.plant.frenet.copy(),
                "cartesian_state": self.plant.cartesian.copy()}

    def _failure(self):
        """Reason to terminate in the current plant state, or None."""
        state, stop_speed = self.plant.frenet, self.options.stop_speed
        if self.plant.body_margin() < 0:
            return "left_track"
        if abs(state[2]) > np.pi / 2:
            return "spun"
        if stop_speed is not None and state[3] < stop_speed:
            return "low_speed"
        if np.any(state[1:] < STATE_LOW) or np.any(state[1:] > STATE_HIGH):
            return "state_out_of_bounds"
        return None

    def _reward(self, progress):
        weights, (ey, epsi) = self.options.reward, self.plant.frenet[1:3]
        lateral = float(self.plant.simulator.sim_constraints.alat(self.plant.cartesian))
        excess = max(0.0, lateral - self._vehicle["lat_acc_max"], self._vehicle["lat_acc_min"] - lateral)
        reward = (weights.progress * progress / (self.speed_max * self.dt)
                  - weights.tracking * (ey**2 + weights.heading * epsi**2) * self.dt
                  - weights.lateral_excess * excess)
        return float(reward), excess

    def _time_bonus(self):
        """Evaluation only: the time saved, counted as progress at full speed."""
        if self.mode not in ("val", "test") or self.options.max_steps is None:
            return 0.0
        return self.options.reward.progress * max(0, self.options.max_steps - self._steps)

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Call reset before step")
        action = np.asarray(action, dtype=float).reshape(2)
        reward, excess, truncated = 0.0, 0.0, False
        self._steps += 1
        if not np.all(np.isfinite(action)):
            reason = "mpc_failure"
        else:
            before = self.plant.frenet[0]
            try:
                self.plant.step(action)
            except PlantStateError:
                reason = "non_finite_state"
            else:
                progress = self.plant.frenet[0] - before
                self._section_left -= progress
                reward, excess = self._reward(progress)
                self._observation = self._observe()
                reason = self._failure()

        terminated = reason is not None
        if terminated:
            reward -= self.options.reward.failure_penalty
        elif self._section_left <= 0:
            truncated, reason = True, "section_end"
            reward += self._time_bonus()
        elif self.options.max_steps is not None and self._steps >= self.options.max_steps:
            truncated, reason = True, "time_limit"

        info = self._state_info()
        info.update(termination=reason, lateral_excess=excess)
        if terminated or truncated:
            info["task"] = {"violation": reason == "left_track", "success": truncated,
                            **{f"end_{name}": float(reason == name) for name in TERMINATIONS + TRUNCATIONS}}
        self._return += reward
        if (terminated or truncated) and self._episode_log is not None:
            self._log_episode(reason)
        return self._observation.copy(), reward, terminated, truncated, info

    def _log_episode(self, reason):
        state, length = self.plant.frenet, self.track.length
        row = {
            "termination": reason, "steps": self._steps, "return": round(self._return, 4),
            "start_s_m": round(self._start[0] % length, 2), "start_vx_mps": round(self._start[3], 3),
            "end_s_m": round(state[0] % length, 2), "end_vx_mps": round(state[3], 3),
            "end_ey_m": round(state[1], 3), "distance_m": round(state[0] - self._start[0], 2),
        }
        new = not self._episode_log.exists()
        self._episode_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self._episode_log, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
