"""The TUM plant on a Frenet track, shared by the runner and the RL environment.

The plant has seven Cartesian states and [a, steering_rate] as input. The MPC
adds progress and the acceleration state, so ``ClosedLoopPlant`` carries both.
As with ``ocp.py``, external/TUM-CONTROL must be on sys.path.
"""

from pathlib import Path

import numpy as np
import yaml


class PlantStateError(ValueError):
    """The plant returned a non-finite state."""


def create_simulator(config_dir, mpc_dt):
    config_dir = Path(config_dir)
    from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator

    with open(config_dir / "sim_main_params.yaml", encoding="utf-8") as f:
        sim_config = yaml.safe_load(f)
    sim_dt = float(sim_config["Ts"])
    ratio = mpc_dt / sim_dt
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, atol=1e-10):
        raise ValueError(f"MPC dt={mpc_dt} is not a multiple of sim dt={sim_dt}")

    # TUM YAML paths include 'EDGAR/' already; provide parent Config/.
    vehicle = PassengerVehicleSimulator(
        config_path=str(config_dir.parent) + "/",
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
            raise PlantStateError("Invalid Cartesian plant state")
    return state, float(acceleration)


class ClosedLoopPlant:
    """Plant, projection and progress unwrapping over one MPC interval per step.

    ``cartesian`` is [X, Y, yaw, vx, vy, r, delta], ``frenet`` the MPC state
    [s, ey, epsi, vx, vy, r, delta, a] with unwrapped s.
    """

    def __init__(self, config_dir, track, dt):
        self.track = track
        self.simulator, self.sim_dt, self.substeps = create_simulator(config_dir, dt)
        self.cartesian = self.frenet = None
        self.wrapped_s = 0.0

    def reset(self, state):
        """Place the car at an MPC state [s, ey, epsi, vx, vy, r, delta, a]."""
        s, ey, epsi, *motion, acceleration = np.array(state, dtype=float).reshape(8)
        xy, yaw, _ = self.track.pose(s)
        if ey:
            xy = self.track.cartesian(s, ey)
        self.cartesian = np.array([*xy, float(yaw) + epsi, *motion])
        # Re-project: next to a vertex, the nearest segment can differ from that of s.
        self.wrapped_s, ey, epsi = self.track.project(*self.cartesian[:3])
        self.frenet = np.array([self.wrapped_s, ey, epsi, *motion, acceleration])

    def step(self, control):
        """Apply [jerk, steering_rate] for one MPC interval."""
        self.cartesian, acceleration = advance_plant(
            self.simulator, self.cartesian, self.frenet[7], control, self.sim_dt, self.substeps,
        )
        wrapped_s, ey, epsi = self.track.project(*self.cartesian[:3])
        half_lap = self.track.length / 2
        change = ((wrapped_s - self.wrapped_s + half_lap) % self.track.length) - half_lap
        self.wrapped_s = wrapped_s
        self.frenet = np.array([self.frenet[0] + change, ey, epsi, *self.cartesian[3:], acceleration])

    def body_margin(self):
        """Distance from the body to the nearest edge; negative off the track."""
        params = self.simulator.params
        return self.track.body_margin(
            *self.cartesian[:3], float(params.veh_length), float(params.veh_width),
        )
