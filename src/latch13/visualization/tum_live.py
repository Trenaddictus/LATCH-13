
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Utils import Logging_Plotting as tum_plot


def reference_pose(track, s):
    """Return X, Y, heading at raceline progress s."""
    s = s % track["L"]

    idx = np.searchsorted(
        track["s"], s, side="right"
    ) - 1
    idx = int(np.clip(idx, 0, len(track["s"]) - 1))

    segment = track["segments"][idx]
    length = track["segment_lengths"][idx]

    alpha = np.clip(
        (s - track["s"][idx]) / length,
        0.0, 1.0
    )

    xy = track["points"][idx] + alpha * segment
    yaw = np.arctan2(segment[1], segment[0])

    return xy[0], xy[1], yaw


def load_boundaries(path, scale):
    """Read F1TENTH centerline and left/right track widths."""
    data = np.loadtxt(
        path, delimiter=";", comments="#", ndmin=2
    )

    if data.shape[1] < 4:
        raise ValueError(
            "Expected centerline columns: x, y, right, left"
        )

    center = data[:, :2] * scale
    right_width = data[:, 2] * scale
    left_width = data[:, 3] * scale

    tangent = (
        np.roll(center, -1, axis=0)
        - np.roll(center, 1, axis=0)
    )
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=1, keepdims=True),
        1e-12
    )

    normal = np.column_stack(
        (-tangent[:, 1], tangent[:, 0])
    )

    left = center + left_width[:, None] * normal
    right = center - right_width[:, None] * normal

    return center, left, right


class TUMLivePlot:
    """
    Adapter around TUM-CONTROL's Mode 1 visualization.

    Physical states: Cartesian, seven elements.
    MPC predictions: Frenet, seven elements.
    """

    def __init__(
        self, track, centerline_path, scale,
        initial_state, vehicle_length, vehicle_width,
        horizon, dt, reference_speed,
    ):
        self.track = track
        self.N = horizon
        self.dt = dt
        self.reference_speed = reference_speed
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.started = time.time()

        center, left, right = load_boundaries(
            centerline_path, scale
        )

        # Allocate expandable simulation history.
        self.capacity = 2000
        self.CiLX = np.full((self.capacity, 7), np.nan)
        self.DisturbedX = np.full((self.capacity, 7), np.nan)
        self.simU = np.full((self.capacity, 2), np.nan)

        self.CiLX[0] = initial_state
        self.DisturbedX[0] = initial_state

        current_pose = np.array([
            initial_state[0],
            initial_state[1],
            initial_state[2],
            np.hypot(initial_state[3], initial_state[4]),
        ])

        # TUM's original interactive plotting functions.
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.xwidth = 100.0
        self.ywidth = 100.0

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
        ) = tum_plot.initLiveVisuMode1(
            self.ax,
            self.CiLX,
            current_pose,
            self.xwidth,
            self.ywidth,
            center[:, 0],
            left[:, 0],
            right[:, 0],
            center[:, 1],
            left[:, 1],
            right[:, 1],
            vehicle_length,
            vehicle_width,
        )

        self.fig.suptitle("LATCH-13 — TUM-CONTROL Visualization")
        plt.show(block=False)

    def _ensure_capacity(self, index):
        if index < self.capacity:
            return

        new_capacity = max(index + 1, self.capacity * 2)

        for name in ("CiLX", "DisturbedX", "simU"):
            old = getattr(self, name)
            new = np.full(
                (new_capacity, old.shape[1]), np.nan
            )
            new[:len(old)] = old
            setattr(self, name, new)

        self.capacity = new_capacity

    def _predictions(self, solver):
        """Convert optimized Frenet states into TUM plot states."""
        prediction = np.zeros((self.N + 1, 8))

        for k in range(self.N + 1):
            xf = np.asarray(
                solver.get(k, "x")
            ).reshape(-1)

            s, ey, epsi, vx, vy, r, delta = xf
            Xref, Yref, psi_ref = reference_pose(
                self.track, s
            )

            prediction[k, :7] = [
                Xref - ey * np.sin(psi_ref),
                Yref + ey * np.cos(psi_ref),
                psi_ref + epsi,
                vx, vy, r, delta,
            ]

            if k < self.N:
                prediction[k, 7] = float(
                    np.asarray(
                        solver.get(k, "u")
                    ).reshape(-1)[0]
                )
            else:
                prediction[k, 7] = prediction[k - 1, 7]

        return prediction

    def _reference(self, progress):
        """Construct the Cartesian reference horizon."""
        rows = []

        for k in range(self.N + 1):
            s = progress + k * self.dt * self.reference_speed
            X, Y, yaw = reference_pose(self.track, s)
            rows.append([X, Y, yaw, self.reference_speed])

        return np.asarray(rows)

    def update(self, step, sim_state, control, solver, progress):
        self._ensure_capacity(step + 2)

        self.CiLX[step + 1] = sim_state
        self.DisturbedX[step + 1] = sim_state
        self.simU[step] = control

        pred_X = self._predictions(solver)
        current_ref_traj = self._reference(progress)

        current_ref_idx = int(np.searchsorted(
            self.track["s"],
            progress % self.track["L"],
            side="right",
        ) - 1)

        # Original TUM visualization update function
        tum_plot.updateLiveVisuMode1(
            step + 1,
            time.time() - self.started,
            self.CiLX,
            self.simU,
            self.DisturbedX,
            current_ref_traj,
            pred_X,
            self.Ref_traj_line,
            self.MPC_pred_line,
            self.Disturbed_traj_line,
            self.ax,
            self.xwidth,
            self.ywidth,
            self.line,
            self.dot,
            self.text,
            self.vehicle_length,
            self.vehicle_width,
            current_ref_idx=current_ref_idx,
        )

        plt.pause(0.001)

        return plt.fignum_exists(self.fig.number)
