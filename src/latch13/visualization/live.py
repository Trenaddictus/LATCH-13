"""Optional live views using TUM-CONTROL's Mode 1 and Mode 2 plot functions.

Mode 1 shows the track, the full reference raceline (red, dashed), the
vehicle and its MPC prediction (blue). Mode 2 adds speed, lateral deviation and a normalised g-g
diagram. This module only converts our Frenet predictions and raceline
reference into the Cartesian arrays TUM's functions expect. Requires
external/TUM-CONTROL on sys.path.

For smooth playback the moving track-view artists are blitted over a cached
background; full redraws happen only when TUM's camera moves or Mode 2's side
panels are due. The runner draws outside its solver timing, so saved results
are identical with or without a live view.
"""
import time

import numpy as np


class _LiveView:
    """Shared history, pacing, blitting and TUM data conversion.

    ``record`` stores the plant state after every step; ``draw`` refreshes the
    figure and returns False once it has been closed. With ``rate`` set,
    drawing waits so simulated time advances at most ``rate`` x wall-clock.
    Subclasses build the figure in ``_create`` and fill it in ``_update``.
    """

    mode, panel_every = None, None

    def __init__(self, track, cartesian_state, frenet_state, vehicle, limits, *,
                 horizon, dt, speed, rate=None, window=100.0,
                 title="Frenet MPC"):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.track, self.dt, self.speed, self.rate = track, dt, float(speed), rate
        self.offsets = np.arange(horizon + 1) * dt
        self.size = (float(vehicle.veh_length), float(vehicle.veh_width))
        self.window = float(window)
        self.limits = limits
        self.boundaries = tuple(np.vstack((edge, edge[:1]))[:, axis] for axis in (0, 1)
                                for edge in (track.center, track.left, track.right))
        self.step, self.frames, self.clock = 0, 0, None
        # Rows are steps: plant state (TUM's seven), raceline reference
        # [X, Y, yaw, v_ref] at the projected progress, and acceleration state.
        self.history = {"states": np.full((2048, 7), np.nan),
                        "references": np.full((2048, 4), np.nan),
                        "acceleration": np.zeros(2048)}
        self._store(cartesian_state, frenet_state)
        # TUM's state-estimation disturbances and applied inputs are not
        # simulated here; NaN rows keep the disturbed-state line empty.
        self.disturbed = np.full((1, 7), np.nan)
        self.inputs = np.zeros((1, 2))

        from Utils.MPC_sim_utils import updateLiveVisuMode1
        self.tum_map_update = updateLiveVisuMode1
        pose = np.array([*cartesian_state[:3], np.hypot(*cartesian_state[3:5])])
        self._create(pose)
        # Show the whole raceline as static background instead of TUM's moving
        # reference horizon, so only the prediction moves with the car. TUM
        # still receives the horizon: its camera frames the view from it.
        for line in (self.disturbed_line, self.reference_line):
            line.set(visible=False, label="_nolegend_")
        raceline = np.vstack((track.points, track.points[:1]))
        self.ax.plot(*raceline.T, "r--", linewidth=1.0, alpha=0.8, zorder=2,
                     label="reference trajectory (raceline)")
        self.prediction_line.set(linewidth=2.0, marker="o", markersize=2.5, zorder=4)
        self.ax.legend(loc="lower right", fontsize="small")
        self.text.set_verticalalignment("bottom")  # status lines above the axes
        self.fig.suptitle(f"{title} | TUM-CONTROL Mode {self.mode}")

        self.moving = (self.prediction_line, self.path_line, self.vehicle, self.text)
        for artist in self.moving:
            artist.set_animated(True)  # excluded from full draws and the background
        self.background, self.view = None, None
        self.fig.canvas.mpl_connect("draw_event", self._capture)
        plt.show(block=False)

    @property
    def states(self):
        return self.history["states"]

    @property
    def panels_due(self):
        return bool(self.panel_every) and self.frames % self.panel_every == 0

    def _store(self, cartesian_state, frenet_state):
        if self.step >= len(self.states):
            for name, old in self.history.items():
                grown = np.full((2 * len(old), *old.shape[1:]), np.nan)
                grown[:len(old)] = old
                self.history[name] = grown
        xy, yaw, _ = self.track.pose(frenet_state[0])
        self.history["states"][self.step] = cartesian_state
        self.history["references"][self.step] = [*xy, yaw, self.speed]
        self.history["acceleration"][self.step] = frenet_state[7]

    def record(self, cartesian_state, frenet_state):
        self.step += 1
        self._store(cartesian_state, frenet_state)

    def draw(self, time_s, frenet_state, prediction, solve_ms):
        """Redraw for the latest recorded state and the prediction made from it."""
        if not self.plt.fignum_exists(self.fig.number):
            return False
        self._pace(time_s)
        self.frames += 1
        xy, yaw, index = self.track.pose(frenet_state[0] + self.offsets * self.speed)
        reference = {"pos_x": xy[:, 0], "pos_y": xy[:, 1], "ref_yaw": yaw}
        self._update(time_s, reference, self.track.cartesian(prediction[:, 0], prediction[:, 1]),
                     int(index[0]))
        self.text.set_text(
            f"{self.text.get_text()}\n"
            f"lap {int(frenet_state[0] // self.track.length)}, ey: {frenet_state[1]:+.3f} m, "
            f"solve: {solve_ms:.1f} ms"
        )
        self._render()
        return self.plt.fignum_exists(self.fig.number)

    def _update_map(self, time_s, reference, prediction, index):
        """TUM's Mode 1 update: track-view artists, camera and status text."""
        self.tum_map_update(
            self.step, time_s, self.states, self.inputs, self.disturbed, reference, prediction,
            self.reference_line, self.prediction_line, self.disturbed_line, self.ax,
            self.window, self.window, self.path_line, self.vehicle, self.text, *self.size,
            current_ref_idx=index,
        )

    def _capture(self, _event):
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.view = (self.ax.get_xlim(), self.ax.get_ylim())

    def _render(self):
        canvas = self.fig.canvas
        if (self.background is None or self.panels_due
                or self.view != (self.ax.get_xlim(), self.ax.get_ylim())):
            canvas.draw()  # redraw static artists and panels; recaptures background
        canvas.restore_region(self.background)
        for artist in self.moving:
            self.ax.draw_artist(artist)
        canvas.blit(self.fig.bbox)
        canvas.flush_events()

    def _pace(self, time_s):
        if not self.rate:
            return
        now = time.perf_counter()
        if self.clock is None:
            self.clock = now - time_s / self.rate
        delay = self.clock + time_s / self.rate - now
        if delay > 0:
            self.fig.canvas.start_event_loop(delay)  # wait, keeping the window responsive

    def close(self):
        self.plt.close(self.fig)


class Mode1View(_LiveView):
    """Track view only."""

    mode = 1

    def _create(self, pose):
        from Utils.MPC_sim_utils import initLiveVisuMode1

        self.fig, ax = self.plt.subplots(figsize=(10, 10))
        (self.prediction_line, self.reference_line, self.disturbed_line, self.ax,
         _, _, _, self.path_line, self.vehicle, self.text) = initLiveVisuMode1(
            ax, self.states, pose, self.window, self.window, *self.boundaries, *self.size,
        )

    def _update(self, time_s, reference, prediction, index):
        self._update_map(time_s, reference, prediction, index)


class Mode2View(_LiveView):
    """Track view plus speed, lateral deviation and g-g diagram.

    The side panels show the whole run's history and rescale on every update,
    so they refresh with a full redraw every ``panel_every`` frames, from a
    thinned history: TUM's Mode 2 update recolours every g-g point in Python.
    The track view always uses TUM's Mode 1 update at full resolution.
    """

    mode, panel_every, panel_points = 2, 25, 1000

    def _create(self, pose):
        from Utils.MPC_sim_utils import initLiveVisuMode2, updateLiveVisuMode2

        self.tum_update = updateLiveVisuMode2
        # TUM's Mode 2 layout; initLiveVisuMode2 draws the g-g scatter into the
        # current axes, so the g-g axes must be created last.
        self.fig = self.plt.figure(figsize=(15, 8))
        for layout in ((2, 2, (1, 3)), (2, 4, 3), (2, 4, 4), (2, 4, (7, 8))):
            self.fig.add_subplot(*layout)
        ax, ax_vel, ax_devlat, ax_gg = self.fig.axes
        (self.prediction_line, self.reference_line, self.disturbed_line, self.ax,
         self.ax_devlat, self.ax_vel, self.ax_gg, self.heatmap, self.cbar, _, _, _,
         self.path_line, self.vehicle, self.text, self.vel_ref_line, self.vel_sim_line,
         self.devlat_line) = initLiveVisuMode2(
            np.linspace(0.0, 5 * self.dt, 5), ax, ax_vel, ax_devlat, ax_gg, self.states,
            self.history["references"], pose, self.window, self.window,
            *self.boundaries, *self.size,
        )
        # Normalise by the controller's own bounds (speed independent).
        self.ax_limit = lambda v: np.full(np.shape(v), self.limits["acc_max"])
        self.ay_limit = lambda v: np.full(np.shape(v), self.limits["lat_acc_max"])

    def _update(self, time_s, reference, prediction, index):
        if self.panels_due:
            self._update_panels(time_s, reference, prediction, index)
        # Track view at full resolution, after the panels' thinned history.
        self._update_map(time_s, reference, prediction, index)
        state = self.states[self.step]
        self.text.set_text(  # TUM's Mode 2 status wording
            f"steering angle: {np.rad2deg(state[6]):.2f}°,      "
            f"velocity: {state[3] * 3.6:.2f}km/h"
        )

    def _update_panels(self, time_s, reference, prediction, index):
        """TUM's Mode 2 update on at most ``panel_points`` evenly spaced steps.

        TUM treats index ``n`` as the current step and ``[:n]`` as history,
        so the thinned rows always end with the current step.
        """
        rows = np.unique(np.r_[np.linspace(0, self.step - 1, min(self.step, self.panel_points))
                               .astype(int), self.step])
        n = len(rows) - 1
        states = self.states[rows]
        self.tum_update(
            n, rows[:-1] * self.dt, time_s, self.history["references"][rows], states,
            self.disturbed, self.inputs, self.history["acceleration"][rows][:n],
            states[:n, 3] * states[:n, 5],  # TUM's a_lat = vx * yaw rate
            reference, prediction, self.limits["acc_min"], self.ax_limit, self.ay_limit,
            self.reference_line, self.prediction_line, self.disturbed_line, self.ax,
            self.ax_devlat, self.ax_vel, self.ax_gg, self.window, self.window,
            self.path_line, self.vehicle, self.text, *self.size, index,
            self.vel_ref_line, self.vel_sim_line, self.devlat_line, self.heatmap, self.cbar,
        )


LIVE_VIEWS = {1: Mode1View, 2: Mode2View}
