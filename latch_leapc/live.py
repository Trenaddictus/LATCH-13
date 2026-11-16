"""Live view of closed-loop rollouts: car, MPC prediction and track, updated while simulating.

Opt-in via `--live` (baseline / compare / all). Needs a display; under WSL2 this is WSLg.
"""
import os

import numpy as np


def _xy(ref, s, ey):
    r = ref.ref_at(s)
    return r["x"] - ey*np.sin(r["psi"]), r["y"] + ey*np.cos(r["psi"])


class LiveView:
    def __init__(self, settings, ref, every=1, window=160.):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("--live needs a display (DISPLAY / WAYLAND_DISPLAY not set).")
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt, self.ref, self.settings = plt, ref, settings
        self.every, self.window = max(int(every), 1), window
        self.closed = False
        plt.ion()
        fig = plt.figure("latch_leapc live", figsize=(15, 8.5))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.])
        self.fig = fig
        self.ax_follow = fig.add_subplot(gs[:, 0])
        self.ax_track = fig.add_subplot(gs[0, 1])
        self.ax_v = fig.add_subplot(gs[1, 1])
        self.ax_ey = fig.add_subplot(gs[2, 1])

        # Track: reference line and the two edges (e_y positive = left).
        r = ref.ref_at(ref.s)
        s_closed = np.r_[ref.s, ref.s[:1]]
        cx, cy = _xy(ref, s_closed, 0.)
        lx, ly = _xy(ref, s_closed, np.r_[r["ey_max"], r["ey_max"][:1]])
        rx, ry = _xy(ref, s_closed, np.r_[r["ey_min"], r["ey_min"][:1]])
        for ax in (self.ax_follow, self.ax_track):
            ax.plot(cx, cy, "k--", lw=.8, alpha=.5, label="reference")
            ax.plot(lx, ly, color="0.35", lw=1.2)
            ax.plot(rx, ry, color="0.35", lw=1.2)
            ax.set_aspect("equal")
        self.ax_track.set_title("full track", fontsize=9)
        self.ax_track.set_xticks([]); self.ax_track.set_yticks([])
        self.ax_follow.set(xlabel="x [m]", ylabel="y [m]")
        self.ax_v.set(ylabel="speed [m/s]")
        self.ax_ey.set(xlabel="time [s]", ylabel="lateral error e_y [m]")
        for ax in (self.ax_follow, self.ax_v, self.ax_ey):
            ax.grid(True, alpha=.3)
        self.text = self.ax_follow.text(
            .02, .98, "", transform=self.ax_follow.transAxes, va="top", family="monospace",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=.85))
        self.run = None
        self.colors = iter(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
        fig.tight_layout()
        fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show(block=False)
        self._draw()

    def _on_close(self, _):
        self.closed = True
        print("Live window closed; simulation continues without live view.", flush=True)

    def _draw(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def begin(self, name, horizon):
        """Start a new rollout. Earlier rollouts stay visible as thin trails."""
        if self.closed:
            return
        if self.run is not None:
            for key in ("pred", "car", "pred_track", "car_track"):
                self.run[key].remove()
            self.run["trail"].set(lw=1., alpha=.6)
        c = next(self.colors, "k")
        dt = self.settings["simulation_dt"]
        self.run = dict(
            name=name, N=horizon, dt=dt, t=[], v=[], vref=[], ey=[], xy=[],
            trail=self.ax_follow.plot([], [], color=c, lw=2., label=f"{name} (N={horizon})")[0],
            pred=self.ax_follow.plot([], [], "o-", color=c, ms=3, lw=1., alpha=.8,
                                     label="MPC prediction")[0],
            car=self.ax_follow.plot([], [], marker=(3, 0, 0), ms=14, color=c, mec="k")[0],
            pred_track=self.ax_track.plot([], [], "-", color=c, lw=2.)[0],
            car_track=self.ax_track.plot([], [], "o", color=c, ms=6, mec="k")[0],
            v_line=self.ax_v.plot([], [], color=c, label=name)[0],
            vref_line=self.ax_v.plot([], [], "k--", lw=.8, alpha=.6)[0],
            ey_line=self.ax_ey.plot([], [], color=c, label=name)[0],
        )
        self.ax_follow.legend(loc="lower right", fontsize=8)
        self.ax_v.legend(loc="lower left", fontsize=8)
        self.ax_follow.set_title(f"{name}: N={horizon} ({horizon*self.settings['prediction_dt']:.2f} s horizon)")

    def update(self, i, xf, cart, xpred, u, elapsed):
        if self.closed or self.run is None:
            return
        R = self.run
        vref = float(self.ref.ref_at(xf[0])["v_ref"])
        R["t"].append((i+1)*R["dt"]); R["v"].append(xf[3]); R["vref"].append(vref)
        R["ey"].append(xf[1]); R["xy"].append(cart[:2].copy())
        if i % self.every:
            return
        px, py = _xy(self.ref, xpred[:, 0], xpred[:, 1])
        xy = np.asarray(R["xy"])
        R["trail"].set_data(xy[:, 0], xy[:, 1])
        R["pred"].set_data(px, py)
        # Triangle marker rotated to the vehicle yaw (marker angle 0 points up).
        R["car"].set_marker((3, 0, np.degrees(cart[2]) - 90.))
        R["car"].set_data([cart[0]], [cart[1]])
        R["pred_track"].set_data(px, py)
        R["car_track"].set_data([cart[0]], [cart[1]])
        h = self.window/2
        cx, cy = cart[0] + .35*h*np.cos(cart[2]), cart[1] + .35*h*np.sin(cart[2])
        self.ax_follow.set_xlim(cx - h, cx + h); self.ax_follow.set_ylim(cy - h, cy + h)
        R["v_line"].set_data(R["t"], R["v"]); R["vref_line"].set_data(R["t"], R["vref"])
        R["ey_line"].set_data(R["t"], R["ey"])
        for ax in (self.ax_v, self.ax_ey):
            ax.relim(); ax.autoscale_view()
        self.text.set_text(
            f"{R['name']}  N={R['N']}   step {i+1}   t={R['t'][-1]:6.2f} s\n"
            f"v   = {xf[3]:6.2f} m/s  (ref {vref:5.2f})\n"
            f"e_y = {xf[1]:+7.3f} m   e_psi = {xf[2]:+.4f} rad\n"
            f"jerk = {u[0]:+6.2f} m/s^3   steer rate = {u[1]:+.3f} rad/s\n"
            f"solve = {1e3*elapsed:6.1f} ms  (period {1e3*R['dt']:.0f} ms)")
        self._draw()

    def failed(self, message):
        if self.closed:
            return
        self.text.set_text(self.text.get_text() + f"\nFAILED: {message}")
        self.text.get_bbox_patch().set_facecolor("mistyrose")
        self._draw()

    def finish(self):
        """Keep the window open after the run until the user closes it."""
        if self.closed:
            return
        print("Live view: close the window to exit.", flush=True)
        self.plt.ioff()
        self.plt.show()
