"""
Standalone tests for Utils/frenet_projection.py.

1. Round trip: random Frenet states -> Cartesian -> Frenet must reproduce (s, e_y, e_psi).
2. Logged closed-loop run (Cartesian TUM-CONTROL controller): project the simulated states and check
   - s increases continuously,
   - e_y vs. the track bounds,
   - kinematic consistency: the time derivatives of the projected (s, e_y, e_psi) must match the
     Frenet kinematics used in the planned NMPC model
         s_dot     = (v_x cos e_psi - v_y sin e_psi) / (1 - kappa e_y)
         e_y_dot   =  v_x sin e_psi + v_y cos e_psi
         e_psi_dot =  psi_dot - kappa s_dot

Usage (from repo root):
    python Utils/test_frenet_projection.py [path/to/full_logs.npz] [--plot out.png]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.frenet_projection import FrenetReference, wrap_angle  # noqa: E402

REF_FILE = "Trajectories/reftraj_silverstone_edgar.json"
DEFAULT_LOG = "Logs/SNMPC_trial2026-11-09_12-47-28/full_logs.npz"   # Silverstone, Tp=3.04, WMPC off
TS = 0.02                                                             # simulator step of the logs [s]
VEH_HALF_WIDTH = 1.941 / 2                                            # EDGAR, veh_params_*.yaml


def test_round_trip(ref, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, ref.L, n)
    r = ref.ref_at(s)
    e_y = rng.uniform(r["ey_min"], r["ey_max"])                      # anywhere on the track
    e_psi = rng.uniform(-0.5, 0.5, n)
    err = np.zeros((n, 3))
    for k in range(n):
        xc = ref.frenet_to_cartesian([s[k], e_y[k], e_psi[k]])
        xf = ref.cartesian_to_frenet(xc)                              # global search, no guess
        ds = (xf[0] - s[k] + ref.L / 2) % ref.L - ref.L / 2
        err[k] = [ds, xf[1] - e_y[k], wrap_angle(xf[2] - e_psi[k])]
    print("[1] round trip over the whole track (n=%d, |e_y| up to %.1f m)" % (n, np.abs(e_y).max()))
    print("    max |error|: s %.3f m, e_y %.3f m, e_psi %.4f rad" % tuple(np.abs(err).max(axis=0)))
    return np.abs(err).max(axis=0)


def test_logged_run(ref, log_file, plot_file=None):
    d = np.load(log_file)
    X = d["CiLX"]                                                     # [x, y, yaw, vlon, vlat, r, delta]
    t = np.arange(len(X)) * TS
    XF, s_un = ref.project_trajectory(X)
    s, e_y, e_psi = XF[:, 0], XF[:, 1], XF[:, 2]
    vx, vy, r = XF[:, 3], XF[:, 4], XF[:, 5]
    rq = ref.ref_at(s)
    kap = rq["kappa"]

    print("\n[2] logged run: %s" % log_file)
    print("    s: %.0f -> %.0f m, min step %.3f m (negative = s went backwards)"
          % (s_un[0], s_un[-1], np.diff(s_un).min()))
    margin = np.minimum(rq["ey_max"] - e_y, e_y - rq["ey_min"]) - VEH_HALF_WIDTH
    print("    e_y: RMS %.2f m, max |e_y| %.2f m; min car-edge-to-track-edge margin %.2f m at t=%.1f s"
          % (np.sqrt(np.mean(e_y**2)), np.abs(e_y).max(), margin.min(), t[np.argmin(margin)]))
    print("    e_psi: RMS %.4f rad, max |e_psi| %.4f rad" % (np.sqrt(np.mean(e_psi**2)), np.abs(e_psi).max()))
    if "dev_lat" in d:
        dl = d["dev_lat"][: len(e_y)]
        print("    TUM dev_lat (time-indexed ref point): RMS %.2f m, max %.2f m  [for comparison only]"
              % (np.sqrt(np.mean(dl**2)), np.abs(dl).max()))

    # kinematic consistency: central differences of projected states vs. Frenet kinematics
    sdot_model = (vx * np.cos(e_psi) - vy * np.sin(e_psi)) / (1 - kap * e_y)
    eydot_model = vx * np.sin(e_psi) + vy * np.cos(e_psi)
    epsidot_model = r - kap * sdot_model
    c = slice(1, -1)
    sdot_num = (s_un[2:] - s_un[:-2]) / (2 * TS)
    eydot_num = (e_y[2:] - e_y[:-2]) / (2 * TS)
    epsidot_num = wrap_angle(e_psi[2:] - e_psi[:-2]) / (2 * TS)
    res = {
        "s_dot [m/s]": sdot_num - sdot_model[c],
        "e_y_dot [m/s]": eydot_num - eydot_model[c],
        "e_psi_dot [rad/s]": epsidot_num - epsidot_model[c],
    }
    print("    kinematic consistency (numerical derivative - Frenet model), over %d steps:" % (len(X) - 2))
    for k, v in res.items():
        print("      %-18s RMS %.4f  max %.4f   (typical magnitude: %.3f)"
              % (k, np.sqrt(np.mean(v**2)), np.abs(v).max(),
                 np.sqrt(np.mean({"s_dot [m/s]": sdot_model, "e_y_dot [m/s]": eydot_model,
                                  "e_psi_dot [rad/s]": epsidot_model}[k][c] ** 2))))

    if plot_file:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        ax[0].plot(t, s_un); ax[0].set_ylabel("s [m]")
        ax[1].plot(t, e_y, label="e_y (projection)")
        if "dev_lat" in d:
            dl = d["dev_lat"]
            ax[1].plot(t[: len(dl)], dl[: len(t)], lw=0.7, alpha=0.7, label="TUM dev_lat")
        ax[1].plot(t, rq["ey_max"] - VEH_HALF_WIDTH, "k--", lw=0.7, label="track edge - half width")
        ax[1].plot(t, rq["ey_min"] + VEH_HALF_WIDTH, "k--", lw=0.7)
        ax[1].set_ylim(-8, 8); ax[1].set_ylabel("e_y [m]"); ax[1].legend(loc="upper right", fontsize=8)
        ax[2].plot(t, e_psi); ax[2].set_ylabel("e_psi [rad]")
        ax[3].plot(t[c], res["s_dot [m/s]"], lw=0.7, label="s_dot residual [m/s]")
        ax[3].plot(t[c], res["e_y_dot [m/s]"], lw=0.7, label="e_y_dot residual [m/s]")
        ax[3].plot(t[c], res["e_psi_dot [rad/s]"], lw=0.7, label="e_psi_dot residual [rad/s]")
        ax[3].set_ylabel("residual"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=8)
        for a in ax:
            a.grid(True)
        fig.suptitle("Frenet projection of logged run")
        fig.tight_layout()
        fig.savefig(plot_file, dpi=100)
        print("    plot saved:", plot_file)
    return XF, s_un


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=DEFAULT_LOG)
    ap.add_argument("--plot", default=None, help="save a check plot to this path")
    a = ap.parse_args()
    ref = FrenetReference(REF_FILE)
    print("reference: %s, %d points, lap length %.1f m" % (REF_FILE, ref.n, ref.L))
    test_round_trip(ref)
    test_logged_run(ref, a.log, a.plot)
