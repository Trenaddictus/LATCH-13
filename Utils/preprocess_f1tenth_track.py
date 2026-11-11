"""
Preprocess an F1TENTH racetrack (1:10-scale) into a full-scale reference for the EDGAR vehicle.

Inputs  (raw, unchanged files from the F1TENTH racetracks repository):
    Trajectories/F1TENTH/<Track>/<Track>_raceline.csv    # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    Trajectories/F1TENTH/<Track>/<Track>_centerline.csv  # x_m, y_m, w_tr_right_m, w_tr_left_m

Processing:
    1. Uniform scaling by k = target_length / raceline_length (x, y, s, track widths scaled by k; kappa by 1/k).
       The raceline geometry is kept; the F1TENTH speed profile is discarded.
    2. Track bounds relative to the raceline: e_y_max(s) (left, > 0) and e_y_min(s) (right, < 0).
    3. New speed profile v_ref(s) from the vehicle's g-g-v limits (Config/<veh>/ggv.csv) with a
       forward-backward pass and circular combined acceleration limits (same shape as combined_acc_limits: 2).

Outputs:
    Trajectories/reftraj_<track>_<veh>.json   # TUM keys (pos_x, pos_y, ref_yaw, ref_v, ref_acc, orientation_*)
                                              # + Frenet keys (s, kappa, ey_max, ey_min)
    Trajectories/track_<track>.json           # X, Y (centerline), X_i/Y_i (right bound), X_o/Y_o (left bound)
    Trajectories/reftraj_<track>_<veh>_meta.json  # processing parameters and summary, for reproducibility

Usage (from repo root):
    python Utils/preprocess_f1tenth_track.py [--plot]
"""

import argparse
import json
import os

import numpy as np

# ---------------- parameters ----------------
TRACK = "Silverstone"
VEHICLE = "EDGAR"
TARGET_LENGTH = 5891.0      # [m] real Silverstone GP circuit length; sets the scale factor k
GG_SCALE = 1.0              # safety factor on the g-g-v limits used for v_ref (1.0 = use limits as is)
V_MAX = None                # [m/s] speed cap; None -> highest velocity in ggv.csv
# --------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw(track):
    d = os.path.join(REPO, "Trajectories", "F1TENTH", track)
    rl = np.loadtxt(os.path.join(d, f"{track}_raceline.csv"), delimiter=";", comments="#")
    cl = np.loadtxt(os.path.join(d, f"{track}_centerline.csv"), delimiter=",", comments="#")
    return rl, cl


def ey_bounds(px, py, psi, cx, cy, w_right, w_left):
    """Signed distances from each raceline point to the left/right track edge, measured along the
    raceline normal. Approximation: uses the offset to the nearest centerline segment and assumes
    raceline and centerline are locally parallel (good where the raceline heading ~ centerline heading)."""
    n_c = len(cx)
    seg_x = np.roll(cx, -1) - cx            # closed centerline segments
    seg_y = np.roll(cy, -1) - cy
    seg_len2 = seg_x**2 + seg_y**2
    ey_max = np.empty_like(px)
    ey_min = np.empty_like(px)
    offset = np.empty_like(px)
    for k in range(len(px)):
        # projection onto all segments, pick nearest
        t = ((px[k] - cx) * seg_x + (py[k] - cy) * seg_y) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        qx, qy = cx + t * seg_x, cy + t * seg_y
        i = np.argmin((px[k] - qx)**2 + (py[k] - qy)**2)
        j = (i + 1) % n_c
        tx, ty = seg_x[i], seg_y[i]
        L = np.sqrt(seg_len2[i])
        nx, ny = -ty / L, tx / L                                  # left normal of centerline
        d = (px[k] - qx[i]) * nx + (py[k] - qy[i]) * ny            # >0: raceline left of centerline
        wl = w_left[i] + t[i] * (w_left[j] - w_left[i])
        wr = w_right[i] + t[i] * (w_right[j] - w_right[i])
        offset[k] = d
        ey_max[k] = wl - d
        ey_min[k] = -(wr + d)
    return ey_max, ey_min, offset


def load_ggv(veh):
    g = np.loadtxt(os.path.join(REPO, "Config", veh, "ggv.csv"), delimiter=",", skiprows=1)
    return g[:, 0], g[:, 1], g[:, 2], g[:, 3]   # v, ax_max, ax_min (<0), ay_max


def speed_profile(s_closed, kappa, ggv, gg_scale, v_max):
    """Forward-backward pass on a closed track. s_closed has len(kappa)+1 entries (last = lap length)."""
    gv, gax, gaxmin, gay = ggv
    ax_max = lambda v: gg_scale * np.interp(v, gv, gax)
    ax_min = lambda v: gg_scale * np.interp(v, gv, gaxmin)    # negative
    ay_max = lambda v: gg_scale * np.interp(v, gv, gay)

    n = len(kappa)
    ds = np.diff(s_closed)                                    # ds[i]: from point i to point i+1 (wraps)
    ak = np.maximum(np.abs(kappa), 1e-9)

    # lateral limit v^2 |kappa| <= ay_max(v): fixed-point iteration (ay_max depends on v)
    v_lat = np.full(n, v_max)
    for _ in range(20):
        v_lat = np.minimum(np.sqrt(ay_max(v_lat) / ak), v_max)

    def avail(v, k, a_lim):
        ratio = np.clip(v**2 * abs(k) / ay_max(v), 0.0, 1.0)
        return a_lim * np.sqrt(1.0 - ratio**2)

    v = v_lat.copy()
    i0 = int(np.argmin(v_lat))                                # start at slowest corner (closed loop)
    for _ in range(2):                                        # two laps for convergence across the start line
        # forward (acceleration)
        for m in range(n):
            i = (i0 + m) % n
            j = (i + 1) % n
            v_next = np.sqrt(v[i]**2 + 2 * avail(v[i], kappa[i], ax_max(v[i])) * ds[i])
            v[j] = min(v[j], v_next)
        # backward (braking)
        for m in range(n):
            j = (i0 - m) % n
            i = (j - 1) % n
            v_prev = np.sqrt(v[j]**2 - 2 * avail(v[j], kappa[j], ax_min(v[j])) * ds[i])
            v[i] = min(v[i], v_prev)
    a = (np.roll(v, -1)**2 - v**2) / (2 * ds)                 # longitudinal acceleration along the profile
    lap_time = float(np.sum(2 * ds / (v + np.roll(v, -1))))
    return v, a, lap_time


def main(plot=False):
    rl, cl = load_raw(TRACK)

    # raceline: last point duplicates the first (closed) -> drop it, keep lap length
    lap_len_raw = rl[-1, 0]
    if np.hypot(rl[-1, 1] - rl[0, 1], rl[-1, 2] - rl[0, 2]) < 1e-6:
        rl = rl[:-1]
    k = TARGET_LENGTH / lap_len_raw

    s = rl[:, 0] * k
    px, py = rl[:, 1] * k, rl[:, 2] * k
    psi = rl[:, 3]
    kappa = rl[:, 4] / k
    lap_len = lap_len_raw * k
    s_closed = np.append(s, lap_len)

    # centerline and bounds (centerline is open in the file: last point ~0.4 m (raw) from the first)
    cx, cy = cl[:, 0] * k, cl[:, 1] * k
    w_right, w_left = cl[:, 2] * k, cl[:, 3] * k
    ey_max, ey_min, offset = ey_bounds(px, py, psi, cx, cy, w_right, w_left)

    # track edges for plotting (centerline normals)
    tx, ty = np.gradient(cx), np.gradient(cy)
    tn = np.hypot(tx, ty)
    nx, ny = -ty / tn, tx / tn
    X_o, Y_o = cx + w_left * nx, cy + w_left * ny             # left edge
    X_i, Y_i = cx - w_right * nx, cy - w_right * ny           # right edge

    # speed profile
    ggv = load_ggv(VEHICLE)
    v_max = V_MAX if V_MAX is not None else float(ggv[0][-1])
    v_ref, a_ref, lap_time = speed_profile(s_closed, kappa, ggv, GG_SCALE, v_max)

    yaw = np.mod(psi, 2 * np.pi)                              # TUM convention: ref_yaw in [0, 2pi)
    ref = {
        "pos_x": px, "pos_y": py,
        "orientation_x": np.zeros_like(px), "orientation_y": np.zeros_like(px),
        "orientation_z": np.sin(yaw / 2), "orientation_w": np.cos(yaw / 2),
        "ref_v": v_ref, "ref_acc": a_ref, "ref_yaw": yaw,
        # Frenet / project-specific keys
        "s": s, "kappa": kappa, "ey_max": ey_max, "ey_min": ey_min,
    }
    ref = {key: [float(x) for x in val] for key, val in ref.items()}
    track = {key: [float(x) for x in val] for key, val in
             {"X": cx, "Y": cy, "X_i": X_i, "Y_i": Y_i, "X_o": X_o, "Y_o": Y_o}.items()}

    name = f"reftraj_{TRACK.lower()}_{VEHICLE.lower()}"
    out_dir = os.path.join(REPO, "Trajectories")
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(ref, f)
    with open(os.path.join(out_dir, f"track_{TRACK.lower()}.json"), "w") as f:
        json.dump(track, f)

    a_lat = v_ref**2 * kappa
    meta = {
        "source": f"Trajectories/F1TENTH/{TRACK}/ (raceline + centerline)",
        "script": "Utils/preprocess_f1tenth_track.py",
        "scale_factor_k": k, "target_length_m": TARGET_LENGTH, "raw_lap_length_m": lap_len_raw,
        "vehicle_ggv": f"Config/{VEHICLE}/ggv.csv", "gg_scale": GG_SCALE, "v_max_mps": v_max,
        "combined_acc_shape": "circle",
        "n_points": len(s), "ds_mean_m": float(np.mean(np.diff(s_closed))),
        "kappa_abs_max": float(np.max(np.abs(kappa))), "min_radius_m": float(1 / np.max(np.abs(kappa))),
        "v_ref_min_mps": float(v_ref.min()), "v_ref_max_mps": float(v_ref.max()),
        "a_lon_min_max": [float(a_ref.min()), float(a_ref.max())],
        "a_lat_abs_max": float(np.abs(a_lat).max()),
        "ey_max_min_m": float(ey_max.min()), "ey_min_max_m": float(ey_min.max()),
        "raceline_offset_from_centerline_m": [float(offset.min()), float(offset.max())],
        "lap_time_estimate_s": lap_time,
    }
    with open(os.path.join(out_dir, name + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for key, val in meta.items():
        print(f"{key:36s} {val}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        ax[0, 0].plot(cx, cy, "k--", lw=0.6, label="centerline")
        ax[0, 0].plot(X_i, Y_i, "k-", lw=0.8); ax[0, 0].plot(X_o, Y_o, "k-", lw=0.8)
        sc = ax[0, 0].scatter(px, py, c=v_ref, s=2, cmap="viridis", label="raceline")
        fig.colorbar(sc, ax=ax[0, 0], label="v_ref [m/s]")
        ax[0, 0].plot(px[0], py[0], "ro"); ax[0, 0].set_aspect("equal"); ax[0, 0].set_title(f"{TRACK} (x{k:.2f})")
        ax[0, 1].plot(s, v_ref); ax[0, 1].set_ylabel("v_ref [m/s]")
        ax[0, 1].twinx().plot(s, kappa, "r", lw=0.6)
        ax[0, 1].set_title("v_ref (blue), kappa (red)")
        ax[1, 0].plot(s, ey_max, label="e_y max (left)"); ax[1, 0].plot(s, ey_min, label="e_y min (right)")
        ax[1, 0].axhline(0, color="k", lw=0.5); ax[1, 0].set_xlabel("s [m]"); ax[1, 0].set_ylabel("[m]"); ax[1, 0].legend()
        ax[1, 1].plot(a_lat, a_ref, ".", ms=2); ax[1, 1].set_xlabel("a_lat [m/s²]"); ax[1, 1].set_ylabel("a_lon [m/s²]")
        ax[1, 1].set_title("g-g of v_ref profile"); ax[1, 1].set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(out_dir, name + "_check.png")
        fig.savefig(path, dpi=110)
        print("plot saved:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="save a check plot next to the outputs")
    main(ap.parse_args().plot)
