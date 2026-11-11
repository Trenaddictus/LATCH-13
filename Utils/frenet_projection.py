"""
Cartesian <-> Frenet conversion with respect to a closed reference line (raceline).

Conventions (match the Frenet NMPC in CLAUDE.md and Utils/preprocess_f1tenth_track.py):
    s      arc length along the reference line [m], in [0, L) for a closed track (L = lap length)
    e_y    signed lateral offset from the reference line [m], positive = LEFT of the driving direction
    e_psi  heading error psi - psi_ref(s) [rad], wrapped to (-pi, pi]
    v_x, v_y, psi_dot, delta are body-frame quantities and pass through unchanged.

Cartesian simulator state (TUM-CONTROL):  [pos_x, pos_y, yaw, v_lon, v_lat, yawrate, delta_f, (a)]
Frenet state (project):                    [s, e_y, e_psi, v_x, v_y, psi_dot, delta]

The reference line is a closed polyline (points every ~2.6 m) with linearly interpolated heading.
A point is projected along the normal of the interpolated heading, so e_y is smooth across reference
points and frenet_to_cartesian() is the exact inverse of cartesian_to_frenet(). The polyline itself
deviates from the smooth curve by at most ds^2 / (8 R_min) ~ 3 cm on the scaled Silverstone.
"""

import json

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))


class FrenetReference:
    """Closed reference line loaded from a TUM-style reftraj JSON that contains the Frenet keys
    (s, kappa, ey_max, ey_min), e.g. Trajectories/reftraj_silverstone_edgar.json."""

    def __init__(self, reftraj_file, search_window=50):
        with open(reftraj_file, "r") as f:
            ref = json.load(f)
        self.x = np.asarray(ref["pos_x"], dtype=float)
        self.y = np.asarray(ref["pos_y"], dtype=float)
        self.s = np.asarray(ref["s"], dtype=float)
        self.psi = np.unwrap(np.asarray(ref["ref_yaw"], dtype=float))
        self.kappa = np.asarray(ref["kappa"], dtype=float)
        self.v_ref = np.asarray(ref["ref_v"], dtype=float)
        self.ey_max = np.asarray(ref["ey_max"], dtype=float)
        self.ey_min = np.asarray(ref["ey_min"], dtype=float)
        self.n = len(self.s)

        # closed polyline: segment i goes from point i to point (i+1) % n
        self.seg_dx = np.roll(self.x, -1) - self.x
        self.seg_dy = np.roll(self.y, -1) - self.y
        self.seg_len = np.hypot(self.seg_dx, self.seg_dy)
        self.L = self.s[-1] + self.seg_len[-1]                   # lap length
        # heading change over each segment (handles the 2*pi jump at the lap closure)
        self.seg_dpsi = wrap_angle(np.roll(self.psi, -1) - self.psi)

        self.search_window = search_window                       # segments searched around a guess

    # ------------------------------------------------------------------ reference quantities at s
    def wrap_s(self, s):
        return np.mod(s, self.L)

    def _interp_closed(self, s, values):
        s_c = np.append(self.s, self.L)
        v_c = np.append(values, values[0])
        return np.interp(self.wrap_s(s), s_c, v_c)

    def ref_at(self, s):
        """Reference quantities at arc length s (any real s; wrapped onto the lap).
        Returns dict with x, y, psi, kappa, v_ref, ey_max, ey_min (linear interpolation)."""
        s = np.asarray(s, dtype=float)
        s_w = self.wrap_s(s)
        i = np.clip(np.searchsorted(self.s, s_w, side="right") - 1, 0, self.n - 1)
        t = (s_w - self.s[i]) / self.seg_len[i]
        return {
            "x": self.x[i] + t * self.seg_dx[i],
            "y": self.y[i] + t * self.seg_dy[i],
            "psi": wrap_angle(self.psi[i] + t * self.seg_dpsi[i]),
            "kappa": self._interp_closed(s, self.kappa),
            "v_ref": self._interp_closed(s, self.v_ref),
            "ey_max": self._interp_closed(s, self.ey_max),
            "ey_min": self._interp_closed(s, self.ey_min),
        }

    # ------------------------------------------------------------------ Cartesian -> Frenet
    def project_point(self, px, py, s_guess=None):
        """Project a point onto the reference line. Returns (s, e_y, psi_ref, segment index).
        With s_guess (e.g. s from the previous time step), only +-search_window segments around it
        are searched: faster, and avoids jumping to a nearby but different part of the track."""
        if s_guess is None:
            idx = np.arange(self.n)
        else:
            i0 = int(np.searchsorted(self.s, self.wrap_s(s_guess), side="right") - 1)
            idx = np.arange(i0 - self.search_window, i0 + self.search_window + 1) % self.n

        # 1) coarse: nearest segment by orthogonal projection onto the polyline
        ax, ay = self.x[idx], self.y[idx]
        dx, dy, l2 = self.seg_dx[idx], self.seg_dy[idx], self.seg_len[idx] ** 2
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        fx, fy = ax + t * dx, ay + t * dy
        i_c = int(idx[int(np.argmin((px - fx) ** 2 + (py - fy) ** 2))])

        # 2) refine on segments i_c-1, i_c, i_c+1: find t such that the point lies on the normal of
        #    the INTERPOLATED heading psi(t) = psi_i + t*dpsi_i (not the segment normal). This makes
        #    e_y smooth across reference points and frenet_to_cartesian() its exact inverse.
        cand = np.array([i_c - 1, i_c, i_c + 1]) % self.n
        a_x, a_y = self.x[cand], self.y[cand]
        d_x, d_y = self.seg_dx[cand], self.seg_dy[cand]
        psi0, dpsi = self.psi[cand], self.seg_dpsi[cand]
        t = np.clip(((px - a_x) * d_x + (py - a_y) * d_y) / self.seg_len[cand] ** 2, 0.0, 1.0)
        for _ in range(4):                                     # Newton on f(t) = (p - foot(t)) . T(t)
            ps = psi0 + t * dpsi
            cx, cy = px - a_x - t * d_x, py - a_y - t * d_y
            f = cx * np.cos(ps) + cy * np.sin(ps)
            df = -(d_x * np.cos(ps) + d_y * np.sin(ps)) + dpsi * (-cx * np.sin(ps) + cy * np.cos(ps))
            t = t - f / df
        ok = (t >= -1e-6) & (t <= 1 + 1e-6)
        t = np.clip(t, 0.0, 1.0)
        ps = psi0 + t * dpsi
        cx, cy = px - a_x - t * d_x, py - a_y - t * d_y
        dist2 = np.where(ok, cx**2 + cy**2, np.inf)
        if not np.isfinite(dist2).any():                       # fallback (should not happen on track)
            dist2 = cx**2 + cy**2
        k = int(np.argmin(dist2))
        i, tk = int(cand[k]), float(t[k])

        s = self.wrap_s(self.s[i] + tk * self.seg_len[i])
        e_y = -cx[k] * np.sin(ps[k]) + cy[k] * np.cos(ps[k])   # along left normal of psi(t)
        psi_ref = wrap_angle(ps[k])
        return s, e_y, psi_ref, i

    def cartesian_to_frenet(self, x_cart, s_guess=None):
        """Map a Cartesian state [pos_x, pos_y, yaw, v_lon, v_lat, yawrate, delta_f, (a)] to the
        Frenet state [s, e_y, e_psi, v_x, v_y, psi_dot, delta]. Extra trailing states (e.g. a) are
        appended unchanged."""
        x_cart = np.asarray(x_cart, dtype=float)
        s, e_y, psi_ref, _ = self.project_point(x_cart[0], x_cart[1], s_guess)
        e_psi = wrap_angle(x_cart[2] - psi_ref)
        return np.concatenate(([s, e_y, e_psi], x_cart[3:]))

    def frenet_to_cartesian(self, x_fren):
        """Inverse map: [s, e_y, e_psi, ...] -> [pos_x, pos_y, yaw, ...] (rest unchanged).
        Uses the interpolated reference heading for the normal direction."""
        x_fren = np.asarray(x_fren, dtype=float)
        r = self.ref_at(x_fren[0])
        px = r["x"] - x_fren[1] * np.sin(r["psi"])
        py = r["y"] + x_fren[1] * np.cos(r["psi"])
        yaw = wrap_angle(r["psi"] + x_fren[2])
        return np.concatenate(([px, py, yaw], x_fren[3:]))

    def project_trajectory(self, X_cart):
        """Project a whole Cartesian trajectory (rows = time steps), using the previous s as guess.
        Returns the Frenet trajectory and an unwrapped s (continuous over lap boundaries)."""
        X_cart = np.atleast_2d(X_cart)
        X_fren = np.empty_like(X_cart, dtype=float)
        s_guess = None
        for k, xc in enumerate(X_cart):
            X_fren[k] = self.cartesian_to_frenet(xc, s_guess)
            s_guess = X_fren[k, 0]
        s_unwrapped = unwrap_s(X_fren[:, 0], self.L)
        return X_fren, s_unwrapped


def unwrap_s(s, L):
    """Make a sequence of wrapped s values continuous across the start/finish line."""
    s = np.asarray(s, dtype=float)
    ds = np.diff(s)
    jumps = np.where(ds < -L / 2, L, np.where(ds > L / 2, -L, 0.0))
    return s + np.concatenate(([0.0], np.cumsum(jumps)))
