"""F1TENTH track geometry: the closed raceline (the MPC's Frenet frame) and edges.

The raceline is a closed piecewise-linear curve through the CSV's XY points.
Arc length is rebuilt from that polyline and curvature from smoothed finite
differences, so projection, curvature and plotting all use one geometry.

Lateral offsets follow ``frenet_model.py``: ``ey`` is positive to the left of
the driving direction. Track edges come from the centerline CSV (track widths
are given relative to the centerline) and are converted to lateral distances
from the raceline, ``ey_min(s) < 0 < ey_max(s)``.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def numeric_csv(path, min_cols):
    """Read numeric CSV rows; tolerate comments, headers, comma/semicolon."""
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = ";" if ";" in line else ","
            parts = [part.strip() for part in line.split(sep)]
            try:
                values = [float(item) for item in parts[:min_cols]]
            except ValueError:
                continue
            if len(values) == min_cols and np.all(np.isfinite(values)):
                rows.append(values)
    if len(rows) < 5:
        raise ValueError(f"Not enough numeric rows in {path}")
    return np.asarray(rows, dtype=float)


def _open_polyline(points):
    """Drop a repeated closing vertex; closure is implied everywhere."""
    points = np.asarray(points, dtype=float)
    return points[:-1] if np.linalg.norm(points[-1] - points[0]) < 1e-6 else points


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _ray_distance(origins, directions, polyline, chunk=256):
    """Distance along each unit ray to its first crossing of a closed polyline.

    Returns inf where a ray does not cross the polyline.
    """
    start = polyline
    edge = np.roll(polyline, -1, axis=0) - polyline
    distance = np.full(len(origins), np.inf)
    for i in range(0, len(origins), chunk):
        o, d = origins[i:i + chunk, None, :], directions[i:i + chunk, None, :]
        # Solve o + t d = start + u edge for t > 0, u in [0, 1].
        w = start[None] - o
        denominator = _cross(d, edge[None])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(w, edge[None]) / denominator
            u = _cross(w, d) / denominator
        hit = (np.abs(denominator) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        distance[i:i + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return distance


@dataclass
class Raceline:
    """Closed raceline with optional track edges; all arrays are per vertex.

    ``points`` (n, 2) are the vertices without a repeated closing vertex;
    ``s`` is the arc length at each vertex (``s[0] = 0``); ``kappa`` the
    smoothed curvature. ``center``, ``left`` and ``right`` are Cartesian
    polylines from the centerline CSV; ``ey_min``/``ey_max`` the lateral
    distances from each raceline vertex to the right/left edge.
    """

    points: np.ndarray
    s: np.ndarray
    kappa: np.ndarray
    center: np.ndarray | None = None
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    ey_min: np.ndarray | None = None
    ey_max: np.ndarray | None = None
    segments: np.ndarray = field(init=False, repr=False)
    segment_lengths: np.ndarray = field(init=False, repr=False)
    length: float = field(init=False)

    def __post_init__(self):
        self.segments = np.roll(self.points, -1, axis=0) - self.points
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        self.length = float(self.segment_lengths.sum())

    def _periodic(self, values, progress):
        return np.interp(np.mod(progress, self.length), np.r_[self.s, self.length],
                         np.r_[values, values[0]])

    def curvature(self, progress):
        return self._periodic(self.kappa, progress)

    def edges(self, progress):
        """Lateral distances (ey_min, ey_max) to the right and left edge."""
        if self.ey_min is None:
            raise ValueError("Raceline was loaded without track edges")
        return self._periodic(self.ey_min, progress), self._periodic(self.ey_max, progress)

    def pose(self, progress):
        """Raceline XY, heading and segment index at (array-valued) progress."""
        s = np.mod(np.asarray(progress, dtype=float), self.length)
        index = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 1)
        segments = self.segments[index]
        alpha = np.clip((s - self.s[index]) / self.segment_lengths[index], 0, 1)
        xy = self.points[index] + alpha[..., None] * segments
        return xy, np.arctan2(segments[..., 1], segments[..., 0]), index

    def cartesian(self, progress, ey):
        """XY of Frenet positions (progress, ey)."""
        xy, yaw, _ = self.pose(progress)
        normal = np.stack((-np.sin(yaw), np.cos(yaw)), axis=-1)
        return xy + np.asarray(ey, dtype=float)[..., None] * normal

    def project(self, x, y, heading):
        """Project a Cartesian pose; return wrapped (s, ey, epsi)."""
        p = np.array([x, y], dtype=float)
        squared = np.maximum(np.einsum("ij,ij->i", self.segments, self.segments), 1e-12)
        alpha = np.clip(np.einsum("ij,ij->i", p - self.points, self.segments) / squared, 0.0, 1.0)
        candidates = self.points + alpha[:, None] * self.segments
        index = int(np.argmin(np.sum((candidates - p) ** 2, axis=1)))
        d, offset = self.segments[index], p - candidates[index]
        ey = _cross(d, offset) / self.segment_lengths[index]
        psi_ref = np.arctan2(d[1], d[0])
        epsi = np.arctan2(np.sin(heading - psi_ref), np.cos(heading - psi_ref))
        s = (self.s[index] + alpha[index] * self.segment_lengths[index]) % self.length
        return float(s), float(ey), float(epsi)

    def vertex_normals(self):
        """Unit left normals at the vertices (central-difference tangents)."""
        tangent = np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        return np.column_stack((-tangent[:, 1], tangent[:, 0]))

    def to_npz(self, path):
        """Save geometry for replay; keys stay readable by older runs' tools."""
        arrays = {"points": self.points, "s": self.s, "kappa": self.kappa, "L": self.length}
        for name in ("center", "left", "right", "ey_min", "ey_max"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            optional = {name: data[name] for name in ("center", "left", "right", "ey_min", "ey_max")
                        if name in data.files}
            return cls(data["points"], data["s"], data["kappa"], **optional)


def track_edge_offsets(raceline, left, right):
    """Lateral distances from each raceline vertex to the left and right edge.

    Intersects the vertex normal with each edge polyline: ``ey_max`` is the
    distance to the left edge along +normal, ``ey_min`` minus the distance to
    the right edge along -normal.
    """
    normals = raceline.vertex_normals()
    ey_max = _ray_distance(raceline.points, normals, left)
    ey_min = -_ray_distance(raceline.points, -normals, right)
    if not (np.all(np.isfinite(ey_max)) and np.all(np.isfinite(ey_min))):
        raise ValueError("Some raceline normals do not reach a track edge")
    return ey_min, ey_max


def load_edges(path, scale=1.0):
    """Centerline CSV (x, y, w_tr_right, w_tr_left) to center, left, right polylines."""
    data = numeric_csv(path, 4)
    data[:, :] *= scale
    if np.linalg.norm(data[-1, :2] - data[0, :2]) < 1e-6:
        data = data[:-1]
    center = data[:, :2]
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return center, center + data[:, 3, None] * normal, center - data[:, 2, None] * normal


def load_raceline(path, scale=1.0, *, smoothing=3.0):
    """Raceline CSV (s; x; y; ...) to a Raceline without edges."""
    points = _open_polyline(numeric_csv(path, 7)[:, 1:3] * scale)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(lengths <= 1e-7):
        raise ValueError("Raceline has duplicate adjacent coordinates")
    s = np.r_[0.0, np.cumsum(lengths[:-1])]
    dx, dy = np.gradient(points[:, 0], s), np.gradient(points[:, 1], s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-8)
    return Raceline(points, s, gaussian_filter1d(kappa, sigma=smoothing, mode="wrap"))


def load_track(root, name, scale=1.0):
    """Raceline of ``root/name`` with edges and raceline-relative edge offsets."""
    root = Path(root) / name
    raceline = load_raceline(root / f"{name}_raceline.csv", scale)
    center, left, right = load_edges(root / f"{name}_centerline.csv", scale)
    raceline.center, raceline.left, raceline.right = center, left, right
    raceline.ey_min, raceline.ey_max = track_edge_offsets(raceline, left, right)
    return raceline
