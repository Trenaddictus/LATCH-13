
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Raceline:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    source_v: np.ndarray

    @property
    def lap_length(self):
        closing_distance = np.hypot(
            self.x[-1] - self.x[0],
            self.y[-1] - self.y[0],
        )
        return float(self.s[-1] + closing_distance)

    def curvature(self, s):
        # Periodic interpolation around the circuit
        s_closed = np.r_[self.s, self.lap_length]
        k_closed = np.r_[self.kappa, self.kappa[0]]

        return np.interp(
            np.mod(s, self.lap_length),
            s_closed,
            k_closed,
        )


def load_raceline(path: str | Path, scale=1.0):
    # F1TENTH racelines have three header rows,
    # followed by seven semicolon-separated columns.
    data = np.loadtxt(
        path,
        delimiter=";",
        skiprows=3,
        usecols=range(7),
        ndmin=2,
    )

    if data.shape[0] < 3:
        raise ValueError("Raceline requires at least 3 points")

    s = (data[:, 0] - data[0, 0]) * scale

    if not np.all(np.diff(s) > 0):
        raise ValueError("Raceline arc length must increase")

    return Raceline(
        s=s,
        x=data[:, 1] * scale,
        y=data[:, 2] * scale,
        psi=np.unwrap(data[:, 3]),
        kappa=data[:, 4] / scale,
        source_v=data[:, 5],
    )


def project_pose(track: Raceline, x, y, psi):
    """Project Cartesian pose onto a closed raceline."""
    points = np.column_stack((track.x, track.y))
    closed = np.vstack((points, points[0]))

    a = closed[:-1]
    d = np.diff(closed, axis=0)
    p = np.array([x, y])

    # Nearest point on each line segment
    alpha = np.clip(
        np.sum((p - a) * d, axis=1)
        / np.maximum(np.sum(d * d, axis=1), 1e-12),
        0.0,
        1.0,
    )

    projections = a + alpha[:, None] * d
    idx = np.argmin(
        np.sum((projections - p) ** 2, axis=1)
    )

    tangent = d[idx]
    error = p - projections[idx]

    # Signed lateral deviation
    ey = np.linalg.det(
        np.stack((tangent, error))
    ) / np.linalg.norm(tangent)

    path_yaw = np.arctan2(tangent[1], tangent[0])
    epsi = np.arctan2(
        np.sin(psi - path_yaw),
        np.cos(psi - path_yaw),
    )

    s_closed = np.r_[track.s, track.lap_length]
    s = s_closed[idx] + alpha[idx] * (
        s_closed[idx + 1] - s_closed[idx]
    )

    return float(s), float(ey), float(epsi)
