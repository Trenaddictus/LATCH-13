
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latch13.tracks.f1tenth import (
    load_raceline,
    project_pose,
)

ROOT = Path(__file__).resolve().parents[1]
path = (
    ROOT / "external/f1tenth_racetracks"
    / "Silverstone/Silverstone_raceline.csv"
)

track = load_raceline(path, scale=10.0)

# Verify projection at the first waypoint
dx = track.x[1] - track.x[0]
dy = track.y[1] - track.y[0]
initial_yaw = np.arctan2(dy, dx)

s, ey, epsi = project_pose(
    track, track.x[0], track.y[0], initial_yaw
)

assert abs(ey) < 1e-6
assert abs(epsi) < 1e-6

print("Number of waypoints:", len(track.s))
print("Track length [m]:", track.lap_length)
print("Initial Frenet state:", s, ey, epsi)

# Check whether the CSV heading matches XY geometry
dx_ds = np.gradient(track.x, track.s)
dy_ds = np.gradient(track.y, track.s)
geometric_yaw = np.arctan2(dy_ds, dx_ds)

heading_error = np.arctan2(
    np.sin(track.psi - geometric_yaw),
    np.cos(track.psi - geometric_yaw),
)
print(
    "Median absolute heading mismatch [rad]:",
    np.median(np.abs(heading_error)),
)

plt.figure()
plt.plot(track.x, track.y, label="Raceline")
plt.scatter(track.x[0], track.y[0], s=30, label="Start")
plt.axis("equal")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.show()
