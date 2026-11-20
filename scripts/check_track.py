"""Sanity-check a track: projection, raceline-relative edges, and plots.

Checks the ROADMAP step 1 acceptance criteria: ey_min(s) < 0 < ey_max(s),
asymmetric bounds in corners, and edges reconstructed from
raceline + ey_min/max * normal coinciding with the edge polylines.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latch13.tracks.f1tenth import load_track

ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"


def distance_to_polyline(points, polyline):
    """Distance from each point to a closed polyline."""
    start, edge = polyline, np.roll(polyline, -1, axis=0) - polyline
    best = np.full(len(points), np.inf)
    for i in range(0, len(points), 256):
        p = points[i:i + 256, None]
        alpha = np.clip(((p - start) * edge).sum(-1) / (edge * edge).sum(-1), 0, 1)
        best[i:i + 256] = np.linalg.norm(p - (start + alpha[..., None] * edge), axis=-1).min(1)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--corner-curvature", type=float, default=0.02,
                        help="|kappa| above which a vertex counts as a corner [1/m]")
    parser.add_argument("--output", help="save the figure here instead of showing it")
    args = parser.parse_args()

    track = load_track(TRACK_ROOT, args.track, args.scale)
    print(f"{args.track} x{args.scale}: {len(track.s)} raceline vertices, lap {track.length:.1f} m")

    yaw0 = np.arctan2(*track.segments[0][::-1])
    s0, ey0, epsi0 = track.project(*track.points[0], yaw0)
    assert abs(ey0) < 1e-9 and abs(epsi0) < 1e-9, "first vertex must project onto itself"

    inside = np.all(track.ey_min < 0) and np.all(track.ey_max > 0)
    print(f"ey_min in [{track.ey_min.min():.2f}, {track.ey_min.max():.2f}] m, "
          f"ey_max in [{track.ey_max.min():.2f}, {track.ey_max.max():.2f}] m -> "
          f"raceline inside track: {inside}")

    corner = np.abs(track.kappa) > args.corner_curvature
    # Positive curvature turns left, so the inner edge is on the left.
    inner = np.where(track.kappa > 0, track.ey_max, -track.ey_min)[corner]
    outer = np.where(track.kappa > 0, -track.ey_min, track.ey_max)[corner]
    print(f"corners (|kappa| > {args.corner_curvature}): {corner.sum()} vertices, "
          f"closer to inner edge at {np.mean(inner < outer):.0%} "
          f"(median inner {np.median(inner):.2f} m, outer {np.median(outer):.2f} m)")

    normals = track.vertex_normals()
    left = track.points + track.ey_max[:, None] * normals
    right = track.points + track.ey_min[:, None] * normals
    error = max(distance_to_polyline(left, track.left).max(),
                distance_to_polyline(right, track.right).max())
    print(f"reconstructed edges: max distance to edge polylines {error:.2e} m")

    fig, (xy, offsets) = plt.subplots(1, 2, figsize=(15, 7), width_ratios=(1, 1.3))
    close = lambda points: np.vstack((points, points[:1])).T  # noqa: E731
    xy.plot(*close(track.left), color="0.4", linewidth=0.8, label="Edges (centerline CSV)")
    xy.plot(*close(track.right), color="0.4", linewidth=0.8)
    xy.plot(*close(left), "c:", linewidth=1.2, label="Raceline + ey_min/max · normal")
    xy.plot(*close(right), "c:", linewidth=1.2)
    xy.plot(*close(track.points), "r-", linewidth=0.8, label="Raceline")
    xy.scatter(*track.points[0], s=30, color="k", zorder=5, label="Start")
    xy.set(xlabel="X [m]", ylabel="Y [m]", aspect="equal", title="Raceline and track edges")
    xy.legend(fontsize="small")

    offsets.fill_between(track.s, track.ey_min, track.ey_max, color="0.9", label="Track")
    offsets.plot(track.s, track.ey_max, label="ey_max (left edge)")
    offsets.plot(track.s, track.ey_min, label="ey_min (right edge)")
    offsets.axhline(0, color="r", linewidth=0.8, label="Raceline")
    curvature = offsets.twinx()
    curvature.plot(track.s, track.kappa, color="0.5", linewidth=0.6, alpha=0.7)
    curvature.set_ylabel("Curvature [1/m]", color="0.5")
    offsets.set(xlabel="s [m]", ylabel="Lateral offset from raceline [m]",
                title="Edge distances relative to the raceline")
    offsets.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=120)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
