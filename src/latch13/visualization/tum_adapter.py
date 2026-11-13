
import numpy as np


def frenet_predictions_to_tum(solver, track, N):
    """
    Convert LATCH-13 Frenet MPC predictions to the
    Cartesian array used by TUM's plotting interface.

    The eighth column contains the acceleration
    command rather than an acceleration state.
    """

    predictions = np.zeros((N + 1, 8))

    for k in range(N + 1):
        xf = np.asarray(
            solver.get(k, "x")
        ).reshape(-1)

        s, ey, epsi, vx, vy, r, delta = xf

        # Locate predicted progress along raceline
        s_wrapped = s % track["L"]

        idx = np.searchsorted(
            track["s"], s_wrapped, side="right"
        ) - 1

        idx = int(np.clip(
            idx, 0, len(track["s"]) - 1
        ))

        segment = track["segments"][idx]
        length = track["segment_lengths"][idx]

        alpha = np.clip(
            (s_wrapped - track["s"][idx]) / length,
            0.0, 1.0,
        )

        reference_xy = (
            track["points"][idx] + alpha * segment
        )

        psi_ref = np.arctan2(
            segment[1], segment[0]
        )

        # Frenet -> Cartesian
        X = reference_xy[0] - ey * np.sin(psi_ref)
        Y = reference_xy[1] + ey * np.cos(psi_ref)
        psi = psi_ref + epsi

        if k < N:
            acceleration = float(
                np.asarray(
                    solver.get(k, "u")
                ).reshape(-1)[0]
            )
        else:
            acceleration = predictions[k - 1, 7]

        predictions[k] = [
            X, Y, psi, vx, vy, r,
            delta, acceleration,
        ]

    return predictions
