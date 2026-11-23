"""MPC side of the closed loop: warm start, stage parameters and solve with retry.

Works with any solver offering the AcadosOcpSolver interface, including
``controllers.leapc_fixed.FixedThetaSolver``.
"""

import numpy as np


def initialize_solver(solver, xf, N, dt, blend_time=1.0):
    """Cold-start guess from the measured state, blending back to the raceline.

    Speed is held at the measured value (at least 1 m/s) and s advances with
    it. The deviations ey, epsi, vy, r, delta and a decay to zero with time
    constant ``blend_time``, so a car that is far off the raceline (e.g. after
    running wide) gets a guess near where it actually is. At the lap start the
    measured deviations are zero, i.e. a constant-speed guess on the raceline.
    """
    speed = max(float(xf[3]), 1.0)
    for j in range(N + 1):
        guess = xf * np.exp(-j * dt / blend_time)
        guess[0], guess[3] = xf[0] + j * dt * speed, speed
        solver.set(j, "x", xf.copy() if j == 0 else guess)
        if j < N:
            solver.set(j, "u", np.zeros(2))


def shift_warm_start(solver, N):
    """Move the last optimized trajectory forward by one controller interval."""
    states = [solver.get(j, "x") for j in range(1, N + 1)]
    controls = [solver.get(j, "u") for j in range(1, N)]
    controls.append(solver.get(N - 1, "u"))
    for j in range(N):
        solver.set(j, "x", states[j])
        solver.set(j, "u", controls[j])
    solver.set(N, "x", states[-1])


def constant_speed_progress(progress, N, dt, speed):
    """Stage progress for a constant-speed guess (first step, cold restarts)."""
    return progress + np.arange(N + 1) * dt * speed


def shifted_progress(predicted, measured):
    """Next step's stage progress from the last solution's predicted s.

    Shift by one stage, extrapolate the last stage linearly, and anchor the
    first stage at the newly measured progress.
    """
    shifted = np.r_[predicted[1:], 2 * predicted[-1] - predicted[-2]]
    return shifted + (measured - predicted[1])


def stage_parameters(track, progress):
    """Per-stage [curvature, ey_min, ey_max] at the given stage progress."""
    return np.column_stack((track.curvature(progress), *track.edges(progress)))


def prediction_snapshot(solver, N, parameters):
    """Rows: eight states, two controls, [curvature, ey_min, ey_max]; terminal u is NaN."""
    snapshot = np.full((N + 1, 13), np.nan)
    for j in range(N + 1):
        snapshot[j, :8] = np.asarray(solver.get(j, "x")).reshape(8)
        snapshot[j, 10:] = parameters[j]
        if j < N:
            snapshot[j, 8:10] = np.asarray(solver.get(j, "u")).reshape(2)
    return snapshot


def solver_diagnostics(solver):
    """Best-effort diagnostics must not hide the original failure."""
    result = {}
    for name in ("time_tot", "nlp_iter", "residuals"):
        try:
            value = solver.get_residuals() if name == "residuals" else solver.get_stats(name)
            result[name] = np.asarray(value).tolist()
        except Exception as error:
            result[name] = {"unavailable": str(error)}
    return result


def solve_with_recovery(solver, xf, parameters, N, dt):
    """Solve once, then try one cold restart at the SAME measured state.

    Record rejected iterates before reset. No failed solution is applied to the
    plant. This follows TUM's reinitialization approach, but requires a successful
    retry rather than continuing with the failed solve's first control.
    """
    attempts = []
    total_solver_ms = 0.0
    for attempt in range(2):
        if attempt:
            solver.reset()
            initialize_solver(solver, xf, N, dt)
        solver.constraints_set(0, "lbx", xf)
        solver.constraints_set(0, "ubx", xf)
        for j, stage in enumerate(parameters):
            solver.set(j, "p", stage)
        status = int(solver.solve())
        details = solver_diagnostics(solver)
        total_solver_ms += float(np.asarray(details["time_tot"]).reshape(-1)[0]) * 1000
        details["status"] = status
        if status != 0:
            details["rejected_prediction"] = prediction_snapshot(solver, N, parameters).tolist()
        attempts.append(details)
        if status == 0:
            break
    return status, attempts, total_solver_ms
