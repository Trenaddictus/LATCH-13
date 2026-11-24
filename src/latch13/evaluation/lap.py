"""Drive one lap with the Phase 1 MPC and score it (used by the Optuna stage-weight tuning).

The loop is the one of ``scripts/run_baseline.py`` (default acados controller,
same plant, warm start, stage parameters and exit check) without its logging,
so a lap here gives the same lap time as the runner.

``warm_start="unshifted"`` keeps the previous plan unshifted and puts the
measured state at its first node. That is what Leap-C does, and what RL training
uses (``--controller leapc-unshifted``), with the acados solver, which compiles
in seconds instead of about a minute.

As with ``ocp.py``, external/TUM-CONTROL must be on sys.path.
"""

from dataclasses import dataclass

import numpy as np

from latch13.controllers.ocp import build_mpc
from latch13.simulation.closed_loop import (
    constant_speed_progress, initialize_solver, shift_warm_start, shifted_progress,
    solve_with_recovery, stage_parameters,
)
from latch13.simulation.plant import ClosedLoopPlant


@dataclass
class LapResult:
    finished: bool
    end: str  # "lap", "left_track", "solver_status_<n>" or "step_limit"
    lap_time: float | None  # s, first lap from the standing start
    lap_times: list  # s, every completed lap
    time: float  # s driven
    distance: float  # m driven
    rms_ey: float
    max_ey: float
    min_margin: float  # closest approach of the body to an edge, m
    mean_solve_ms: float
    max_solve_ms: float


def drive_lap(config_dir, track, N, output_dir, *, dt=0.08, v_init=10.0, stage_weights=None,
              terminal_cost=None, constraints=None, max_steps=None, laps=1, warm_start="shifted"):
    """``laps`` laps from a standing start at s = 0 (raceline, ``v_init``)."""
    if warm_start not in ("shifted", "unshifted"):
        raise ValueError(f"Unknown warm start {warm_start!r}")
    x0 = np.array([0, 0, 0, v_init, 0, 0, 0, 0], dtype=float)
    solver = build_mpc(config_dir, x0, N=N, dt=dt, output_dir=output_dir, stage_weights=stage_weights,
                       terminal_cost=terminal_cost, constraints=constraints)
    plant = ClosedLoopPlant(config_dir, track, dt)
    plant.reset(x0)
    x = plant.frenet
    initialize_solver(solver, x, N, dt)
    horizon = constant_speed_progress(x[0], N, dt, v_init)
    limit = max_steps if max_steps is not None else int(3 * laps * track.length / (v_init * dt))
    ey, margins, solve_ms, lap_steps, end, steps = [], [], [], [], "step_limit", 0
    while steps < limit:
        if warm_start == "unshifted":
            solver.set(0, "x", x)
        status, _, milliseconds = solve_with_recovery(solver, x, stage_parameters(track, horizon), N, dt)
        if status != 0:
            end = f"solver_status_{status}"
            break
        plant.step(np.asarray(solver.get(0, "u"), dtype=float).reshape(2))
        steps += 1
        x = plant.frenet
        ey.append(x[1])
        margins.append(plant.body_margin())
        solve_ms.append(milliseconds)
        if margins[-1] < 0:
            end = "left_track"
            break
        if x[0] >= (len(lap_steps) + 1) * track.length:
            lap_steps.append(steps)
            if len(lap_steps) == laps:
                end = "lap"
                break
        horizon = shifted_progress(np.array([solver.get(j, "x")[0] for j in range(N + 1)]), x[0])
        if warm_start == "shifted":
            shift_warm_start(solver, N)
    ey = np.asarray(ey) if ey else np.zeros(1)
    return LapResult(
        finished=end == "lap", end=end, lap_time=lap_steps[0] * dt if lap_steps else None,
        lap_times=[float(value) for value in np.diff([0, *lap_steps]) * dt],
        time=steps * dt, distance=float(x[0]), rms_ey=float(np.sqrt(np.mean(ey**2))),
        max_ey=float(np.max(np.abs(ey))), min_margin=float(min(margins, default=np.inf)),
        mean_solve_ms=float(np.mean(solve_ms)) if solve_ms else 0.0,
        max_solve_ms=float(np.max(solve_ms)) if solve_ms else 0.0,
    )
