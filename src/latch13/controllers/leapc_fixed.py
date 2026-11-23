"""Drive the Phase 1 closed loop with the Leap-C MPC at a fixed theta (ROADMAP step 9).

``FixedThetaSolver`` offers the part of the AcadosOcpSolver interface that the
closed-loop runner uses, and makes every solve through ``AcadosDiffMpcTorch``,
the module the RL code calls.

Warm starts:

* ``shift=True``: x and u are shifted by one stage, as in Phase 1.
* ``shift=False``: the previous context is passed on unchanged, which is what
  mpc-sac does during training and evaluation.

Either way Leap-C retries a failed warm-started solve once from
``RacelineInitializer``; ``retries`` counts these.
"""

from dataclasses import replace

import numpy as np
import torch

from latch13.controllers.leapc_ocp import build_diff_mpc
from latch13.controllers.ocp import QuadraticTerminalCost


def phase1_theta():
    """Theta that reproduces the Phase 1 terminal cost (overspeed term off).

    These values deliberately lie on or outside the RL bounds: log-weights
    cannot express a zero weight, so the speed weight is 1e-12.
    """
    cost = QuadraticTerminalCost()
    return {
        "log_weights": np.log([cost.ey, cost.epsi, 1e-12]),
        "v_threshold": np.array([30.0]),
    }


class FixedThetaSolver:
    def __init__(self, mpc, theta, N, shift=True):
        self._mpc, self._N, self._shift = mpc, int(N), bool(shift)
        self._theta = {
            name: torch.as_tensor(np.asarray(value)[None], dtype=torch.float64)
            for name, value in theta.items()
        }
        self._x0 = np.zeros(8)
        self._track = np.zeros((self._N + 1, 3))
        self.retries = 0
        self.reset()

    def reset(self):
        """Forget the warm start; the next solve starts from the initializer."""
        self._ctx = None
        self._x = np.zeros((self._N + 1, 8))
        self._u = np.zeros((self._N, 2))

    @property
    def _acados(self):
        return self._mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]

    def set(self, stage, field, value):
        if field == "p":
            self._track[stage] = value
        elif field in ("x", "u"):
            if self._shift:
                getattr(self, f"_{field}")[stage] = value
        else:
            raise ValueError(f"Unsupported field: {field}")

    def constraints_set(self, stage, field, value):
        if stage != 0 or field not in ("lbx", "ubx"):
            raise ValueError("Only the initial state can be changed")
        self._x0 = np.array(value, dtype=float)

    def solve(self):
        warm_start = self._ctx
        if warm_start is not None and self._shift:
            warm_start = replace(warm_start, iterate=replace(
                warm_start.iterate, x=self._x.reshape(1, -1).copy(), u=self._u.reshape(1, -1).copy(),
            ))
        with torch.no_grad():
            ctx, _, x, u, _ = self._mpc(
                torch.as_tensor(self._x0[None], dtype=torch.float64),
                params={**self._theta, "track": self._track[None].copy()},
                ctx=warm_start,
            )
        # retry_rate is also nonzero for a failed cold start, which is not retried.
        self.retries += int(warm_start is not None and ctx.log["retry_rate"] > 0)
        self._ctx = ctx
        self._x, self._u = x[0].numpy().copy(), u[0].numpy().copy()
        return int(ctx.status[0])

    def get(self, stage, field):
        if field == "x":
            return self._x[stage].copy()
        if field == "u":
            return self._u[stage].copy()
        return self._acados.get(stage, field)

    def get_stats(self, field):
        if field == "time_tot":  # includes Leap-C's internal retry
            return self._ctx.log["solving_time"]
        return self._acados.get_stats(field)

    def get_residuals(self):
        return self._acados.get_residuals()


def build_fixed_theta_solver(
    config_dir, export_directory, N=10, dt=0.08, *,
    theta=None, shift=True, stage_weights=None, constraints=None,
):
    """Single-problem Leap-C solver for the closed-loop runner."""
    mpc, _, _ = build_diff_mpc(
        config_dir, export_directory, N, dt, stage_weights=stage_weights,
        constraints=constraints, n_batch_init=1, num_threads=1, dtype=torch.float64,
    )
    return FixedThetaSolver(mpc, theta if theta is not None else phase1_theta(), N, shift=shift)
