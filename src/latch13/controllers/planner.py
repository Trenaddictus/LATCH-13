"""Leap-C planner for mpc-sac: the terminal-cost MPC driven from env observations (ROADMAP step 11).

The planner sees only the observation of ``simulation/env.py``, where s = 0 at
the car. It rebuilds the MPC's inputs from it:

* x0 = [0, ey, epsi, vx, vy, r, delta, a] (``planner_state``);
* the stage parameters [curvature, ey_min, ey_max] at the stage progress of
  the previous plan, shifted by one stage, as in the Phase 1 runner. Without a
  previous plan (episode start), at constant speed;
* the warm start: the previous plan, moved back by its predicted progress of
  one step, so that it is in the frame of the new observation.

The returned context is Leap-C's own, so mpc-sac can pass it back. mpc-sac does
not reset it between validation episodes; the planner starts cold whenever the
observation does not continue the previous plan.

A failed solve returns a non-finite action, which ends the episode in the env
(``mpc_failure``). This suits SAC-ZOP, which does not store actions. SAC-FOP
stores them, so switch it off there (``nan_on_failure=False``).
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.parameters.utils import broadcast_default_param
from leap_c.utils.collate import collate_acados_diff_mpc_ctx
from leapc_lab.planner import ParameterizedPlanner

from latch13.controllers.leapc_ocp import DiagonalTerminalTheta, build_diff_mpc
from latch13.controllers.ocp import ConstraintOptions
from latch13.simulation.env import EnvOptions, planner_road


@dataclass(kw_only=True)
class RacingPlannerConfig:
    """Settings of the planner; the MPC settings default to those of Phase 1.

    ``near_vertices`` and ``near_distance`` must match the env's ``EnvOptions``:
    the observation describes the road only that far, so the horizon at full
    speed must fit inside it (N = 10 does, N = 38 does not).
    ``new_episode_tolerance`` is the largest difference between the observed
    state and the previous plan's prediction for it that still counts as the
    same episode (largest entry of ey, epsi, vx, vy, r, delta, a).
    """

    N: int = 10
    dt: float = 0.08
    theta: DiagonalTerminalTheta = field(default_factory=DiagonalTerminalTheta)
    stage_weights: object = None
    constraints: object = None
    solver_options: object = None
    near_vertices: int = EnvOptions.near_vertices
    near_distance: float = EnvOptions.near_distance
    new_episode_tolerance: float = 0.5
    nan_on_failure: bool = True
    n_batch_init: int = 1
    num_threads: int = 1
    dtype: torch.dtype | None = None


class RacingPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(self, config_dir, export_directory, cfg=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else RacingPlannerConfig()
        speed_max = (self.cfg.constraints or ConstraintOptions()).speed_max
        if self.cfg.N * self.cfg.dt * speed_max > self.cfg.near_distance:
            raise ValueError(f"The horizon reaches {self.cfg.N * self.cfg.dt * speed_max:.0f} m, but the "
                             f"observation describes only {self.cfg.near_distance:.0f} m of road")
        self.diff_mpc, self._param_space, self._default_param = build_diff_mpc(
            config_dir, Path(export_directory), self.cfg.N, self.cfg.dt, theta=self.cfg.theta,
            stage_weights=self.cfg.stage_weights, constraints=self.cfg.constraints,
            solver_options=self.cfg.solver_options, n_batch_init=self.cfg.n_batch_init,
            num_threads=self.cfg.num_threads, dtype=self.cfg.dtype,
        )
        self.cold_starts = 0  # episode starts and broken plans, for tests and logs

    def default_param(self, obs=None):
        return broadcast_default_param(self._default_param, obs)

    def forward(self, obs, action=None, params=None, ctx=None):
        observation = np.asarray(obs.detach().cpu().numpy(), dtype=float)
        x0 = np.column_stack((np.zeros(len(observation)), observation[:, :7]))
        ctx, progress = self._continue_plan(ctx, x0)
        track = np.stack([
            planner_road(row, ahead, self.cfg.near_vertices) for row, ahead in zip(observation, progress)
        ])
        ctx, u0, x, u, value = self.diff_mpc(
            torch.as_tensor(x0, dtype=torch.float64), u0=action,
            params={**(params or {}), "track": track}, ctx=ctx,
        )
        if self.cfg.nan_on_failure:
            failed = torch.as_tensor(np.asarray(ctx.status) != 0, device=u0.device)[:, None]
            u0 = torch.where(failed, torch.full_like(u0, torch.nan), u0)
        return ctx, u0, x, u, value

    def _continue_plan(self, ctx, x0):
        """Warm start and stage progress in the frame of the new observation."""
        N, batch = self.cfg.N, len(x0)
        plan = None
        if ctx is not None and ctx.iterate.x.shape[0] == batch:
            plan = np.asarray(ctx.iterate.x, dtype=float).reshape(batch, N + 1, -1)
            jump = np.abs(plan[:, 1, 1:] - x0[:, 1:]).max(axis=1)
            if np.any(jump > self.cfg.new_episode_tolerance) or np.any(np.asarray(ctx.status) != 0):
                plan = None
        if plan is None:
            self.cold_starts += batch
            speed = np.maximum(x0[:, 3:4], 1.0)
            return None, np.arange(N + 1) * self.cfg.dt * speed

        # Phase 1 lookup: the plan shifted by one stage, last stage extrapolated,
        # anchored at the car (s = 0, so minus the predicted progress of one step).
        moved = plan[:, 1:2, 0]
        s = plan[:, :, 0]
        progress = np.column_stack((s[:, 1:], 2 * s[:, -1] - s[:, -2])) - moved
        rebased = plan.copy()
        rebased[:, :, 0] -= moved
        iterate = replace(ctx.iterate, x=rebased.reshape(batch, -1))
        return replace(ctx, iterate=iterate), progress
