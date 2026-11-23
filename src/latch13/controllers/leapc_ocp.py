"""Leap-C differentiable MPC with a learnable diagonal terminal cost (ROADMAP step 8).

Dynamics, stage cost, constraints and solver settings come from ``ocp.build_ocp``;
this module changes only what Leap-C needs:

* dynamics become an explicit RK4 map (``DISCRETE``), because acados provides
  parameter sensitivities only for discrete dynamics. Substeps match the ERK
  integrator of the Phase 1 controller.
* expressions are rebuilt in SX around the parameter manager's symbols.
* the terminal cost reads theta (``p_global``, differentiable). Curvature and
  track edges stay non-differentiable stage parameters ("track").

Terminal cost, docs/PROBLEM_SPEC.md section 3, stage A:

    V_f = xi' L L' xi + fixed * [vy, r, delta]^2
    xi  = [ey, epsi, softplus(vx - v_threshold)]
    L   = diag(exp(log_weights / 2))

Stage B (full matrix) only has to fill the lower triangle of L.

As with ``ocp.py``, external/TUM-CONTROL must be on sys.path.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters import AcadosParameterManager
from leap_c.torch import AcadosDiffMpcTorch

from latch13.controllers.ocp import (
    QuadraticTerminalCost, SolverOptions, build_ocp, zero_terminal_cost,
)

# Use whenever gradients through the MPC matter (SAC-FOP). At the Phase 1
# stationarity tolerance (1e-3) the forward solve is fine for driving, but
# about 36 % of the nonzero du0/dtheta miss the 1e-3 accuracy target and about
# 4 % are off by more than 10 % (active set not resolved). Costs about 2.2x the
# solve time. SAC-ZOP does not need it.
SENSITIVITY_SOLVER_OPTIONS = SolverOptions(
    nlp_solver_tol_stat=1e-8,
    nlp_solver_tol_eq=1e-10,
    nlp_solver_tol_ineq=1e-10,
    nlp_solver_tol_comp=1e-9,
    qp_solver_tol_stat=1e-11,
    qp_solver_tol_eq=1e-11,
    qp_solver_tol_ineq=1e-11,
    qp_solver_tol_comp=1e-11,
    nlp_solver_max_iter=2000,
)


@dataclass(frozen=True)
class DiagonalTerminalTheta:
    """Initial values and RL bounds of theta.

    Keep defaults strictly inside the bounds: the SAC actor squashes with tanh,
    so a default on a bound has almost no gradient and no exploration.
    """

    weights: tuple = (40.0, 20.0, 0.01)  # ey, epsi, overspeed
    weights_min: tuple = (1.0, 1.0, 1e-4)
    weights_max: tuple = (1000.0, 1000.0, 100.0)
    v_threshold: float = 25.0
    v_threshold_min: float = 5.0
    v_threshold_max: float = 30.0
    softplus_beta: float = 2.0

    def defaults(self):
        return OrderedDict(
            log_weights=np.log(np.asarray(self.weights, dtype=float)),
            v_threshold=np.array([self.v_threshold], dtype=float),
        )

    def space(self):
        low = OrderedDict(
            log_weights=np.log(np.asarray(self.weights_min, dtype=float)),
            v_threshold=np.array([self.v_threshold_min], dtype=float),
        )
        high = OrderedDict(
            log_weights=np.log(np.asarray(self.weights_max, dtype=float)),
            v_threshold=np.array([self.v_threshold_max], dtype=float),
        )
        for name, value in self.defaults().items():
            if not np.all((low[name] < value) & (value < high[name])):
                raise ValueError(f"Default {name} must lie strictly inside its bounds")
        return gym.spaces.Dict(OrderedDict(
            (name, gym.spaces.Box(low[name], high[name], dtype=np.float64)) for name in low
        ))


def _rk4(rhs, x, u, p, dt, substeps):
    """Explicit RK4 map over one shooting interval with constant controls."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = rhs(x, u, p)
        k2 = rhs(x + h / 2 * k1, u, p)
        k3 = rhs(x + h / 2 * k2, u, p)
        k4 = rhs(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _terminal_cost(x, log_weights, v_threshold, beta, fixed):
    overspeed = ca.log1p(ca.exp(beta * (x[3] - v_threshold))) / beta
    xi = ca.vertcat(x[1], x[2], overspeed)
    L = ca.diag(ca.exp(log_weights / 2))
    # The fixed lateral terms are needed for SQP convergence (STEP3_SOLVER_ISSUES.md).
    return (ca.sumsqr(L.T @ xi) + fixed.vy * x[4]**2
            + fixed.yaw_rate * x[5]**2 + fixed.steering_angle * x[6]**2)


def build_leapc_ocp(
    config_dir, N=10, dt=0.08, *,
    theta=None, stage_weights=None, constraints=None, solver_options=None,
):
    """Return (ocp, manager, param_space, defaults) without generating code."""
    theta = theta if theta is not None else DiagonalTerminalTheta()
    param_space, defaults = theta.space(), theta.defaults()

    # x0 is a placeholder; the differentiable MPC sets it at every solve.
    x0 = np.array([0, 0, 0, 10.0, 0, 0, 0, 0])
    ocp = build_ocp(
        config_dir, x0, N, dt, stage_weights=stage_weights, terminal_cost=zero_terminal_cost,
        constraints=constraints, solver_options=solver_options,
    )
    options, phase1 = ocp.solver_options, ocp.model
    substeps = int(np.ravel(options.sim_method_num_steps)[0])
    if options.integrator_type != "ERK" or int(np.ravel(options.sim_method_num_stages)[0]) != 4:
        raise ValueError("Expected the Phase 1 four-stage ERK integrator to reproduce with RK4")
    expressions = ca.Function(
        "phase1_expressions", [phase1.x, phase1.u, phase1.p],
        [phase1.f_expl_expr, phase1.cost_expr_ext_cost, phase1.con_h_expr],
    )

    manager = AcadosParameterManager(N_horizon=N, casadi_type="SX")
    track = manager.register_parameter("track", np.array(ocp.parameter_values, dtype=float))
    symbols = {name: manager.register_parameter(name, value, differentiable=True)
               for name, value in defaults.items()}

    x, u = ca.SX.sym("x", phase1.x.numel()), ca.SX.sym("u", phase1.u.numel())
    _, stage_cost, h = expressions(x, u, track)

    model = AcadosModel()
    model.name = f"latch13_leapc_N{N}"
    model.x, model.u = x, u
    model.p = track
    model.disc_dyn_expr = _rk4(lambda *args: expressions(*args)[0], x, u, track, dt, substeps)
    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = _terminal_cost(
        x, symbols["log_weights"], symbols["v_threshold"], theta.softplus_beta,
        QuadraticTerminalCost(),
    )
    model.con_h_expr = h
    model.con_h_expr_e = h
    ocp.model = model
    options.integrator_type = "DISCRETE"
    return ocp, manager, param_space, defaults


class RacelineInitializer(AcadosDiffMpcInitializer):
    """Cold start from the measured state, blending back to the raceline.

    Same guess as the Phase 1 runner; Leap-C's default zero iterate would put
    the vehicle model at vx = 0, where the slip angles are undefined.
    """

    def __init__(self, ocp, dt, blend_time=1.0):
        self.ocp, self.dt, self.blend_time = ocp, float(dt), float(blend_time)
        self._template = None  # dimensions exist only after solver generation

    def single_iterate(self, solver_input):
        if self._template is None:
            self._template = self.ocp.create_default_initial_iterate().flatten()
        iterate = deepcopy(self._template)
        x0 = np.asarray(solver_input.x0, dtype=float).reshape(-1)
        time = self.dt * np.arange(self.ocp.solver_options.N_horizon + 1)
        speed = max(float(x0[3]), 1.0)
        guess = x0 * np.exp(-time / self.blend_time)[:, None]
        guess[:, 0], guess[:, 3] = x0[0] + time * speed, speed
        guess[0] = x0
        iterate.x = guess.ravel()
        iterate.u = np.zeros_like(iterate.u)
        return iterate


def build_diff_mpc(
    config_dir, export_directory, N=10, dt=0.08, *,
    theta=None, stage_weights=None, constraints=None, solver_options=None,
    n_batch_init=64, num_threads=8, dtype=None, verbose=False,
):
    """Compile the batched differentiable MPC; return (mpc, param_space, defaults).

    Call it with ``params`` holding "log_weights" (B, 3) and "v_threshold" (B, 1)
    tensors, plus "track" (B, N+1, 3) = [curvature, ey_min, ey_max]. Use float64
    states: progress s reaches kilometres. Give each configuration its own
    export directory, so compiled code can be reused.

    The default solver options are those of Phase 1. Pass
    ``solver_options=SENSITIVITY_SOLVER_OPTIONS`` when using the gradients.
    """
    ocp, manager, param_space, defaults = build_leapc_ocp(
        config_dir, N, dt, theta=theta, stage_weights=stage_weights,
        constraints=constraints, solver_options=solver_options,
    )
    mpc = AcadosDiffMpcTorch(
        ocp, manager, initializer=RacelineInitializer(ocp, dt),
        export_directory=export_directory, n_batch_init=n_batch_init,
        num_threads_batch_solver=num_threads, dtype=dtype, verbose=verbose,
    )
    return mpc, param_space, defaults
