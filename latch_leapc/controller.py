"""Build the LATCH OCP and expose its terminal weights through Leap-C."""
from copy import deepcopy
from pathlib import Path
import time

import casadi as ca
import numpy as np

from .runtime import ROOT, parameters
from .vehicle import prediction_model, rk4_map, preview


def build_ocp(settings, N):
    from acados_template import AcadosOcp
    from leap_c.parameters import AcadosParameterManager

    mp, cfg = parameters()
    original = prediction_model()
    dt = settings["prediction_dt"]
    manager = AcadosParameterManager(N_horizon=N)
    road = manager.register_parameter("road", np.array([0., 15., -10., 10.]), differentiable=False)
    P0 = np.array([mp["P_ey"], mp["P_epsi"], mp["P_v"]])
    terminal = manager.register_parameter("P_terminal", P0, differentiable=True)
    x, u = ca.SX.sym("state", 8), ca.SX.sym("action", 2)
    step = rk4_map(original, dt, settings["rk_steps"])
    ocp = AcadosOcp()
    ocp.model.name = f"latch_frenet_N{N}"
    ocp.model.x, ocp.model.u = x, u
    # model.p / p_global are assigned by the manager, not copied from LATCH.
    ocp.model.disc_dyn_expr = step(x, u, road)
    error = ca.vertcat(x[1], x[2], x[3] - road[1])
    Q = np.array([mp["q_ey"], mp["q_epsi"], mp["q_v"]])
    R = np.array([mp["r_jerk"], mp["r_steering_rate"]])
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.005 * (ca.dot(Q*error, error) + ca.dot(R*u, u))
    ocp.model.cost_expr_ext_cost_e = 0.005 * ca.dot(terminal*error, error)

    ggv = np.genfromtxt(cfg / "ggv.csv", delimiter=",", names=True)
    ax = ca.interpolant("latch_ax", "linear", [ggv["vel_max_mps"]], ggv["ax_max_mps2"])
    ay = ca.interpolant("latch_ay", "linear", [ggv["vel_max_mps"]], ggv["ay_max_mps2"])
    ax_limit = ca.if_else(x[7] < 0, -original.acc_min, ax(x[3]))
    acc_circle = (x[7]/ax_limit)**2 + (x[3]*x[5]/ay(x[3]))**2
    h = ca.vertcat(acc_circle, x[1] - road[2], road[3] - x[1])
    width = original.params.veh_width/2 + mp["track_margin"]
    low, high = np.array([0., width, width]), np.array([1., 1e4, 1e4])
    for suffix in ["", "_0", "_e"]:
        setattr(ocp.model, "con_h_expr" + suffix, h)
        setattr(ocp.constraints, "lh" + suffix, low.copy())
        setattr(ocp.constraints, "uh" + suffix, high.copy())
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([original.jerk_min, original.delta_f_dot_min])
    ocp.constraints.ubu = np.array([original.jerk_max, original.delta_f_dot_max])
    for suffix in ["", "_e"]:
        setattr(ocp.constraints, "idxbx" + suffix, np.array([6, 7]))
        setattr(ocp.constraints, "lbx" + suffix, np.array([original.delta_f_min, original.acc_min]))
        setattr(ocp.constraints, "ubx" + suffix, np.array([original.delta_f_max, original.acc_max]))
        setattr(ocp.constraints, "idxsbx" + suffix, np.array([0, 1]))
    ocp.constraints.x0 = np.array([0., 0., 0., 15., 0., 0., 0., 0.])
    z1h = np.array([mp["L1_pen"], mp["L1_pen_track"], mp["L1_pen_track"]])
    z2h = np.array([mp["L2_pen"], mp["L2_pen_track"], mp["L2_pen_track"]])
    for suffix in ["", "_0", "_e"]:
        setattr(ocp.constraints, "idxsh" + suffix, np.arange(3))
        z1 = z1h if suffix == "_0" else np.r_[[mp["L1_pen"]]*2, z1h]
        z2 = z2h if suffix == "_0" else np.r_[[mp["L2_pen"]]*2, z2h]
        for key, value in [("zl", z1), ("zu", z1), ("Zl", z2), ("Zu", z2)]:
            setattr(ocp.cost, key + suffix, value.copy())
    opts = ocp.solver_options
    opts.N_horizon = N
    opts.tf = N*dt
    opts.integrator_type = "DISCRETE"  # Required for this acados parameter-sensitivity interface.
    opts.nlp_solver_type = "SQP"       # Converged solution for implicit differentiation.
    # Same cost Hessian approximation as the original. Leap-C makes an exact-Hessian
    # backward solver for the KKT sensitivities, including dynamics and constraints.
    # opts.hessian_approx = "GAUSS_NEWTON"
    # GN ignores the dynamics curvature; with the weak (x0.005) cost this stalled SQP
    # far from the solution (full lap: step 662 for N=38, 1047 for N=10). The exact
    # Hessian solved those states in 2-5 iterations and N=38 completed a full lap.
    opts.hessian_approx = "EXACT"
    opts.regularize_method = "MIRROR"
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.qp_solver_ric_alg = 0
    opts.qp_solver_cond_N = N
    opts.qp_solver_iter_max = mp["qp_solver_iter_max"]
    opts.nlp_solver_max_iter = settings["nlp_max_iter"]
    opts.globalization = "MERIT_BACKTRACKING"
    # Previously not set, i.e. the acados default for MERIT_BACKTRACKING:
    # opts.globalization_alpha_min = 0.05
    # With 0.05 the line search got stuck at alpha=0.7^8≈0.058 near the solution
    # (step 287 of the 500-step long baseline: 3x100 SQP iterations, status 2).
    # 0.5 broke the cold-start solve at step 0; 0.2 (with tol_stat 1e-6) completed
    # all 500 steps without retries, max 41 SQP iterations, actions within 4e-6 of before.
    opts.globalization_alpha_min = 0.2
    # for key in ["stat", "eq", "ineq", "comp"]:
    #     setattr(opts, "nlp_solver_tol_" + key, settings["nlp_tolerance"])
    for key in ["eq", "ineq", "comp"]:
        setattr(opts, "nlp_solver_tol_" + key, settings["nlp_tolerance"])
    # Stationarity tolerance loosened from nlp_tolerance (1e-7) as a safety net.
    # opts.nlp_solver_tol_stat = 1e-6
    # With the exact Hessian, stationarity reaches a floor of ~1e-5 at some states
    # (full lap, N=38 step 837), so 1e-6 was unreachable there.
    opts.nlp_solver_tol_stat = 1e-5
    return ocp, manager, P0


def make_initializer(ocp, dt):
    from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer

    class VehicleInitializer(AcadosDiffMpcInitializer):
        template = None

        def single_iterate(self, solver_input):
            if self.template is None:
                self.template = ocp.create_default_initial_iterate().flatten()
            result = deepcopy(self.template)
            x0 = np.asarray(solver_input.x0, dtype=float)
            N = ocp.solver_options.N_horizon
            X = np.tile(x0, (N + 1, 1))
            X[:, 0] += max(x0[3], 1.0)*dt*np.arange(N + 1)
            X[:, 1:3] *= np.exp(-dt*np.arange(N + 1))[:, None]
            result.x = X.ravel()
            result.u[:] = 0.
            return result

    return VehicleInitializer()


def ensure_success(ctx):
    status = np.asarray(ctx.status)
    if np.any(status != 0):
        raise RuntimeError(f"MPC did not converge: status={status.tolist()}. No action was applied.")


class VehicleMpc:
    def __init__(self, settings, N, batch_size=1, verbose=False):
        import torch
        from leap_c.torch import AcadosDiffMpcTorch
        self.settings, self.N = settings, N
        self.ocp, self.manager, self.P0 = build_ocp(settings, N)
        # acados validates the entire OCP hash before binary reuse. Keep the path
        # stable so changing plotting/training code does not force C recompilation.
        cache = ROOT / "latch_leapc_build" / f"N{N}"
        print(f"Creating vehicle MPC: N={N}, T={N*settings['prediction_dt']:.2f}s", flush=True)
        self.module = AcadosDiffMpcTorch(
            self.ocp, self.manager,
            initializer=make_initializer(self.ocp, settings["prediction_dt"]),
            export_directory=Path(cache), n_batch_init=batch_size,
            num_threads_batch_solver=1, dtype=torch.float64, verbose=verbose,
        )
        self.ctx = None

    def batch(self, states, road, terminal=None, ctx=None):
        import torch
        x = torch.as_tensor(states, dtype=torch.float64)
        if x.ndim == 1:
            x = x[None]
        params = {"road": np.ascontiguousarray(road, dtype=float)}
        if terminal is not None:
            if not torch.is_tensor(terminal):
                terminal = torch.as_tensor(terminal, dtype=torch.float64)
            params["P_terminal"] = terminal.expand(x.shape[0], -1)
        result = self.module(x, params=params, ctx=ctx)
        # GN can converge slowly near active soft constraints. Continue a max-iteration
        # result from its own primal/dual iterate, without using its action or gradient.
        # A failed QP / NaN is not treated as a usable warm start.
        for _ in range(2):
            status = np.asarray(result[0].status)
            if not np.any(status == 2) or np.any(~np.isin(status, [0, 2])):
                break
            result = self.module(x, params=params, ctx=result[0])
        ensure_success(result[0])
        if any(not torch.isfinite(v).all() for v in result[1:]):
            raise RuntimeError("Non-finite MPC result; no action was applied.")
        return result

    def solve(self, state, ref, terminal=None):
        import torch
        road = preview(ref, state, self.N, self.settings["prediction_dt"])
        start = time.perf_counter()
        with torch.no_grad():
            ctx, u0, X, U, value = self.batch(state, road, terminal, ctx=self.ctx)
        self.ctx = ctx
        return (u0[0].numpy(), X[0].numpy(), U[0].numpy(),
                float(value[0, 0]), time.perf_counter() - start)
