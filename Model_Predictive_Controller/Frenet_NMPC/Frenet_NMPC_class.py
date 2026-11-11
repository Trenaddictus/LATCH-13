"""
Frenet NMPC (SSY226 project) with the same interface as TUM-CONTROL's MPC classes, so main.py,
sim_step and the Logger work unchanged:

    MPC = Frenet_Nonlinear_Model_Predictive_Controller(config_path, MPC_params_file, sim_main_params, X0_MPC)
    u0, pred_X, stats = MPC.solve(current_ref_traj)     # current_ref_traj (time-indexed, Cartesian) is ignored
    MPC.set_initial_state(x_cart)                       # Cartesian [x, y, yaw, vx, vy, r, delta, a]
    MPC.reintialize_solver(x_cart)

Internally the state is projected onto the reference line (Utils/frenet_projection.py); the reference is
indexed by s and passed as stage parameters p_k = [kappa_k, v_ref_k, ey_min_k, ey_max_k], evaluated
along the shifted previous prediction (kappa at the interval midpoint). pred_X is returned in Cartesian
coordinates (8 states, same layout as TUM) for plotting and for sim_step (which uses pred_X[1, 7] = a).
The Frenet prediction is available as self.pred_X_frenet, the current Frenet state as self.x0_frenet.
s is kept unwrapped (continuous over lap boundaries); the reference wraps it internally.
"""

import numpy as np
import yaml
import casadi as cs

from Utils.frenet_projection import FrenetReference
from Model_Predictive_Controller.Frenet_NMPC.Frenet_NMPC_acados_settings import acados_settings


def calculate_velvar_latlon_acc(ggv_file):
    """Speed-dependent ax/ay limits from the ggv table (same as TUM's helper, without pandas)."""
    g = np.genfromtxt(ggv_file, delimiter=",", names=True)
    ax = cs.interpolant("ax_max_interpolant", "linear", [g["vel_max_mps"]], g["ax_max_mps2"])
    ay = cs.interpolant("ay_max_interpolant", "linear", [g["vel_max_mps"]], g["ay_max_mps2"])
    return ax, ay


class Frenet_Nonlinear_Model_Predictive_Controller:

    def __init__(self, config_path, MPC_params_file, sim_main_params, X0_MPC, terminal_cost_fun=None):
        with open(config_path + MPC_params_file, "r") as f:
            self.MPC_params = mp = yaml.safe_load(f)
        self.Tp = sim_main_params["Tp"]
        self.Ts = sim_main_params["Ts"]
        self.Ts_MPC = sim_main_params["Ts_MPC"]
        self.N = int(self.Tp / self.Ts_MPC)

        # reference line (s-indexed)
        self.ref = FrenetReference(sim_main_params["trajectory_path"] + sim_main_params["ref_traj_file"])

        # weights
        self.Q = np.diag([mp["q_ey"], mp["q_epsi"], mp["q_v"]])
        self.R = np.diag([mp["r_jerk"], mp["r_steering_rate"]])
        self.P = np.array([mp["P_ey"], mp["P_epsi"], mp["P_v"]])
        self.ax_lim, self.ay_lim = calculate_velvar_latlon_acc(config_path + mp["lookuptable_gg_limits"])

        self.veh_params_full_path = config_path + sim_main_params["veh_params_file_MPC"]
        self.tire_params_full_path = config_path + sim_main_params["tire_params_file_MPC"]
        self._terminal_cost_fun = terminal_cost_fun

        # initial Frenet state (global projection, no guess)
        X0_MPC = np.asarray(X0_MPC, dtype=float)
        self.x0 = X0_MPC
        self.x0_frenet = self.ref.cartesian_to_frenet(X0_MPC)
        self._build_solver(self.x0_frenet, mp["solver_generate_C_code"], mp["solver_build"])

        self.nx = 8
        self.nu = 2
        self.stats = np.zeros(5)
        self.WMPC = False
        self.costfunction_type = "NONLINEAR_LS"
        self._init_guess(self.x0_frenet)

    # ------------------------------------------------------------------ setup helpers
    def _build_solver(self, x0_frenet, generate, build):
        mp = self.MPC_params
        self.constraint, self.model, self.acados_solver, self.ocp = acados_settings(
            self.Tp, self.N, x0_frenet, self.Q, self.R, self.P, self.ax_lim, self.ay_lim,
            self.veh_params_full_path, self.tire_params_full_path,
            track_margin=mp["track_margin"], L1_pen=mp["L1_pen"], L2_pen=mp["L2_pen"],
            L1_pen_track=mp["L1_pen_track"], L2_pen_track=mp["L2_pen_track"],
            qp_solver=mp["qp_solver"], qp_solver_iter_max=mp["qp_solver_iter_max"],
            nlp_solver_type=mp["nlp_solver_type"], sim_method_num_steps=mp["sim_method_num_steps"],
            terminal_cost_fun=self._terminal_cost_fun,
            solver_generate_C_code=generate, solver_build=build)

    def _init_guess(self, x0f):
        """Constant-speed initial guess along the reference line."""
        v = max(x0f[3], 1.0)
        X = np.tile(x0f, (self.N + 1, 1))
        X[:, 0] = x0f[0] + v * self.Ts_MPC * np.arange(self.N + 1)
        X[:, 1:3] = 0.0
        X[0] = x0f
        self.pred_X_frenet = X
        self.pred_U = np.zeros((self.N, self.nu))
        for k in range(self.N + 1):
            self.acados_solver.set(k, "x", X[k])
        for k in range(self.N):
            self.acados_solver.set(k, "u", self.pred_U[k])

    # ------------------------------------------------------------------ interface used by main.py
    def set_initial_state(self, x_cart):
        x_cart = np.asarray(x_cart, dtype=float)
        s_prev = self.x0_frenet[0]
        xf = self.ref.cartesian_to_frenet(x_cart, s_guess=s_prev)
        # keep s continuous (unwrapped) across the start/finish line
        L = self.ref.L
        xf[0] = xf[0] + L * np.round((s_prev - xf[0]) / L)
        self.x0_frenet = xf
        self.x0 = x_cart
        self.acados_solver.constraints_set(0, "lbx", xf)
        self.acados_solver.constraints_set(0, "ubx", xf)

    def _stage_parameters(self, s_nodes):
        """p_k for k = 0..N from predicted s at the nodes; kappa at interval midpoints."""
        s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        r_mid = self.ref.ref_at(s_mid)
        r_nod = self.ref.ref_at(s_nodes)
        P = np.column_stack((np.append(r_mid["kappa"], r_nod["kappa"][-1]),
                             r_nod["v_ref"], r_nod["ey_min"], r_nod["ey_max"]))
        return P

    def solve(self, current_ref_traj=None):
        N, x0f = self.N, self.x0_frenet

        # warm start: shift previous solution by one interval and re-anchor s at the measured s0
        Xs = np.vstack((self.pred_X_frenet[1:], self.pred_X_frenet[-1]))
        Xs[-1, 0] = Xs[-2, 0] + max(Xs[-2, 3], 1.0) * self.Ts_MPC
        Us = np.vstack((self.pred_U[1:], self.pred_U[-1]))
        Xs[:, 0] += x0f[0] - Xs[0, 0]
        Xs[0] = x0f
        s_nodes = Xs[:, 0]

        self.stage_params = self._stage_parameters(s_nodes)
        for k in range(N + 1):
            self.acados_solver.set(k, "p", self.stage_params[k])
            self.acados_solver.set(k, "x", Xs[k])
        for k in range(N):
            self.acados_solver.set(k, "u", Us[k])

        status = self.acados_solver.solve()
        u0 = self.acados_solver.get(0, "u")

        X = np.array([self.acados_solver.get(k, "x") for k in range(N + 1)])
        U = np.array([self.acados_solver.get(k, "u") for k in range(N)])
        if status == 0 and np.all(np.isfinite(X)):
            self.pred_X_frenet, self.pred_U = X, U
        else:
            # keep a usable trajectory for the next stage parameters / warm start
            self.pred_X_frenet, self.pred_U = Xs, Us

        # stats after every solve (also failed ones): [cost, time_tot, sqp_iter, max qp_iter, status]
        self.stats[0] = self.acados_solver.get_cost()
        self.stats[1] = self.acados_solver.get_stats("time_tot")
        self.stats[2] = self.acados_solver.get_stats("sqp_iter")
        self.stats[3] = np.max(self.acados_solver.get_stats("qp_iter"))
        self.stats[4] = status

        # Cartesian prediction for sim_step (pred_X[1, 7] = a) and plotting
        self.pred_X = np.array([self.ref.frenet_to_cartesian(x) for x in X])
        return u0, self.pred_X, self.stats

    def reintialize_solver(self, x_cart, solver_generate_C_code=False, solver_build=False):
        """Reset the solver (no rebuild) and restart from a constant-speed guess at the current state."""
        self.acados_solver.reset()
        self.set_initial_state(x_cart)
        self._init_guess(self.x0_frenet)

    def reset(self, x_cart):
        self.reintialize_solver(x_cart)
