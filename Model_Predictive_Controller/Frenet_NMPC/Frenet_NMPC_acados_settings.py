# OCP formulation of the Frenet NMPC (SSY226 project), built on the acados pattern of TUM-CONTROL's
# Nominal_NMPC/NMPC_STM_acados_settings.py.
#
# x = [s, e_y, e_psi, v_x, v_y, psi_dot, delta, a], u = [jerk, delta_dot],
# stage parameters p = [kappa, v_ref, ey_min, ey_max]  (see Prediction_Models/pred_model_frenet_stm_pacejka.py)
#
# Stage cost (NONLINEAR_LS):  0.5 * || [e_y, e_psi, v_x - v_ref, jerk, delta_dot] ||^2_W,  W = 0.01 diag(Q, R)
# Terminal cost (EXTERNAL):   V_f(x_N, p_N); default quadratic 0.5 * 0.01 * xi^T diag(P) xi,
#                             xi = [e_y, e_psi, v_x - v_ref]. Pass `terminal_cost_fun` to swap in another V_f.
# Constraints:
#   hard:  jerk, delta_dot (input bounds)
#   soft:  delta, a (state bounds); combined acceleration (circle, ggv limits);
#          track bounds  e_y - ey_min >= w,  ey_max - e_y >= w,  w = half vehicle width + margin

import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import vertcat, if_else

from Prediction_Models.pred_model_frenet_stm_pacejka import pred_frenet_stm


def quadratic_terminal_cost(x, p, P):
    """Benchmark terminal cost 0.5 * 0.01 * xi^T diag(P) xi with xi = [e_y, e_psi, v_x - v_ref]."""
    xi = vertcat(x[1], x[2], x[3] - p[1])
    return 0.5 * 0.01 * (P[0] * xi[0] ** 2 + P[1] * xi[1] ** 2 + P[2] * xi[2] ** 2)


def acados_settings(Tf, N, x0, Q, R, P, ax_max_interpolant, ay_max_interpolant, veh_params_file, tire_params_file,
                    track_margin=0.0, L1_pen=106.7, L2_pen=9.9, L1_pen_track=1000.0, L2_pen_track=1000.0,
                    qp_solver="PARTIAL_CONDENSING_HPIPM", qp_solver_iter_max=50, nlp_solver_type="SQP_RTI",
                    sim_method_num_steps=3, terminal_cost_fun=None,
                    solver_generate_C_code=True, solver_build=True):

    pred_model, constraint = pred_frenet_stm(veh_params_file, tire_params_file)

    ocp = AcadosOcp()
    model = AcadosModel()
    model.x, model.u, model.p = pred_model.x, pred_model.u, pred_model.p
    model.xdot = pred_model.xdot
    model.f_expl_expr = pred_model.f_expl_expr
    model.f_impl_expr = pred_model.f_impl_expr
    model.name = "FNMPC"
    ocp.model = model
    ocp.solver_options.N_horizon = N

    x, u, p = model.x, model.u, model.p
    e_y, e_psi, v_x, psi_dot, a = x[1], x[2], x[3], x[5], x[7]
    kappa, v_ref, ey_min, ey_max = p[0], p[1], p[2], p[3]
    ocp.parameter_values = np.array([0.0, 0.0, -10.0, 10.0])

    # ---------------- cost ----------------
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = vertcat(e_y, e_psi, v_x - v_ref, u)
    ocp.cost.W = 0.01 * scipy.linalg.block_diag(Q, R)
    ocp.cost.yref = np.zeros(5)

    ocp.cost.cost_type_e = "EXTERNAL"
    if terminal_cost_fun is None:
        ocp.model.cost_expr_ext_cost_e = quadratic_terminal_cost(x, p, P)
    else:
        ocp.model.cost_expr_ext_cost_e = terminal_cost_fun(x, p)

    # ---------------- nonlinear constraints h(x, p) ----------------
    # combined acceleration (same as TUM, circle): (a_lon / ax_max)^2 + (a_lat / ay_max)^2 <= 1
    a_lat = v_x * psi_dot
    ay_max = ay_max_interpolant(v_x)
    ax_max = if_else(a < 0, -pred_model.acc_min, ax_max_interpolant(v_x))
    h_acc = (a / ax_max) ** 2 + (a_lat / ay_max) ** 2
    # track bounds relative to the reference line (vehicle edge inside the track)
    w = pred_model.params.veh_width / 2 + track_margin
    h_lo = e_y - ey_min
    h_hi = ey_max - e_y
    BIG = 1e4

    ocp.model.con_h_expr = vertcat(h_acc, h_lo, h_hi)
    ocp.constraints.lh = np.array([0.0, w, w])
    ocp.constraints.uh = np.array([1.0, BIG, BIG])
    ocp.model.con_h_expr_0 = ocp.model.con_h_expr
    ocp.constraints.lh_0, ocp.constraints.uh_0 = ocp.constraints.lh, ocp.constraints.uh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr
    ocp.constraints.lh_e, ocp.constraints.uh_e = ocp.constraints.lh, ocp.constraints.uh
    nh = 3

    # ---------------- bounds ----------------
    # inputs: hard
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([pred_model.jerk_min, pred_model.delta_f_dot_min])
    ocp.constraints.ubu = np.array([pred_model.jerk_max, pred_model.delta_f_dot_max])
    # states delta, a: soft (stages 1..N)
    ocp.constraints.idxbx = np.array([6, 7])
    ocp.constraints.lbx = np.array([pred_model.delta_f_min, pred_model.acc_min])
    ocp.constraints.ubx = np.array([pred_model.delta_f_max, pred_model.acc_max])
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    ocp.constraints.lbx_e, ocp.constraints.ubx_e = ocp.constraints.lbx, ocp.constraints.ubx
    ocp.constraints.x0 = x0

    # ---------------- slacks ----------------
    # slack order per stage: [bx slacks, h slacks]
    ocp.constraints.Jsbx = np.eye(2)
    ocp.constraints.Jsh = np.eye(nh)
    ocp.constraints.Jsbx_e = np.eye(2)
    ocp.constraints.Jsh_e = np.eye(nh)
    ocp.constraints.Jsh_0 = np.eye(nh)

    z1_h = np.array([L1_pen, L1_pen_track, L1_pen_track])
    z2_h = np.array([L2_pen, L2_pen_track, L2_pen_track])
    z1 = np.concatenate((np.full(2, L1_pen), z1_h))
    z2 = np.concatenate((np.full(2, L2_pen), z2_h))
    ocp.cost.zl, ocp.cost.zu, ocp.cost.Zl, ocp.cost.Zu = z1, z1, z2, z2
    ocp.cost.zl_e, ocp.cost.zu_e, ocp.cost.Zl_e, ocp.cost.Zu_e = z1, z1, z2, z2
    ocp.cost.zl_0, ocp.cost.zu_0, ocp.cost.Zl_0, ocp.cost.Zu_0 = z1_h, z1_h, z2_h, z2_h

    # ---------------- solver ----------------
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.qp_solver_iter_max = qp_solver_iter_max
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = sim_method_num_steps

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_FNMPC.json",
                                    generate=solver_generate_C_code, build=solver_build)
    return constraint, pred_model, acados_solver, ocp
