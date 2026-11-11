# Frenet-frame dynamic single-track model with Pacejka tyres, for the Frenet NMPC (SSY226 project).
#
# Vehicle dynamics (forces, v_x/v_y/yaw-rate equations) are identical to TUM-CONTROL's Cartesian
# prediction model (Prediction_Models/pred_model_dynamic_stm_pacejka.py, B. Zarrouki). Only the
# pose kinematics (posx, posy, yaw) are replaced by Frenet kinematics w.r.t. a reference line.
#
# State  x = [s, e_y, e_psi, v_x, v_y, psi_dot, delta, a]
#   s      arc length along the reference line [m]
#   e_y    lateral offset, positive = left of the reference line [m]
#   e_psi  heading error psi - psi_ref(s) [rad]
#   v_x, v_y, psi_dot, delta  as in TUM-CONTROL (body frame, CoG)
#   a      longitudinal acceleration command [m/s^2] (the plant input; same role as TUM's state 7)
# Input  u = [jerk, delta_dot]                         (same as TUM-CONTROL's MPC)
# Stage parameters p = [kappa, v_ref, ey_min, ey_max]  (indices in P_IDX)
#   kappa is used by the dynamics; v_ref and the e_y bounds are reserved for the OCP cost/constraints
#   (acados uses one parameter vector per stage for model, cost and constraints).
#
# Kinematics (planning report):
#   s_dot     = (v_x cos e_psi - v_y sin e_psi) / (1 - kappa e_y)
#   e_y_dot   =  v_x sin e_psi + v_y cos e_psi
#   e_psi_dot =  psi_dot - kappa s_dot
# Longitudinal dynamics differ from the report's simplified v_x_dot = a_x + psi_dot v_y - F_yf sin(delta)/m:
# TUM's model additionally contains rolling resistance and aerodynamic drag, i.e. the report's a_x
# corresponds to a - (F_aero + F_r,r + F_r,f cos(delta)) / m here. Kept, because the simulator has them too.

import types

import numpy as np
import yaml
from casadi import MX, vertcat, sin, cos, arctan, arcsin, sqrt, fmax, fmin, if_else, Function

P_IDX = {"kappa": 0, "v_ref": 1, "ey_min": 2, "ey_max": 3}
X_IDX = {"s": 0, "e_y": 1, "e_psi": 2, "v_x": 3, "v_y": 4, "psi_dot": 5, "delta": 6, "a": 7}


def pred_frenet_stm(veh_params_file, tire_params_file):
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()
    params = types.SimpleNamespace()

    # ---------------- parameters (same sources and constants as TUM's pred_stm) ----------------
    with open(veh_params_file, "r") as f:
        veh = yaml.safe_load(f)
    with open(tire_params_file, "r") as f:
        tire = yaml.safe_load(f)

    lf, lr, m, Iz = veh["lf"], veh["lr"], veh["m"], veh["Iz"]
    g = 9.81
    banking = 0.0
    ro, S, Cd = veh["ro"], veh["S"], veh["Cd"]
    fr0, fr1, fr4 = 0.009, 0.002, 0.0003                  # rolling resistance factors (TUM, Gerdts 2003)
    Bf, Cf, Df, Ef = (tire["tire_params"]["front"][k] for k in ("Bf", "Cf", "Df", "Ef"))
    Br, Cr, Dr, Er = (tire["tire_params"]["rear"][k] for k in ("Br", "Cr", "Dr", "Er"))
    mu = tire["mu"]

    for k, v in dict(lf=lf, lr=lr, m=m, Iz=Iz, Bf=Bf, Cf=Cf, Df=Df, Ef=Ef, Br=Br, Cr=Cr, Dr=Dr, Er=Er,
                     veh_length=veh["veh_length"], veh_width=veh["veh_width"]).items():
        setattr(params, k, v)

    # ---------------- symbols ----------------
    s, e_y, e_psi = MX.sym("s"), MX.sym("e_y"), MX.sym("e_psi")
    vlong, vlat, yawrate = MX.sym("v_x"), MX.sym("v_y"), MX.sym("psi_dot")
    delta_f, a = MX.sym("delta"), MX.sym("a")
    x = vertcat(s, e_y, e_psi, vlong, vlat, yawrate, delta_f, a)

    jerk, steering_rate = MX.sym("jerk"), MX.sym("delta_dot")
    u = vertcat(jerk, steering_rate)

    kappa, v_ref, ey_min, ey_max = MX.sym("kappa"), MX.sym("v_ref"), MX.sym("ey_min"), MX.sym("ey_max")
    p = vertcat(kappa, v_ref, ey_min, ey_max)

    xdot = MX.sym("xdot", x.shape[0])

    # ---------------- forces: copied from TUM pred_stm (lines 115-173 there) ----------------
    v_kmh = sqrt(vlong**2 + vlat**2) * 3.6
    fr = fr0 + fr1 * v_kmh / 100 + fr4 * (v_kmh / 100) ** 4
    Fz_f = m * lr * g / (lf + lr)
    Fz_r = m * lf * g / (lf + lr)
    Fr_f = fr * Fz_f
    Fr_r = fr * Fz_r

    Fbanking_x = m * g * np.sin(banking) * np.sin(mu)
    Fbanking_y = m * g * np.sin(banking) * np.cos(mu)
    Faero = 0.5 * ro * S * Cd * vlong**2

    Fd = m * a                                             # driving force (rear-wheel drive)
    Fx_f = -Fr_f
    Fx_r = Fd - Fr_r

    alpha_f = if_else(vlong > 0.001, delta_f - arctan((vlat + lf * yawrate) / vlong), 0.0)
    alpha_r = if_else(vlong > 0.001, arctan((lr * yawrate - vlat) / vlong), 0.0)

    Fy_f_lat = Df * sin(Cf * arctan(Bf * alpha_f - Ef * (Bf * alpha_f - arctan(Bf * alpha_f))))
    Fy_r_lat = Dr * sin(Cr * arctan(Br * alpha_r - Er * (Br * alpha_r - arctan(Br * alpha_r))))

    Fmax_f = sqrt(Fz_f**2 + (Cf * Fz_f) ** 2)
    Fmax_r = sqrt(Fz_r**2 + (Cr * Fz_r) ** 2)
    Gy_f = fmax(fmin(Fx_f / Fmax_f, 0.98), -0.98)
    Gy_r = fmax(fmin(Fx_r / Fmax_r, 0.98), -0.98)
    Fy_f = Fy_f_lat * cos(arcsin(Gy_f))
    Fy_r = Fy_r_lat * cos(arcsin(Gy_r))

    vlong_dot = 1 / m * (Fx_r - Faero - Fy_f * sin(delta_f) + Fx_f * cos(delta_f) - Fbanking_x + m * vlat * yawrate)
    vlat_dot = 1 / m * (Fy_r + Fy_f * cos(delta_f) + Fx_f * sin(delta_f) - Fbanking_y - m * vlong * yawrate)
    yawrate_dot = 1 / Iz * (lf * (Fy_f * cos(delta_f) + Fx_f * sin(delta_f)) - lr * Fy_r)

    # ---------------- Frenet kinematics ----------------
    s_dot = (vlong * cos(e_psi) - vlat * sin(e_psi)) / (1 - kappa * e_y)
    e_y_dot = vlong * sin(e_psi) + vlat * cos(e_psi)
    e_psi_dot = yawrate - kappa * s_dot

    f_expl = vertcat(s_dot, e_y_dot, e_psi_dot, vlong_dot, vlat_dot, yawrate_dot, steering_rate, jerk)

    # ---------------- constraint data (same yaml keys as TUM) ----------------
    model.jerk_min, model.jerk_max = veh["jerk_min"], veh["jerk_max"]
    model.acc_min, model.acc_max = veh["acc_min"], veh["acc_max"]
    model.delta_f_min, model.delta_f_max = veh["delta_f_min"], veh["delta_f_max"]
    model.delta_f_dot_min, model.delta_f_dot_max = veh["delta_f_dot_min"], veh["delta_f_dot_max"]
    constraint.lat_acc_min, constraint.lat_acc_max = veh["lat_acc_min"], veh["lat_acc_max"]
    constraint.alat = Function("a_lat", [x], [vlong * yawrate])

    # ---------------- model struct (same fields as TUM's pred_stm, plus f_impl_expr) ----------------
    model.x, model.u, model.p, model.z = x, u, p, vertcat()
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.name = "frenet_dynamic_bicycle_model"
    model.params = params
    model.x0 = np.zeros(8)
    model.f = Function("f_frenet", [x, u, p], [f_expl], ["x", "u", "p"], ["xdot"])
    model.X_IDX, model.P_IDX = X_IDX, P_IDX
    return model, constraint
