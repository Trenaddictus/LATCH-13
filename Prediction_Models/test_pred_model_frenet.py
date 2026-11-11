"""
Standalone checks for Prediction_Models/pred_model_frenet_stm_pacejka.py.

A. Equivalence with TUM's Cartesian prediction model: integrate both open loop from the same logged
   states with the same logged inputs, project the Cartesian result to Frenet and compare.
   kappa(s) is evaluated continuously -> differences come only from the reference-line geometry.
B. Stage-wise constant kappa (as it will be in the OCP: kappa per shooting interval of Ts_MPC, RK4 with
   3 steps) vs. continuous kappa(s): error introduced by the parameterisation.
C. Prediction error vs. the simulated plant (logged closed-loop run): Frenet model driven with the
   logged acceleration and steering-rate inputs, compared with the projected plant trajectory.
D. Singularity margin: min of (1 - kappa e_y) over the whole drivable track width.

Usage (from repo root):  python Prediction_Models/test_pred_model_frenet.py [full_logs.npz]
"""

import os
import sys

import numpy as np
from casadi import Function

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Prediction_Models.pred_model_frenet_stm_pacejka import pred_frenet_stm  # noqa: E402
from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm  # noqa: E402
from Utils.frenet_projection import FrenetReference, wrap_angle  # noqa: E402

CFG = "Config/EDGAR_Frenet/"
REF_FILE = "Trajectories/reftraj_silverstone_edgar.json"
DEFAULT_LOG = "Logs/SNMPC_trial2026-11-09_12-47-28/full_logs.npz"   # Silverstone, Tp=3.04, WMPC off
TS = 0.02          # simulator / log step [s]
TS_MPC = 0.08      # MPC shooting interval [s]
RK_STEPS = 3       # integrator steps per shooting interval (acados sim_method_num_steps in TUM's OCP)


def rk4(f, x, u, p, h):
    k1 = f(x, u, p); k2 = f(x + h / 2 * k1, u, p); k3 = f(x + h / 2 * k2, u, p); k4 = f(x + h * k3, u, p)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def main(log_file):
    ref = FrenetReference(REF_FILE)
    fm, _ = pred_frenet_stm(CFG + "veh_params_pred.yaml", CFG + "pacejka_params.yaml")
    cm, _ = pred_stm(CFG + "veh_params_pred.yaml", CFG + "pacejka_params.yaml")
    f_fren = lambda x, u, p: np.array(fm.f(x, u, p)).ravel()
    f_cart_cas = Function("fc", [cm.x, cm.u], [cm.f_expl_expr[:8]])
    f_cart = lambda x, u, p: np.array(f_cart_cas(x, u)).ravel()

    def p_at(s):
        r = ref.ref_at(s)
        return np.array([r["kappa"], r["v_ref"], r["ey_min"], r["ey_max"]])
    # continuous kappa: parameters re-evaluated inside each RK stage
    f_fren_cont = lambda x, u, p: f_fren(x, u, p_at(x[0]))

    d = np.load(log_file)
    X = d["CiLX"]          # plant states  [x, y, yaw, vx, vy, r, delta]           rows 0..n
    U = d["simU"]          # MPC output    [jerk, delta_dot]                         rows 0..n-1
    M = d["MPC_SimX"]      # MPC node-1 states; M[k+1, 7] = acceleration applied in step k
    n = len(U)
    a_appl = M[1:n + 1, 7]
    XF, _ = ref.project_trajectory(X)

    # sanity check of the log indexing: one plant step reproduced by TUM's Cartesian model?
    e1 = []
    for k in range(0, n - 1, 50):
        xc = np.append(X[k], a_appl[k])
        xn = rk4(f_cart, xc, np.array([0.0, U[k, 1]]), None, TS)
        e1.append(np.abs(xn[:7] - X[k + 1]).max())
    print("log indexing check (1 plant step, Cartesian model): max state error %.2e" % max(e1))

    # ---------------- D: singularity margin ----------------
    den = np.minimum(1 - ref.kappa * ref.ey_max, 1 - ref.kappa * ref.ey_min)
    print("\n[D] min over track of (1 - kappa*e_y) at the track edges: %.3f  (at s=%.0f m)"
          % (den.min(), ref.s[np.argmin(den)]))

    starts = np.arange(0, n - int(3.0 / TS) - 1, int(2.0 / TS))
    for H in (1.0, 3.0):
        nH = int(round(H / TS))
        errA, errB, errC = [], [], []
        for k0 in starts:
            # inputs over the window: jerk from the applied acceleration profile, logged steering rate
            jerk = np.diff(a_appl[k0:k0 + nH + 1]) / TS
            sr = U[k0:k0 + nH, 1]
            xf0 = np.append(XF[k0], a_appl[k0])                 # Frenet state incl. a
            xc0 = np.append(X[k0], a_appl[k0])                  # Cartesian state incl. a

            # A: Frenet (continuous kappa) vs Cartesian model, same inputs, 0.02 s RK4
            xf, xc = xf0.copy(), xc0.copy()
            for j in range(nH):
                u = np.array([jerk[j], sr[j]])
                xf = rk4(f_fren_cont, xf, u, None, TS)
                xc = rk4(f_cart, xc, u, None, TS)
            xc_f = ref.cartesian_to_frenet(xc[:7], s_guess=xf[0])
            errA.append([(xf[0] - xc_f[0] + ref.L / 2) % ref.L - ref.L / 2, xf[1] - xc_f[1],
                         wrap_angle(xf[2] - xc_f[2]), xf[3] - xc[3]])
            xA = xf.copy()

            # B: stage-wise constant parameters (OCP-like): Ts_MPC intervals, RK_STEPS RK4 steps each
            xb = xf0.copy()
            n_int = int(round(H / TS_MPC))
            spi = int(round(TS_MPC / TS))
            for i in range(n_int):
                p = p_at(xb[0])                                  # kappa fixed at the start of the interval
                u = np.array([np.mean(jerk[i * spi:(i + 1) * spi]), np.mean(sr[i * spi:(i + 1) * spi])])
                for _ in range(RK_STEPS):
                    xb = rk4(f_fren, xb, u, p, TS_MPC / RK_STEPS)
            # reference for B: continuous kappa with the same piecewise-constant inputs
            xr = xf0.copy()
            for i in range(n_int):
                u = np.array([np.mean(jerk[i * spi:(i + 1) * spi]), np.mean(sr[i * spi:(i + 1) * spi])])
                for _ in range(spi):
                    xr = rk4(f_fren_cont, xr, u, None, TS)
            errB.append([xb[0] - xr[0], xb[1] - xr[1], wrap_angle(xb[2] - xr[2]), xb[3] - xr[3]])

            # C: Frenet model (from A) vs logged plant, projected
            xp = XF[k0 + nH]
            errC.append([(xA[0] - xp[0] + ref.L / 2) % ref.L - ref.L / 2, xA[1] - xp[1],
                         wrap_angle(xA[2] - xp[2]), xA[3] - xp[3]])

        for name, e in (("A Frenet vs Cartesian model", errA), ("B const-kappa stages vs continuous", errB),
                        ("C Frenet model vs plant (log)", errC)):
            e = np.abs(np.array(e))
            print("[%s] H=%.0fs (%d windows): max |err| s %.3f m, e_y %.3f m, e_psi %.4f rad, v_x %.3f m/s"
                  % (name[0], H, len(e), *e.max(axis=0)) + "   " + name[2:])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG)
