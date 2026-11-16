"""LATCH model, road preview and independent Cartesian plant adapter."""
import casadi as ca
import numpy as np

from .runtime import DATA, parameters
from .vendor.frenet_model import pred_frenet_stm
from .vendor.frenet_projection import FrenetReference


def reference():
    return FrenetReference(str(DATA / "reftraj_silverstone_edgar.json"))


def prediction_model():
    _, cfg = parameters()
    return pred_frenet_stm(str(cfg / "veh_params_pred.yaml"), str(cfg / "pacejka_params.yaml"))[0]


def rk4_map(model, dt, steps):
    """Same RK4, 3 substeps / prediction interval as the uploaded ERK configuration."""
    x, u, p = ca.SX.sym("x", 8), ca.SX.sym("u", 2), ca.SX.sym("road", 4)
    xn = x
    h = dt / steps
    for _ in range(steps):
        k1 = model.f(xn, u, p)
        k2 = model.f(xn + h * k1 / 2, u, p)
        k3 = model.f(xn + h * k2 / 2, u, p)
        k4 = model.f(xn + h * k3, u, p)
        xn = xn + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return ca.Function("latch_rk4", [x, u, p], [xn])


def preview(ref, states, N, dt):
    """Fixed road forecast per solve, shaped (B, N+1, 4).

    Spatial nodes are based on current longitudinal speed; the same prefix is used
    for teacher and student. Road lookup is outside the differentiable solver.
    """
    states = np.atleast_2d(states)
    s = states[:, 0, None] + np.maximum(states[:, 3, None], 1.0) * dt * np.arange(N + 1)
    mid = (s[:, :-1] + s[:, 1:]) / 2
    nodes = ref.ref_at(s)
    curvature = np.concatenate((ref.ref_at(mid)["kappa"], nodes["kappa"][:, -1:]), axis=1)
    return np.stack((curvature, nodes["v_ref"], nodes["ey_min"], nodes["ey_max"]), axis=-1)


def initial_state(ref, settings, s=None, ey=0.0, epsi=0.0, speed_offset=0.0):
    if s is None:
        s = ref.s[int(settings["initial_reference_index"])]
    r = ref.ref_at(s)
    return np.array([s, ey, epsi, max(float(r["v_ref"]) + speed_offset, 2.0), 0., 0., 0., 0.])


class CartesianPlant:
    """Original 7-state simulator dynamics; jerk supplies an eighth actuator state.

    consistent_jerk: hold jerk and steering rate over the actual simulator step,
    updating acceleration within RK4. This reuses the original plant RHS and
    makes the action physically consistent across 0.08/0.02-second discretizations.
    latch_legacy: original sim_step convention (node-1 acceleration held for Ts).
    """
    def __init__(self, settings):
        from .vendor.plant_model import sim_stm
        _, cfg = parameters()
        m, _ = sim_stm(str(cfg / "veh_params_sim.yaml"), str(cfg / "pacejka_params.yaml"))
        self.f = ca.Function("latch_cartesian_rhs", [m.x, m.u], [m.xdot])
        self.dt = settings["simulation_dt"]
        self.mode = settings["actuation"]

    def step(self, state, action, predicted_acceleration=None):
        state, action = np.asarray(state, float), np.asarray(action, float)
        if self.mode == "latch_legacy":
            if predicted_acceleration is None:
                raise ValueError("Legacy actuation needs prediction node-1 acceleration.")
            x = state[:7].copy()
            f = lambda v: np.asarray(self.f(v, [predicted_acceleration, action[1]])).ravel()
        elif self.mode == "consistent_jerk":
            x = state.copy()
            f = lambda v: np.r_[np.asarray(self.f(v[:7], [v[7], action[1]])).ravel(), action[0]]
        else:
            raise ValueError(f"Unknown actuation mode: {self.mode}")
        # Four elements as in the original CasADi plant integrator.
        h = self.dt / 4
        for _ in range(4):
            k1 = f(x); k2 = f(x + h*k1/2); k3 = f(x + h*k2/2); k4 = f(x + h*k3)
            x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        return np.r_[x, predicted_acceleration] if self.mode == "latch_legacy" else x


def project(ref, cartesian, previous_s):
    xf = ref.cartesian_to_frenet(cartesian, s_guess=previous_s)
    xf[0] += ref.L * np.round((previous_s - xf[0]) / ref.L)
    return xf
