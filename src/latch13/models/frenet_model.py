"""Frenet-coordinate adapter for TUM's dynamic bicycle prediction model."""

from types import SimpleNamespace

import casadi as ca

from acados_template import AcadosModel
from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm


def build_frenet_model(vehicle_yaml, tire_yaml):
    """Reuse TUM's eight-state model, replacing only pose kinematics.

    States: [s, ey, epsi, vx, vy, r, delta, a].
    Inputs: [jerk, steering_rate], in the original TUM order.
    Parameter: [track_curvature].

    The returned AcadosModel retains TUM's ``params``, initial state ``x0``,
    and bound names: ``jerk_min/max``, ``acc_min/max``, ``delta_f_min/max``,
    and ``delta_f_dot_min/max``. Its ``constraint`` attribute exposes TUM's
    ``lat_acc_min/max`` and ``alat(x)`` for the Frenet state vector.
    These are metadata for the OCP builder; bounds are not applied here.

    ``x0`` uses TUM's physical initial values with zero Frenet progress and
    errors (the reference origin), rather than a Cartesian initial pose.
    Frenet coordinates require 1 - track_curvature * ey != 0.
    """
    tum, constraint = pred_stm(str(vehicle_yaml), str(tire_yaml))

    if tum.x.numel() != 8 or tum.u.numel() != 2:
        raise ValueError("Expected TUM's eight-state, two-input prediction model")
    if tum.xdot.numel() != 8 or tum.p.numel() != 0:
        raise ValueError("Expected eight TUM state derivatives and no parameters")

    x = ca.MX.sym("x_frenet", 8)
    curvature = ca.MX.sym("curvature", 1)
    s, ey, epsi, vx, vy, r, delta, a = [x[i] for i in range(8)]

    # Keep the upstream control symbols and their ordering unchanged.
    u = tum.u

    # TUM stores the eight physical derivatives in `xdot`. Its
    # `f_expl_expr` also appends the two controls, so it is not an
    # eight-state acados RHS and must not be copied directly.
    tum_rhs = ca.Function("tum_vehicle_rhs", [tum.x, tum.u], [tum.xdot])
    cartesian_state = ca.vertcat(0, 0, 0, vx, vy, r, delta, a)
    physical_rhs = tum_rhs(cartesian_state, u)

    s_dot = (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1 - curvature * ey)
    ey_dot = vx * ca.sin(epsi) + vy * ca.cos(epsi)
    epsi_dot = r - curvature * s_dot
    rhs = ca.vertcat(s_dot, ey_dot, epsi_dot, physical_rhs[3:8])

    model = AcadosModel()

    # Carry upstream numeric metadata forward without duplicating YAML
    # values. Rebuild symbolic fields below in the new coordinates.
    symbolic_fields = {"x", "xdot", "u", "z", "p", "name", "f_expl_expr"}
    for name, value in vars(tum).items():
        if name not in symbolic_fields:
            setattr(model, name, value)
    model.x0 = tum.x0.copy()
    model.x0[:3] = 0.0

    model.name = "latch13_frenet"
    model.x = x
    model.u = u
    model.z = tum.z
    model.p = curvature
    model.xdot = ca.MX.sym("xdot", 8)
    model.f_expl_expr = rhs
    model.f_impl_expr = model.xdot - rhs

    # Preserve TUM's constraint interface, with its state-dependent
    # function explicitly bound to the Frenet state vector.
    model.constraint = SimpleNamespace(**vars(constraint))
    model.constraint.alat = ca.Function(
        "a_lat_frenet", [x], [constraint.alat(cartesian_state)]
    )

    return model
