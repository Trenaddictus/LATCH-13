
import casadi as ca

from acados_template import AcadosModel
from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm


def build_frenet_model(vehicle_yaml, tire_yaml):
    """
    LATCH-13 Frenet dynamic bicycle model.

    States (7):
        x = [s, ey, epsi, vx, vy, r, delta]

    Controls (2):
        u = [longitudinal_acceleration, steering_rate]

    Parameter (1):
        p = [track_curvature]

    Reuses TUM-CONTROL's vehicle dynamics and Pacejka
    tire model, replacing Cartesian position and
    heading kinematics with Frenet coordinates.
    """

    # --------------------------------------------------
    # 1. Load TUM's original dynamic bicycle model
    # --------------------------------------------------

    tum, _ = pred_stm(str(vehicle_yaml), str(tire_yaml))

    assert tum.x.numel() == 8
    assert tum.u.numel() == 2

    f_tum = ca.Function(
        "tum_vehicle_rhs",
        [tum.x, tum.u],
        [tum.xdot],
    )

    # --------------------------------------------------
    # 2. Define Frenet states and controls
    # --------------------------------------------------

    x = ca.MX.sym("x_frenet", 7)
    u = ca.MX.sym("u_frenet", 2)
    p = ca.MX.sym("curvature", 1)

    s, ey, epsi, vx, vy, r, delta = [
        x[i] for i in range(7)
    ]

    acceleration = u[0]
    steering_rate = u[1]
    kappa = p[0]

    # --------------------------------------------------
    # 3. Reuse TUM's physical vehicle dynamics
    # --------------------------------------------------

    # Original TUM prediction model states:
    # [X, Y, yaw, vx, vy, r, delta, acceleration]
    #
    # Acceleration is now a direct control input.

    x_tum = ca.vertcat(
        0, 0, 0,
        vx, vy, r, delta,
        acceleration,
    )

    # TUM prediction inputs:
    # [jerk, steering_rate]
    #
    # Jerk does not affect the first seven derivatives
    # when acceleration is supplied directly.
    u_tum = ca.vertcat(
        0,
        steering_rate,
    )

    f_original = f_tum(x_tum, u_tum)

    # --------------------------------------------------
    # 4. Frenet kinematics
    # --------------------------------------------------

    denominator = 1 - kappa * ey

    s_dot = (
        vx * ca.cos(epsi)
        - vy * ca.sin(epsi)
    ) / denominator

    ey_dot = (
        vx * ca.sin(epsi)
        + vy * ca.cos(epsi)
    )

    epsi_dot = r - kappa * s_dot

    # --------------------------------------------------
    # 5. Complete state dynamics
    # --------------------------------------------------

    f = ca.vertcat(
        s_dot,
        ey_dot,
        epsi_dot,
        f_original[3],  # vx_dot
        f_original[4],  # vy_dot
        f_original[5],  # yaw_rate_dot
        f_original[6],  # steering_angle_dot
    )

    # --------------------------------------------------
    # 6. Construct acados model
    # --------------------------------------------------

    model = AcadosModel()

    model.name = "latch13_frenet"

    model.x = x
    model.u = u
    model.p = p

    model.xdot = ca.MX.sym("xdot", 7)

    model.f_expl_expr = f
    model.f_impl_expr = model.xdot - f

    return model
