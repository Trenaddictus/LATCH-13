
"""
LATCH-13: Frenet-coordinate nonlinear MPC.

Prediction dynamics:
    TUM-CONTROL dynamic bicycle model + Pacejka tires,
    transformed into Frenet coordinates by LATCH-13.

States (7):
    x = [s, ey, epsi, vx, vy, r, delta]

Controls (2):
    u = [longitudinal_acceleration, steering_rate]

Stage parameter:
    p = [track_curvature]

Terminal cost:
    Conventional quadratic approximation.

This module creates the acados solver.
It does not run the vehicle simulation.
"""

from pathlib import Path

import casadi as ca
import numpy as np

from acados_template import AcadosOcp, AcadosOcpSolver

from latch13.models.frenet_model import build_frenet_model


def build_mpc(
    config_dir,
    x0,
    N=38,
    dt=0.08,
    v_ref=8.0,
    output_dir="results/generated",
):
    """
    Construct the LATCH-13 Frenet NMPC solver.

    Args:
        config_dir:
            Directory containing TUM vehicle and tire YAML files.

        x0:
            Initial Frenet state, shape (7,).

        N:
            Number of shooting intervals.

        dt:
            MPC discretization interval in seconds.

        v_ref:
            Constant reference longitudinal velocity in m/s.

        output_dir:
            Directory for generated acados solver files.

    Returns:
        AcadosOcpSolver
    """

    # --------------------------------------------------
    # 1. Configuration and input validation
    # --------------------------------------------------

    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle_file = config_dir / "veh_params_pred.yaml"
    tire_file = config_dir / "pacejka_params.yaml"

    if not vehicle_file.is_file():
        raise FileNotFoundError(vehicle_file)

    if not tire_file.is_file():
        raise FileNotFoundError(tire_file)

    N = int(N)
    dt = float(dt)
    v_ref = float(v_ref)

    if N < 1 or dt <= 0:
        raise ValueError("N and dt must be positive")

    x0 = np.asarray(x0, dtype=float).reshape(-1)

    if x0.shape != (7,):
        raise ValueError(
            f"Expected x0 with 7 states, got {x0.shape}"
        )

    if not np.all(np.isfinite(x0)):
        raise ValueError("Initial state contains NaN or Inf")

    # --------------------------------------------------
    # 2. Build Frenet prediction model
    # --------------------------------------------------

    model = build_frenet_model(
        vehicle_file,
        tire_file,
    )

    if model.x.numel() != 7:
        raise ValueError("Frenet model must have 7 states")

    if model.u.numel() != 2:
        raise ValueError("Frenet model must have 2 inputs")

    if model.p.numel() != 1:
        raise ValueError(
            "Expected one curvature parameter"
        )

    x = model.x
    u = model.u

    # x = [s, ey, epsi, vx, vy, r, delta]
    # u = [acceleration, steering_rate]

    s = x[0]
    ey = x[1]
    epsi = x[2]
    vx = x[3]
    vy = x[4]
    r = x[5]
    delta = x[6]

    acceleration = u[0]
    steering_rate = u[1]

    # --------------------------------------------------
    # 3. Define the stage cost
    # --------------------------------------------------

    # Initial tuning weights.
    # These are not yet the validated research weights.

    Q_ey = 20.0
    Q_epsi = 10.0
    Q_vx = 1.0
    Q_vy = 0.1
    Q_r = 0.1

    R_acc = 0.05
    R_steer_rate = 0.05

    stage_cost = (
        Q_ey * ey**2
        + Q_epsi * epsi**2
        + Q_vx * (vx - v_ref)**2
        + Q_vy * vy**2
        + Q_r * r**2
        + R_acc * acceleration**2
        + R_steer_rate * steering_rate**2
    )

    # --------------------------------------------------
    # 4. Define conventional terminal cost
    # --------------------------------------------------

    # This is what LEAP-C will eventually replace.

    Qf_ey = 40.0
    Qf_epsi = 20.0
    Qf_vx = 2.0

    terminal_cost = (
        Qf_ey * ey**2
        + Qf_epsi * epsi**2
        + Qf_vx * (vx - v_ref)**2
    )

    # --------------------------------------------------
    # 5. Create optimal control problem
    # --------------------------------------------------

    ocp = AcadosOcp()

    ocp.model = model

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    # External CasADi cost functions
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # acados automatically scales stage costs by dt.
    # Do not multiply stage_cost by dt again.
    ocp.model.cost_expr_ext_cost = stage_cost

    # Terminal cost is not scaled by dt.
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # --------------------------------------------------
    # 6. State constraints
    # --------------------------------------------------

    # Constrain:
    # x[3] = longitudinal speed
    # x[6] = steering angle

    ocp.constraints.idxbx = np.array(
        [3, 6], dtype=int
    )

    ocp.constraints.lbx = np.array([
        1.0,       # Minimum speed [m/s]
        -0.45,     # Minimum steering angle [rad]
    ])

    ocp.constraints.ubx = np.array([
        25.0,      # Maximum speed [m/s]
        0.45,      # Maximum steering angle [rad]
    ])

    # --------------------------------------------------
    # 7. Control constraints
    # --------------------------------------------------

    # u[0] = longitudinal acceleration
    # u[1] = steering rate

    ocp.constraints.idxbu = np.array(
        [0, 1], dtype=int
    )

    ocp.constraints.lbu = np.array([
        -4.0,      # Maximum braking [m/s^2]
        -0.35,     # Steering rate lower bound [rad/s]
    ])

    ocp.constraints.ubu = np.array([
        3.0,       # Maximum acceleration [m/s^2]
        0.35,      # Steering rate upper bound [rad/s]
    ])

    # --------------------------------------------------
    # 8. Initial state constraint
    # --------------------------------------------------

    ocp.constraints.x0 = x0

    # --------------------------------------------------
    # 9. Track curvature parameter
    # --------------------------------------------------

    # Initial value only.
    #
    # The simulation runner updates this parameter
    # at each shooting node before every MPC solve.

    ocp.parameter_values = np.array([0.0])

    # --------------------------------------------------
    # 10. acados solver configuration
    # --------------------------------------------------

    ocp.solver_options.qp_solver = (
        "PARTIAL_CONDENSING_HPIPM"
    )

    ocp.solver_options.nlp_solver_type = "SQP"

    # Full nonlinear Hessian
    ocp.solver_options.hessian_approx = "EXACT"

    # Explicit Runge-Kutta prediction integration
    ocp.solver_options.integrator_type = "ERK"

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    # Increased from 40 for debugging difficult solves.
    # Can be reduced after controller tuning.
    ocp.solver_options.nlp_solver_max_iter = 100

    # Mild Hessian regularization
    ocp.solver_options.levenberg_marquardt = 1e-5

    ocp.solver_options.print_level = 0

    # --------------------------------------------------
    # 11. Generate and compile solver
    # --------------------------------------------------

    ocp.code_export_directory = str(
        output_dir / "c_generated_code"
    )

    solver = AcadosOcpSolver(
        ocp,
        json_file=str(output_dir / "acados_ocp.json"),
        generate=True,
        build=True,
    )

    return solver
