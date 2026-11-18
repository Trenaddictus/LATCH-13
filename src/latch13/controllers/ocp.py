"""Reusable Frenet NMPC formulation and acados solver construction.

State: [s, ey, epsi, vx, vy, r, delta, a]; input: [jerk, steering_rate].
Runtime stage parameters remain [track_curvature]. Long- and short-horizon
experiments use the same formulation, changing N and the terminal strategy.
Inspired by TUM's Nominal_NMPC/NMPC_class.py and its acados settings module:
model construction, costs, constraints, and solver creation are separate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from latch13.models.frenet_model import build_frenet_model


class TerminalCost(Protocol):
    """Build a scalar CasADi cost from terminal state, curvature, and speed.

    Implementations may be functions, callable objects, or symbolic learned
    networks. Use only the supplied symbols and fixed numeric parameters;
    terminal controls and unrelated/free symbols are not supported. Expressions
    must support CasADi differentiation and acados C code generation. A PyTorch
    forward pass alone does not satisfy this interface. Captured learned weights
    are fixed at solver generation; changing them requires rebuilding the solver.
    """

    def __call__(self, x: ca.MX, p: ca.MX, v_ref: float) -> ca.MX: ...


@dataclass(frozen=True)
class StageCostWeights:
    """Nonnegative weights; stage costs are integrated by acados using dt."""

    #Q matrix entries
    ey: float = 20.0
    epsi: float = 10.0
    vx: float = 1.0
    vy: float = 0.1
    yaw_rate: float = 0.1
    acceleration: float = 0.05

    #R matrix entries
    jerk: float = 0.01
    steering_rate: float = 0.05


@dataclass(frozen=True)
class QuadraticTerminalCost:
    """Baseline terminal strategy, replaceable independently of stage costs."""
    
    #Terminal cost double the stage cost. zero for the vy, yaw_rate and acceleration
    ey: float = 40.0
    epsi: float = 20.0
    vx: float = 2.0

    def __call__(self, x, p, v_ref):
        _validate_weights(self)
        return self.ey * x[1]**2 + self.epsi * x[2]**2 + self.vx * (x[3] - v_ref)**2


def zero_terminal_cost(x, p, v_ref):
    """Explicit strategy for experiments without a terminal penalty."""
    return ca.MX(0)


@dataclass(frozen=True)
class ConstraintOptions:
    """Additional speed bounds, not supplied by TUM's prediction model.

    All upstream model bounds remain hard constraints. Optionally soften only
    lateral acceleration, using TUM's linear/quadratic slack penalty convention.
    Initial state is fixed by measurement; state box bounds apply at nodes 1..N.
    Lateral acceleration applies at nodes 0..N, including the measured state.
    """

    speed_min: float = 1.0
    speed_max: float = 25.0
    lateral_slack_linear: float | None = None
    lateral_slack_quadratic: float = 0.0


@dataclass(frozen=True)
class SolverOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver_type: str = "SQP"
    hessian_approx: str = "EXACT"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    # Four RK4 substeps per shooting interval (0.02 s at dt=0.08).
    sim_method_num_steps: int = 4
    nlp_solver_max_iter: int = 100
    # Explicit convergence tolerances: retain tight feasibility while avoiding
    # iteration-limit exits for negligible stationarity residuals.
    nlp_solver_tol_stat: float = 1e-5
    nlp_solver_tol_eq: float = 1e-6
    nlp_solver_tol_ineq: float = 1e-6
    nlp_solver_tol_comp: float = 1e-6
    globalization: str = "MERIT_BACKTRACKING"
    levenberg_marquardt: float = 1e-2
    print_level: int = 0


def _validate_weights(weights):
    values = np.asarray(list(vars(weights).values()), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Cost weights must be finite and nonnegative")


def _stage_cost(model, v_ref, weights):
    _validate_weights(weights)
    x, u = model.x, model.u
    return (
        weights.ey * x[1]**2
        + weights.epsi * x[2]**2
        + weights.vx * (x[3] - v_ref)**2
        + weights.vy * x[4]**2
        + weights.yaw_rate * x[5]**2
        + weights.acceleration * x[7]**2
        + weights.jerk * u[0]**2
        + weights.steering_rate * u[1]**2
    )


def _configure_costs(ocp, v_ref, weights, terminal_cost):
    model = ocp.model
    terminal = ca.MX(terminal_cost(model.x, model.p, v_ref))
    if terminal.shape != (1, 1):
        raise ValueError("Terminal cost must be a scalar CasADi expression")
    # Catch free symbols/terminal controls before generating C code.
    ca.Function("terminal_cost_validation", [model.x, model.p], [terminal])
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    model.cost_expr_ext_cost = _stage_cost(model, v_ref, weights)
    model.cost_expr_ext_cost_e = terminal


def _bound_pair(lower, upper, name):
    pair = np.asarray([lower, upper], dtype=float)
    if not np.all(np.isfinite(pair)) or lower > upper:
        raise ValueError(f"Invalid {name} bounds: {pair}")
    return pair


def _configure_constraints(ocp, x0, options):
    model, bounds = ocp.model, ocp.constraints
    pairs = [
        _bound_pair(options.speed_min, options.speed_max, "speed"),
        _bound_pair(model.delta_f_min, model.delta_f_max, "steering angle"),
        _bound_pair(model.acc_min, model.acc_max, "acceleration"),
    ]
    # Same physical state indices as TUM's eight-state prediction model.
    for suffix in ("", "_e"):
        setattr(bounds, "idxbx" + suffix, np.array([3, 6, 7], dtype=int))
        setattr(bounds, "lbx" + suffix, np.array([pair[0] for pair in pairs]))
        setattr(bounds, "ubx" + suffix, np.array([pair[1] for pair in pairs]))

    inputs = [
        _bound_pair(model.jerk_min, model.jerk_max, "jerk"),
        _bound_pair(model.delta_f_dot_min, model.delta_f_dot_max, "steering rate"),
    ]
    bounds.idxbu = np.array([0, 1], dtype=int)
    bounds.lbu = np.array([pair[0] for pair in inputs])
    bounds.ubu = np.array([pair[1] for pair in inputs])

    lateral = model.constraint
    lower, upper = _bound_pair(lateral.lat_acc_min, lateral.lat_acc_max, "lateral acceleration")
    model.con_h_expr = lateral.alat(model.x)
    model.con_h_expr_e = model.con_h_expr
    bounds.lh = np.array([lower])
    bounds.uh = np.array([upper])
    bounds.lh_e = bounds.lh.copy()
    bounds.uh_e = bounds.uh.copy()

    if options.lateral_slack_linear is not None:
        linear, quadratic = options.lateral_slack_linear, options.lateral_slack_quadratic
        if not np.all(np.isfinite([linear, quadratic])) or min(linear, quadratic) < 0:
            raise ValueError("Slack penalties must be finite and nonnegative")
        if linear == quadratic == 0:
            raise ValueError("Soft lateral constraints require a positive penalty")
        for suffix in ("", "_e"):
            setattr(bounds, "idxsh" + suffix, np.array([0], dtype=int))
            for name, value in (("zl", linear), ("zu", linear), ("Zl", quadratic), ("Zu", quadratic)):
                setattr(ocp.cost, name + suffix, np.array([value]))

    bounds.x0 = x0


def build_ocp(
    config_dir, x0, N=38, dt=0.08, v_ref=8.0, *,
    stage_weights=None, terminal_cost=None, constraints=None, solver_options=None,
):
    """Construct an inspectable OCP without generating or compiling code.

    ``terminal_cost=None`` selects QuadraticTerminalCost; explicitly pass
    zero_terminal_cost to disable it, or any TerminalCost callable to replace it.
    Keep stage weights/dt identical when comparing different horizon lengths.
    """
    if isinstance(N, bool) or int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if not np.all(np.isfinite([dt, v_ref])) or dt <= 0 or v_ref <= 0:
        raise ValueError("dt and v_ref must be finite and positive")
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape != (8,) or not np.all(np.isfinite(x0)):
        raise ValueError("Expected a finite eight-state initial Frenet state")

    config_dir = Path(config_dir).resolve()
    vehicle_file = config_dir / "veh_params_pred.yaml"
    tire_file = config_dir / "pacejka_params.yaml"
    for path in (vehicle_file, tire_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    ocp = AcadosOcp()
    ocp.model = build_frenet_model(vehicle_file, tire_file)
    if (ocp.model.x.numel(), ocp.model.u.numel(), ocp.model.p.numel()) != (8, 2, 1):
        raise ValueError("Expected eight states, two controls, and one curvature parameter")
    _configure_costs(
        ocp, float(v_ref), stage_weights if stage_weights is not None else StageCostWeights(),
        terminal_cost if terminal_cost is not None else QuadraticTerminalCost(),
    )
    _configure_constraints(ocp, x0, constraints if constraints is not None else ConstraintOptions())
    ocp.parameter_values = np.zeros(1)
    options = solver_options if solver_options is not None else SolverOptions()
    for name, value in vars(options).items():
        setattr(ocp.solver_options, name, value)
    ocp.solver_options.N_horizon = int(N)
    ocp.solver_options.tf = int(N) * float(dt)
    return ocp


def build_mpc(
    config_dir, x0, N=38, dt=0.08, v_ref=8.0, output_dir="results/generated", *,
    stage_weights=None, terminal_cost=None, constraints=None, solver_options=None,
):
    """Build the OCP and compile its acados solver; return AcadosOcpSolver.

    Use separate output directories for concurrent experiments. Changing horizon
    or terminal expressions requires regeneration. The runner must supply eight
    states and interpret u[0] as jerk, not acceleration.
    """
    ocp = build_ocp(
        config_dir, x0, N, dt, v_ref, stage_weights=stage_weights,
        terminal_cost=terminal_cost, constraints=constraints, solver_options=solver_options,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(output_dir / "c_generated_code")
    return AcadosOcpSolver(
        ocp, json_file=str(output_dir / "acados_ocp.json"), generate=True, build=True,
    )
