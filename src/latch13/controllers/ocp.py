"""Reusable Frenet NMPC formulation and acados solver construction.

State: [s, ey, epsi, vx, vy, r, delta, a]; input: [jerk, steering_rate].
Runtime stage parameters: [track_curvature, ey_min, ey_max]. There is no
reference speed: the stage cost rewards progress along the raceline, so speed
is limited only by the constraints. Long- and short-horizon experiments use
the same formulation, changing N and the terminal strategy.
Inspired by TUM's Nominal_NMPC/NMPC_class.py and its acados settings module:
model construction, costs, constraints, and solver creation are separate.
"""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosOcp, AcadosOcpSolver

from latch13.models.frenet_model import build_frenet_model


class TerminalCost(Protocol):
    """Build a scalar CasADi cost from the terminal state and stage parameters.

    Implementations may be functions, callable objects, or symbolic learned
    networks. Use only the supplied symbols and fixed numeric parameters;
    terminal controls and unrelated/free symbols are not supported. Expressions
    must support CasADi differentiation and acados C code generation. A PyTorch
    forward pass alone does not satisfy this interface. Captured learned weights
    are fixed at solver generation; changing them requires rebuilding the solver.
    """

    def __call__(self, x: ca.MX, p: ca.MX) -> ca.MX: ...


@dataclass(frozen=True)
class StageCostWeights:
    """Nonnegative weights; stage costs are integrated by acados using dt.

    ``progress`` multiplies the reward -s_dot (raceline progress rate, exact
    model expression); all other weights penalise squares.
    """

    progress: float = 1.0

    #Q matrix entries
    ey: float = 20.0
    epsi: float = 10.0
    vy: float = 0.1
    yaw_rate: float = 0.1
    acceleration: float = 0.05

    #R matrix entries
    jerk: float = 0.01
    steering_rate: float = 0.05


# Tuned with Optuna, one set per horizon (scripts/tune_stage_weights.py; ROADMAP "Stage-weight
# tuning"). FROZEN on 2026-11-24 for RL training: changing them invalidates every trained theta
# (ROADMAP Phase 4). The hand-picked weights are the StageCostWeights defaults above
# ("--stage-weights hand-picked").
TUNED_STAGE_WEIGHTS = {
    38: StageCostWeights(
        progress=1.0, ey=16.119283547937755, epsi=24.273681370192005, vy=0.1448771083975021,
        yaw_rate=0.23230960919035312, acceleration=0.0790032702661071, jerk=0.005197761196121792,
        steering_rate=0.3368912295147675,
    ),
    # N = 10: trial 84 of the first study (trial 2 of the robust study). Not the best score: the
    # best-scoring weights of both studies failed outside the laps they were tuned on, while these
    # finished every check (ROADMAP "Stage-weight tuning").
    10: StageCostWeights(
        progress=1.0, ey=150.06957542133145, epsi=89.58353978567126, vy=0.0020449173600114146,
        yaw_rate=0.5174571029260523, acceleration=0.023017294097330474, jerk=0.0048361170310622864,
        steering_rate=0.3774461756518242,
    ),
}


def default_stage_weights(N):
    """The tuned weights for N = 10 and N = 38, the hand-picked ones for other horizons."""
    return TUNED_STAGE_WEIGHTS.get(int(N), StageCostWeights())


def load_stage_weights(path):
    """Stage weights from a JSON file with a "stage_weights" object, such as the
    best_stage_weights.json written by scripts/tune_stage_weights.py.
    ``"hand-picked"`` gives the hand-picked weights of Phase 1."""
    if str(path) == "hand-picked":
        return StageCostWeights()
    values = json.loads(Path(path).read_text(encoding="utf-8"))["stage_weights"]
    expected = {field.name for field in fields(StageCostWeights)}
    if set(values) != expected:
        raise ValueError(f"{path}: expected stage weights {sorted(expected)}, got {sorted(values)}")
    weights = StageCostWeights(**{name: float(value) for name, value in values.items()})
    _validate_weights(weights)
    return weights


def weights_tag(weights):
    """Short fingerprint of a weights dataclass, e.g. for a compiled-code directory name."""
    text = json.dumps({field.name: getattr(weights, field.name) for field in fields(weights)}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class QuadraticTerminalCost:
    """Baseline terminal strategy, replaceable independently of stage costs.

    Equals the initial P of the learned terminal cost (docs/PROBLEM_SPEC.md
    section 3), which reads its initial weights from here.
    """

    #Terminal cost double the stage cost (ROADMAP step 3).
    ey: float = 40.0
    epsi: float = 20.0
    # Lateral states at the end of the horizon are otherwise almost free: with
    # a progress reward that direction is nearly flat/indefinite and the SQP
    # drifts there (STEP3_SOLVER_ISSUES.md, effect 1).
    vy: float = 10.0
    yaw_rate: float = 10.0
    steering_angle: float = 10.0
    # Small pull on the terminal speed and acceleration (initial P, 2026-11-23). The speed
    # weight sets the cruise speed: 0.01 gives about 20.5 m/s (docs/PROBLEM_SPEC.md section 3).
    vx: float = 0.01
    acceleration: float = 0.01

    def __call__(self, x, p):
        _validate_weights(self)
        return (self.ey * x[1]**2 + self.epsi * x[2]**2 + self.vx * x[3]**2
                + self.vy * x[4]**2 + self.yaw_rate * x[5]**2 + self.steering_angle * x[6]**2
                + self.acceleration * x[7]**2)


def zero_terminal_cost(x, p):
    """Explicit strategy for experiments without a terminal penalty."""
    return ca.MX(0)


@dataclass(frozen=True)
class ConstraintOptions:
    """Speed bounds, track edges and slack penalties beyond TUM's model bounds.

    All upstream model bounds remain hard constraints. The initial state is
    fixed by measurement, so state bounds and nonlinear constraints apply at
    nodes 1..N only (acados has no node-0 ``h`` here).

    Lateral acceleration and track edges are soft, with an exact L1 penalty
    (``linear`` must exceed the constraint's multiplier) plus a small L2 term.
    Lateral acceleration is soft because near its limit the SQP's linearised
    constraint can make the QP infeasible (QP failure at the first iteration)
    although the problem itself is feasible; with the soft version the same
    problem solves with zero slack. Set ``lateral_slack_linear=None`` for a
    hard limit. Track edges: the vehicle body, half its width plus
    ``track_margin``, must stay between ey_min and ey_max.
    """

    speed_min: float = 2.0
    speed_max: float = 25.0
    lateral_slack_linear: float | None = 1e3
    lateral_slack_quadratic: float = 1e1
    track_margin: float = 0.3
    track_slack_linear: float = 1e4
    track_slack_quadratic: float = 1e1


@dataclass(frozen=True)
class SolverOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver_type: str = "SQP"
    hessian_approx: str = "EXACT"
    # The progress reward -s_dot is not convex, so the exact Hessian can be
    # indefinite; without this the cold-start solve fails (status 3).
    regularize_method: str = "MIRROR"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    # Four RK4 substeps per shooting interval (0.02 s at dt=0.08).
    sim_method_num_steps: int = 4
    #nlp_solver_max_iter: int = 100
    nlp_solver_max_iter: int = 1000
    # Explicit convergence tolerances: tight feasibility; stationarity 1e-4
    # because with the progress reward the SQP reaches a residual floor of
    # ~1e-5 and otherwise stops at the iteration limit (STEP3_SOLVER_ISSUES.md).
    #nlp_solver_tol_stat: float = 1e-4
    nlp_solver_tol_stat: float = 1e-3
    nlp_solver_tol_eq: float = 1e-6
    nlp_solver_tol_ineq: float = 1e-6
    nlp_solver_tol_comp: float = 1e-6
    # Accurate QP subproblems: acados' default QP accuracy raised that
    # residual floor (failures at s ~ 600 m); 1e-8 and 1e-10 behave the same.
    qp_solver_tol_stat: float = 1e-8
    qp_solver_tol_eq: float = 1e-8
    qp_solver_tol_ineq: float = 1e-8
    qp_solver_tol_comp: float = 1e-8
    qp_solver_iter_max: int = 200
    # Funnel line search: with the soft track-edge constraints, merit
    # backtracking repeatedly stalled at its minimum step (up to ~100 SQP
    # iterations, sensitive to the slack weight); funnel keeps 3-5 iterations.
    globalization: str = "FUNNEL_L1PEN_LINESEARCH"
    levenberg_marquardt: float = 1e-2
    print_level: int = 0


def _validate_weights(weights):
    values = np.asarray(list(vars(weights).values()), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Cost weights must be finite and nonnegative")


def _stage_cost(model, weights):
    _validate_weights(weights)
    x, u = model.x, model.u
    progress_rate = model.f_expl_expr[0]  # s_dot = (vx cos epsi - vy sin epsi) / (1 - kappa ey)
    return (
        -weights.progress * progress_rate
        + weights.ey * x[1]**2
        + weights.epsi * x[2]**2
        + weights.vy * x[4]**2
        + weights.yaw_rate * x[5]**2
        + weights.acceleration * x[7]**2
        + weights.jerk * u[0]**2
        + weights.steering_rate * u[1]**2
    )


def _configure_costs(ocp, weights, terminal_cost):
    model = ocp.model
    terminal = ca.MX(terminal_cost(model.x, model.p))
    if terminal.shape != (1, 1):
        raise ValueError("Terminal cost must be a scalar CasADi expression")
    # Catch free symbols/terminal controls before generating C code.
    ca.Function("terminal_cost_validation", [model.x, model.p], [terminal])
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    model.cost_expr_ext_cost = _stage_cost(model, weights)
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

    # h = [a_lat, ey - upper edge, ey - lower edge]; edges come from p[1:3],
    # shrunk by half the vehicle width plus the safety margin.
    lateral = model.constraint
    lat_lower, lat_upper = _bound_pair(lateral.lat_acc_min, lateral.lat_acc_max, "lateral acceleration")
    if not np.isfinite(options.track_margin) or options.track_margin < 0:
        raise ValueError("Track margin must be finite and nonnegative")
    body = 0.5 * float(model.params.veh_width) + options.track_margin
    ey, ey_min, ey_max = model.x[1], model.p[1], model.p[2]
    model.con_h_expr = ca.vertcat(lateral.alat(model.x), ey - (ey_max - body), ey - (ey_min + body))
    model.con_h_expr_e = model.con_h_expr
    bounds.lh = np.array([lat_lower, -ACADOS_INFTY, 0.0])
    bounds.uh = np.array([lat_upper, 0.0, ACADOS_INFTY])
    bounds.lh_e = bounds.lh.copy()
    bounds.uh_e = bounds.uh.copy()

    soft = [(1, options.track_slack_linear, options.track_slack_quadratic),
            (2, options.track_slack_linear, options.track_slack_quadratic)]
    if options.lateral_slack_linear is not None:
        soft.insert(0, (0, options.lateral_slack_linear, options.lateral_slack_quadratic))
    for _, linear, quadratic in soft:
        if not np.all(np.isfinite([linear, quadratic])) or min(linear, quadratic) < 0:
            raise ValueError("Slack penalties must be finite and nonnegative")
        if linear == quadratic == 0:
            raise ValueError("Soft constraints require a positive penalty")
    index, linear, quadratic = (np.array(column) for column in zip(*soft))
    for suffix in ("", "_e"):
        setattr(bounds, "idxsh" + suffix, index.astype(int))
        for name, value in (("zl", linear), ("zu", linear), ("Zl", quadratic), ("Zu", quadratic)):
            setattr(ocp.cost, name + suffix, value.astype(float))

    bounds.x0 = x0


def build_ocp(
    config_dir, x0, N=38, dt=0.08, *,
    stage_weights=None, terminal_cost=None, constraints=None, solver_options=None,
):
    """Construct an inspectable OCP without generating or compiling code.

    ``stage_weights=None`` selects ``default_stage_weights(N)``.
    ``terminal_cost=None`` selects QuadraticTerminalCost; explicitly pass
    zero_terminal_cost to disable it, or any TerminalCost callable to replace it.
    Keep stage weights/dt identical when comparing different horizon lengths.
    """
    if isinstance(N, bool) or int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
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
    if (ocp.model.x.numel(), ocp.model.u.numel(), ocp.model.p.numel()) != (8, 2, 3):
        raise ValueError("Expected eight states, two controls, and [curvature, ey_min, ey_max]")
    _configure_costs(
        ocp, stage_weights if stage_weights is not None else default_stage_weights(N),
        terminal_cost if terminal_cost is not None else QuadraticTerminalCost(),
    )
    _configure_constraints(ocp, x0, constraints if constraints is not None else ConstraintOptions())
    # Placeholder until the runner sets per-stage values: straight, wide track.
    ocp.parameter_values = np.array([0.0, -1e3, 1e3])
    options = solver_options if solver_options is not None else SolverOptions()
    for name, value in vars(options).items():
        setattr(ocp.solver_options, name, value)
    ocp.solver_options.N_horizon = int(N)
    ocp.solver_options.tf = int(N) * float(dt)
    return ocp


def build_mpc(
    config_dir, x0, N=38, dt=0.08, output_dir="results/generated", *,
    stage_weights=None, terminal_cost=None, constraints=None, solver_options=None,
):
    """Build the OCP and compile its acados solver; return AcadosOcpSolver.

    Use separate output directories for concurrent experiments. Changing horizon
    or terminal expressions requires regeneration. The runner must supply eight
    states and interpret u[0] as jerk, not acceleration.
    """
    ocp = build_ocp(
        config_dir, x0, N, dt, stage_weights=stage_weights,
        terminal_cost=terminal_cost, constraints=constraints, solver_options=solver_options,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(output_dir / "c_generated_code")
    return AcadosOcpSolver(
        ocp, json_file=str(output_dir / "acados_ocp.json"), generate=True, build=True,
    )
