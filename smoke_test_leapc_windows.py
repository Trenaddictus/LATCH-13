"""Windows smoke test: batched MPC, terminal-cost gradients and finite differences.

Run from the Leap-C repository root after installing the two matching patches:
    python smoke_test_leapc_windows.py
No training, plotting, GPU or optional RL project is required.
"""

from pathlib import Path
import inspect
import os
import sys


def configure_paths():
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from your Leap-C repository root.")
    acados = root / "external" / "acados"
    if not (acados / "CMakeLists.txt").is_file():
        raise RuntimeError(f"The bundled acados source is missing: {acados}")
    if sys.platform == "win32":
        required = (
            acados / "bin" / "acados.dll",
            acados / "include" / "acados_c" / "ocp_nlp_interface.h",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Build/install the acados core with rebuild_acados_windows_v2.cmd first. "
                "Missing: " + "; ".join(missing)
            )
    os.environ["ACADOS_SOURCE_DIR"] = str(acados)
    os.environ["PATH"] = os.pathsep.join(
        [str(acados / "bin"), str(acados / "lib"), os.environ.get("PATH", "")]
    )
    tmp = root / "leapc_smoke_tmp"
    tmp.mkdir(exist_ok=True)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    dll_handles = []
    if sys.platform == "win32":
        for path in (acados / "bin", acados / "lib"):
            if path.is_dir():
                dll_handles.append(os.add_dll_directory(str(path)))
    return root, acados, dll_handles


def main():
    root, acados, dll_handles = configure_paths()
    import casadi as ca
    import numpy as np
    import torch
    import acados_template
    from acados_template import AcadosOcp, AcadosOcpBatchSolver
    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch
    import leap_c.diff_mpc.utils.create_solver as solver_module

    print("Python:", sys.executable, flush=True)
    print("acados_template:", acados_template.__file__, flush=True)
    print("Leap-C solver module:", solver_module.__file__, flush=True)
    loaded = Path(acados_template.__file__).resolve()
    if not loaded.is_relative_to((acados / "interfaces" / "acados_template").resolve()):
        raise RuntimeError("This environment loaded a different acados_template checkout.")
    if "cmake_builder" not in inspect.signature(AcadosOcpBatchSolver).parameters:
        raise RuntimeError("Install the patched acados_ocp_batch_solver.py first.")
    if "cmake_builder" not in inspect.getsource(solver_module.create_batch_solver):
        raise RuntimeError("Install the matching patched Leap-C create_solver.py first.")

    steps = 12
    dt = 0.1
    manager = AcadosParameterManager(N_horizon=steps)
    mass = manager.register_parameter("mass", default=np.array([1.0]), differentiable=False)
    terminal = manager.register_parameter(
        "Q_terminal", default=np.array([2.0, 0.5]), differentiable=True
    )
    state = ca.SX.sym("state", 2)
    control = ca.SX.sym("control", 1)
    ocp = AcadosOcp()
    ocp.model.name = "leapc_windows_terminal_smoke"
    ocp.model.x = state
    ocp.model.u = control
    ocp.model.disc_dyn_expr = ca.vertcat(
        state[0] + dt * state[1], state[1] + dt * control[0] / mass
    )
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        state[0] ** 2 + 0.1 * state[1] ** 2 + 0.05 * control[0] ** 2
    )
    ocp.model.cost_expr_ext_cost_e = (
        terminal[0] * state[0] ** 2 + terminal[1] * state[1] ** 2
    )
    ocp.constraints.x0 = np.array([1.0, 0.0])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-10.0])
    ocp.constraints.ubu = np.array([10.0])
    opts = ocp.solver_options
    opts.N_horizon = steps
    opts.tf = steps * dt
    opts.integrator_type = "DISCRETE"
    opts.nlp_solver_type = "SQP"
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.qp_solver_ric_alg = 0
    opts.qp_solver_cond_N = steps
    opts.hessian_approx = "EXACT"
    opts.nlp_solver_max_iter = 50
    opts.nlp_solver_tol_stat = 1e-9
    opts.nlp_solver_tol_eq = 1e-9
    opts.nlp_solver_tol_ineq = 1e-9
    opts.nlp_solver_tol_comp = 1e-9

    # Start with one solver; the two-row input also exercises batch growth.
    print("Creating the differentiable MPC; first run compiles C code...", flush=True)
    mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        export_directory=root / "leapc_smoke_build",
        n_batch_init=1,
        num_threads_batch_solver=2,
        dtype=torch.float64,
        verbose=True,
    )
    x0 = torch.tensor([[1.0, 0.0], [-0.8, 0.1]], dtype=torch.float64)
    weights = torch.tensor(
        [[2.0, 0.5], [2.0, 0.5]], dtype=torch.float64, requires_grad=True
    )

    def objective(u0, value):
        # Exercise sensitivities of both the optimal value and optimal action.
        return value.sum() + 0.01 * u0.square().sum()

    def check_status(ctx):
        status = ctx.status
        if torch.is_tensor(status):
            status = status.detach().cpu().numpy()
        status = np.asarray(status)
        if not np.all(status == 0):
            raise RuntimeError(f"The MPC did not converge successfully: {status}")
        return status

    ctx, u0, _, _, value = mpc(x0, params={"Q_terminal": weights})
    status = check_status(ctx)
    loss = objective(u0, value)
    if not torch.isfinite(loss):
        raise RuntimeError("The objective is not finite.")
    loss.backward()
    if weights.grad is None or not torch.isfinite(weights.grad).all():
        raise RuntimeError("Terminal-cost gradients are missing or non-finite.")
    if not torch.any(weights.grad.abs() > 1e-12):
        raise RuntimeError("All terminal-cost gradients are zero.")
    analytic = weights.grad[0, 0].item()
    print("Solver status:", status, flush=True)
    print("First actions:", u0.detach().cpu().numpy(), flush=True)
    print("Terminal-cost gradient:", weights.grad.detach().cpu().numpy(), flush=True)

    epsilon = 1e-4
    with torch.no_grad():
        losses = []
        for sign in (1.0, -1.0):
            perturbed = weights.detach().clone()
            perturbed[0, 0] += sign * epsilon
            new_ctx, new_u0, _, _, new_value = mpc(
                x0, params={"Q_terminal": perturbed}
            )
            check_status(new_ctx)
            losses.append(objective(new_u0, new_value).item())
    finite_difference = (losses[0] - losses[1]) / (2.0 * epsilon)
    print("Autograd dL/dQ_terminal[0,0]:", analytic, flush=True)
    print("Finite-difference gradient:", finite_difference, flush=True)
    np.testing.assert_allclose(analytic, finite_difference, rtol=5e-3, atol=5e-5)
    print("PASS: batched MPC, terminal-cost backpropagation and gradient check.", flush=True)
    # Keep Windows DLL directory handles alive until all solver calls finish.
    _ = dll_handles


if __name__ == "__main__":
    main()
