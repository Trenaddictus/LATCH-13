"""Vehicle smoke test, teacher data, terminal-weight training and closed-loop evaluation."""
import json
import time

import casadi as ca
import numpy as np

from .runtime import DATA, fingerprint, parameters, workspace
from .vehicle import (CartesianPlant, initial_state, prediction_model, preview,
                      project, reference, rk4_map)


def check_models(settings):
    """No compiler needed: compare discretization, Jacobian and Cartesian/Frenet dynamics."""
    ref, model = reference(), prediction_model()
    step = rk4_map(model, settings["prediction_dt"], settings["rk_steps"])
    plant = CartesianPlant(settings)
    rng = np.random.default_rng(settings["seed"])
    max_rk, max_dynamics, max_roundtrip, max_jac = 0., 0., 0., 0.
    integrator = ca.integrator("reference_erk", "rk",
        {"x": model.x, "p": ca.vertcat(model.u, model.p), "ode": model.f_expl_expr},
        0., settings["prediction_dt"], {"number_of_finite_elements": settings["rk_steps"]})
    jac = ca.Function("step_jac", step.sx_in(), [ca.jacobian(step(*step.sx_in()), step.sx_in(0))])
    for _ in range(20):
        x = initial_state(ref, settings, s=rng.uniform(0., ref.L), ey=rng.uniform(-.4, .4),
                          epsi=rng.uniform(-.04, .04), speed_offset=rng.uniform(-1., 1.))
        x[4:] = [0.05, 0.02, 0.01, 0.2]
        u = np.array([0.3, 0.02]); p = preview(ref, x, 2, settings["prediction_dt"])[0, 0]
        y1 = np.asarray(step(x, u, p)).ravel()
        y2 = np.asarray(integrator(x0=x, p=np.r_[u, p])["xf"]).ravel()
        max_rk = max(max_rk, np.max(np.abs(y1-y2)))
        cart = ref.frenet_to_cartesian(x)
        projected = project(ref, cart, x[0])
        max_roundtrip = max(max_roundtrip, np.max(np.abs(projected-x)))
        a = np.asarray(model.f(x, u, p)).ravel()[3:7]
        b = np.asarray(plant.f(cart[:7], [x[7], u[1]])).ravel()[3:7]
        max_dynamics = max(max_dynamics, np.max(np.abs(a-b)))
        eps = 1e-5
        Jfd = np.column_stack([(np.asarray(step(x+eps*np.eye(8)[i], u, p)).ravel()
                                -np.asarray(step(x-eps*np.eye(8)[i], u, p)).ravel())/(2*eps)
                               for i in range(8)])
        max_jac = max(max_jac, np.max(np.abs(np.asarray(jac(x, u, p))-Jfd)))
    # Distinguish the two time intervals explicitly (jerk=1 m/s^3).
    cart = ref.frenet_to_cartesian(initial_state(ref, settings))
    if settings["actuation"] == "consistent_jerk":
        np.testing.assert_allclose(plant.step(cart, [1., 0.])[7], settings["simulation_dt"], atol=1e-12)
    assert max_rk < 1e-9, max_rk
    assert max_dynamics < 1e-9, max_dynamics
    assert max_roundtrip < 1e-5, max_roundtrip
    assert max_jac < 2e-6, max_jac
    report = dict(max_RK4_error=max_rk, max_vehicle_rhs_error=max_dynamics,
                  max_projection_error=max_roundtrip, max_jacobian_error=max_jac)
    print(json.dumps(report, indent=2))
    print("PASS: LATCH vehicle dynamics, RK4 discretization, projection, and actuator timing.", flush=True)
    return report


def smoke(settings, verbose=False):
    import torch
    from .controller import VehicleMpc
    ref = reference()
    mpc = VehicleMpc(settings, settings["short_horizon"], verbose=verbose)
    # Smallest action changes in this check are ~1e-8: use tighter tolerances
    # than in general simulation. Warm retries continue max-iteration solves.
    for solver in mpc.module.diff_mpc_fun.forward_batch_solver.ocp_solvers:
        for field in ["tol_stat", "tol_eq", "tol_ineq", "tol_comp",
                      "qp_tol_stat", "qp_tol_eq", "qp_tol_ineq", "qp_tol_comp"]:
            solver.options_set(field, 1e-10)
    x = initial_state(ref, settings, ey=.25, epsi=.015, speed_offset=-1.)
    x[7] = .2
    roads = preview(ref, x, mpc.N, settings["prediction_dt"])
    P = torch.tensor(mpc.P0[None], dtype=torch.float64, requires_grad=True)
    ctx, u, _, _, value = mpc.batch(x, roads, P)
    # Test both value gradients and the action sensitivities needed by imitation learning.
    steering_grad = torch.autograd.grad(u[0, 1], P, retain_graph=True)[0].numpy().ravel().copy()
    loss = value.sum() + 0.01*(u/torch.tensor([8., .322])).square().sum()
    loss.backward()
    assert P.grad is not None and torch.isfinite(P.grad).all()
    analytic = P.grad.numpy().ravel().copy()
    numerical, steering_fd = np.zeros(3), np.zeros(3)
    for i in range(3):
        eps = 1e-3*max(mpc.P0[i], 1.)
        vals, steer = [], []
        for sign in [1., -1.]:
            pp = mpc.P0.copy(); pp[i] += sign*eps
            with torch.no_grad():
                _, ui, _, _, vi = mpc.batch(x, roads, pp, ctx=ctx)
                vals.append(float((vi.sum()+.01*(ui/torch.tensor([8., .322])).square().sum())))
                steer.append(float(ui[0, 1]))
        numerical[i] = (vals[0]-vals[1])/(2*eps)
        steering_fd[i] = (steer[0]-steer[1])/(2*eps)
    print("First action:", u.detach().numpy())
    print("Terminal weight gradients:", analytic)
    print("Finite differences:", numerical)
    print("Steering-action gradient:", steering_grad)
    print("Steering-action finite differences:", steering_fd)
    np.testing.assert_allclose(analytic, numerical, rtol=.05, atol=2e-5)
    np.testing.assert_allclose(steering_grad, steering_fd, rtol=.05, atol=1e-6)
    assert np.max(np.abs(analytic)) > 1e-9
    assert np.max(np.abs(steering_grad)) > 1e-10
    print("PASS: LATCH vehicle MPC and terminal-cost action/value gradient check.", flush=True)


def collect(settings, samples, verbose=False):
    import torch
    from .controller import VehicleMpc
    ref = reference()
    teacher = VehicleMpc(settings, settings["long_horizon"], verbose=verbose)
    student = VehicleMpc(settings, settings["short_horizon"], verbose=verbose)
    rng = np.random.default_rng(settings["seed"])
    states, actions, roads, splits, rejected, boundary_excluded = [], [], [], [], 0, 0
    for i in range(samples):
        s = (i + rng.uniform(.1, .9))*ref.L/samples
        x = initial_state(ref, settings, s=s, ey=rng.uniform(-.5, .5),
                          epsi=rng.uniform(-.04, .04), speed_offset=rng.uniform(-2., 2.))
        r = ref.ref_at(s)
        # Start near steady cornering, with modest perturbations, rather than with r=delta=0
        # at every curved-road sample. The first baseline still uses the original zero values.
        x[5] = x[3]*r["kappa"] + rng.uniform(-.02, .02)
        x[6] = np.arctan(3.128*r["kappa"]) + rng.uniform(-.01, .01)
        x[4] = 1.644*x[5]
        x[7] = rng.uniform(.1, .3)
        p = preview(ref, x, teacher.N, settings["prediction_dt"])
        p_short = preview(ref, x, student.N, settings["prediction_dt"])
        group = int(10*s/ref.L)
        sample_split = 1 if group in [2, 7] else 2 if group == 9 else 0
        # Exclude training starts whose preview reaches a held-out spatial region.
        forecast_s = s + max(x[3], 1.)*settings["prediction_dt"]*np.arange(teacher.N+1)
        forecast_groups = (10*np.mod(forecast_s, ref.L)/ref.L).astype(int)
        if sample_split == 0 and np.isin(forecast_groups, [2, 7, 9]).any():
            boundary_excluded += 1
            continue
        try:
            with torch.no_grad():
                _, u, _, _, _ = teacher.batch(x, p)
                student.batch(x, p_short)
        except RuntimeError as error:
            rejected += 1
            print(f"Rejected sample {i} (teacher or initial student failed): {error}", flush=True)
            continue
        states.append(x); actions.append(u[0].numpy())
        roads.append(p_short[0])
        # Spatial groups 2,7 are validation; group 9 is an untouched test set.
        splits.append(sample_split)
        if (i+1) % 10 == 0:
            print(f"Teacher data: {i+1}/{samples}, accepted={len(states)}", flush=True)
    if len(states) < 12 or not all(np.any(np.asarray(splits) == i) for i in [0, 1, 2]):
        raise RuntimeError("Insufficient converged data in all spatial splits. Increase --samples.")
    path = workspace()/"teacher_data.npz"
    np.savez_compressed(path, states=np.asarray(states), actions=np.asarray(actions),
                        roads_short=np.asarray(roads), split=np.asarray(splits),
                        fingerprint=fingerprint(settings), N_teacher=teacher.N,
                        rejected=rejected, boundary_excluded=boundary_excluded, seed=settings["seed"])
    print(f"Saved {path}: {len(states)} accepted, {rejected} failed, {boundary_excluded} split-boundary exclusions.", flush=True)


def load_dataset(settings):
    path = workspace()/"teacher_data.npz"
    if not path.is_file():
        raise RuntimeError("Collect teacher data first: python -m latch_leapc collect")
    d = np.load(path, allow_pickle=False)
    if str(d["fingerprint"]) != fingerprint(settings):
        raise RuntimeError("Model/config/code changed since data collection. Re-run collect.")
    return d


def train(settings, epochs, verbose=False):
    import torch
    from .controller import VehicleMpc
    d = load_dataset(settings)
    cfg = settings["training"]
    mpc = VehicleMpc(settings, settings["short_horizon"], batch_size=cfg["batch_size"], verbose=verbose)
    # Learn bounded positive multipliers; theta=0 is exactly the original terminal cost.
    theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    base = torch.as_tensor(mpc.P0, dtype=torch.float64)
    opt = torch.optim.Adam([theta], lr=cfg["learning_rate"])
    scales = torch.tensor([8., .322], dtype=torch.float64)
    rng = np.random.default_rng(settings["seed"])
    train_idx, val_idx = [np.flatnonzero(d["split"] == i) for i in [0, 1]]
    N = mpc.N

    def batch_loss(idx):
        ctx, u, _, _, _ = mpc.batch(d["states"][idx], d["roads_short"][idx], base*theta.exp())
        target = torch.as_tensor(d["actions"][idx], dtype=torch.float64)
        return ((u-target)/scales).square().mean()

    def validation():
        loss_sum, count = 0., 0
        with torch.no_grad():
            for i in range(0, len(val_idx), cfg["batch_size"]):
                idx = val_idx[i:i+cfg["batch_size"]]
                loss_sum += float(batch_loss(idx))*len(idx)
                count += len(idx)
        return loss_sum/count

    best = validation()
    best_theta = theta.detach().clone()
    stale_epochs = 0
    history = [{"epoch": 0, "validation_action_mse": best, "weights": mpc.P0.tolist()}]
    print(f"Initial validation loss: {best:.8g}", flush=True)
    for epoch in range(1, epochs+1):
        idx_all = rng.permutation(train_idx)
        loss_sum, count = 0., 0
        for i in range(0, len(idx_all), cfg["batch_size"]):
            idx = idx_all[i:i+cfg["batch_size"]]
            opt.zero_grad()
            loss = batch_loss(idx)
            loss.backward()
            if theta.grad is None or not torch.isfinite(theta.grad).all():
                raise RuntimeError("Terminal weight gradient is missing/non-finite. Run smoke first.")
            torch.nn.utils.clip_grad_norm_([theta], 10.)
            opt.step()
            with torch.no_grad():
                theta.clamp_(np.log(cfg["multiplier_min"]), np.log(cfg["multiplier_max"]))
            loss_sum += float(loss.detach())*len(idx); count += len(idx)
        score = validation()
        if score < best:
            best, best_theta = score, theta.detach().clone()
            stale_epochs = 0
        else:
            stale_epochs += 1
        weights = (base*theta.detach().exp()).tolist()
        row = {"epoch": epoch, "training_action_mse": loss_sum/count,
               "validation_action_mse": score, "weights": weights}
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: train={loss_sum/count:.7g}, val={score:.7g}, P={weights}", flush=True)
        # Save each completed epoch, so a long training run can be interrupted safely.
        (workspace()/"training_history.json").write_text(json.dumps(history, indent=2))
        (workspace()/"terminal_weights.json").write_text(json.dumps({
            "fingerprint": fingerprint(settings), "method": "long-horizon action imitation",
            "P_terminal": (base*best_theta.exp()).tolist(), "best_validation_loss": best,
            "epochs_completed": epoch, "short_horizon": N,
            "prediction_dt": settings["prediction_dt"], "actuation": settings["actuation"],
        }, indent=2))
        if stale_epochs >= cfg["early_stopping_patience"]:
            print(f"Early stopping: no validation improvement for {stale_epochs} epochs.", flush=True)
            break
    print("Saved best validation weights. Test-group data were not used for training/selection.", flush=True)


def load_weights(settings):
    path = workspace()/"terminal_weights.json"
    if not path.is_file():
        raise RuntimeError("Train terminal weights first: python -m latch_leapc train")
    value = json.loads(path.read_text())
    if value["fingerprint"] != fingerprint(settings):
        raise RuntimeError("Model/config/code changed since training. Recollect and retrain.")
    return np.asarray(value["P_terminal"], dtype=float)


def rollout(settings, horizon, steps, start_fraction=0., weights=None, verbose=False, viewer=None):
    from .controller import VehicleMpc
    ref, plant = reference(), CartesianPlant(settings)
    mpc = VehicleMpc(settings, horizon, verbose=verbose)
    xf = initial_state(ref, settings, s=start_fraction*ref.L)
    cart = ref.frenet_to_cartesian(xf)
    Xf, Xc, U, timings = [xf.copy()], [cart.copy()], [], []
    failure = None
    for i in range(steps):
        try:
            u, xp, _, _, elapsed = mpc.solve(xf, ref, weights)
            cart = plant.step(cart, u, xp[1, 7])
            xf = project(ref, cart, xf[0])
            if not np.isfinite(xf).all() or abs(xf[2]) > np.pi/2 or xf[3] < 1.:
                raise RuntimeError("Vehicle left the forward-driving domain of the Frenet model.")
        except RuntimeError as error:
            failure = f"Step {i}: {error}"
            print(failure, flush=True)
            if viewer is not None:
                viewer.failed(failure)
            break
        Xf.append(xf.copy()); Xc.append(cart.copy()); U.append(u); timings.append(elapsed)
        if viewer is not None:
            viewer.update(i, xf, cart, xp, u, elapsed)
        if (i+1) % 50 == 0:
            print(f"Closed loop: {i+1}/{steps}, ey={xf[1]:.3f}m, vx={xf[3]:.2f}m/s", flush=True)
    x = np.asarray(Xf); r = ref.ref_at(x[:, 0]); mp, cfg = parameters()
    import yaml
    veh = yaml.safe_load((cfg/"veh_params_pred.yaml").read_text())
    w = veh["veh_width"]/2 + mp["track_margin"]
    margin = np.minimum(x[:, 1]-r["ey_min"], r["ey_max"]-x[:, 1])-w
    ev = x[:, 3]-r["v_ref"]
    g = np.genfromtxt(cfg/"ggv.csv", delimiter=",", names=True)
    ax = np.where(x[:, 7] < 0, -veh["acc_min"], np.interp(x[:, 3], g["vel_max_mps"], g["ax_max_mps2"]))
    ay = np.interp(x[:, 3], g["vel_max_mps"], g["ay_max_mps2"])
    circle = (x[:, 7]/ax)**2+(x[:, 3]*x[:, 5]/ay)**2
    metrics = {
        "horizon_steps": horizon, "horizon_seconds": horizon*settings["prediction_dt"],
        "requested_steps": steps, "completed_steps": len(U), "failure": failure,
        "rms_lateral_error_m": float(np.sqrt(np.mean(x[:, 1]**2))),
        "rms_heading_error_rad": float(np.sqrt(np.mean(x[:, 2]**2))),
        "rms_speed_error_mps": float(np.sqrt(np.mean(ev**2))),
        "minimum_track_margin_m": float(np.min(margin)),
        "track_violation_samples": int(np.count_nonzero(margin < -1e-5)),
        "max_acceleration_circle_excess": float(max(0., np.max(circle)-1.)),
        "max_steering_bound_excess_rad": float(max(0., np.max(x[:, 6]-veh["delta_f_max"]),
                                                  np.max(veh["delta_f_min"]-x[:, 6]))),
        "max_acceleration_bound_excess_mps2": float(max(0., np.max(x[:, 7]-veh["acc_max"]),
                                                      np.max(veh["acc_min"]-x[:, 7]))),
        "mean_solve_ms": float(np.mean(timings)*1e3) if timings else None,
        "p95_solve_ms": float(np.quantile(timings, .95)*1e3) if timings else None,
        "max_solve_ms": float(np.max(timings)*1e3) if timings else None,
        "deadline_miss_samples": int(np.count_nonzero(np.asarray(timings) > settings["simulation_dt"])),
        "actuation": settings["actuation"], "start_fraction": start_fraction,
    }
    return {"frenet": x, "cartesian": np.asarray(Xc), "actions": np.asarray(U),
            "solve_seconds": np.asarray(timings), "track_margin": margin,
            "speed_reference": r["v_ref"], "metrics": metrics}


def save_rollout(settings, name, result):
    out = workspace()
    np.savez_compressed(out/f"{name}.npz", **{k: v for k, v in result.items() if k != "metrics"})
    (out/f"{name}_metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["metrics"], indent=2), flush=True)


def plot_rollouts(settings, results, filename):
    # Write plots to disk; no blocking GUI loop during control or training.
    # import matplotlib
    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Object-oriented Figure (no pyplot): saves to file without switching the global
    # backend, so a --live window (TkAgg) stays open while the summary plots are written.
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 8))
    ax = fig.subplots(2, 2)
    ref = reference()
    ax[0, 0].plot(ref.x, ref.y, "k--", alpha=.4, label="reference")
    for name, data in results.items():
        t = np.arange(len(data["frenet"]))*settings["simulation_dt"]
        ax[0, 0].plot(data["cartesian"][:, 0], data["cartesian"][:, 1], label=name)
        ax[0, 1].plot(t, data["frenet"][:, 1], label=name)
        ax[1, 0].plot(t, data["frenet"][:, 3]-data["speed_reference"], label=name)
        ax[1, 1].plot(t[1:], 1e3*data["solve_seconds"], label=name)
    ax[0, 0].set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax[0, 1].set(xlabel="time [s]", ylabel="lateral error [m]")
    ax[1, 0].set(xlabel="time [s]", ylabel="speed error [m/s]")
    ax[1, 1].set(xlabel="time [s]", ylabel="solve wall time [ms]")
    ax[1, 1].axhline(settings["simulation_dt"]*1e3, color="k", ls="--", label="control period")
    for a in ax.flat:
        a.legend(fontsize=8); a.grid(True, alpha=.3)
    fig.tight_layout()
    path = workspace()/filename
    fig.savefig(path, dpi=150)
    # plt.close(fig)
    print("Plot:", path, flush=True)
