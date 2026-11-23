"""Acceptance test for ROADMAP step 8: MPC gradients against finite differences.

Checks dV/dtheta and du0/dtheta of the Leap-C OCP, through torch autograd, on
sampled states and thetas (default, braking, and random within the RL bounds).

A finite difference is only used as reference where it is itself reliable,
i.e. two step sizes agree to 2e-4. References are tried from most to least
accurate, and the first reliable one is used:

1. exact-Newton polish of the solution (quadratic, but stalls on ~10 %);
2. MIRROR polish to 1e-10 (covers nearly all of those), large steps only.

A warm-started, linearly converging solver stops early after a small
perturbation. Its differences then agree across step sizes and are still
biased, so small steps are not used with MIRROR, and the solver under test is
never its own reference. The problem has several local optima, so polishes and
perturbed solves count only if they stay at the same one. Gradients with |analytic| and |reference| below 1e-6 agree as zero.
Everything without a reliable reference is inconclusive, never passed.

Exit code 0 only if no conclusive check exceeds the 1e-3 tolerance, all base
solves converge and at least 90 % of the checks are conclusive.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from latch13.controllers.leapc_ocp import (  # noqa: E402
    SENSITIVITY_SOLVER_OPTIONS, DiagonalTerminalTheta, build_diff_mpc,
)
from latch13.controllers.ocp import SolverOptions  # noqa: E402
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
TOLERANCE, AGREEMENT = 1e-3, 2e-4
ZERO = 1e-4  # gradients below this are compared absolutely
NEGLIGIBLE = 1e-6  # both below this: output pinned at a bound, gradient zero
STEPS = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5)  # exact Newton
LINEAR_STEPS = (1e-2, 5e-3, 2e-3, 1e-3)  # linearly converging polish
SAME_OPTIMUM = 1e-3  # plan distance above which it is another local optimum


def sample_states(track, count, rng):
    """States near the raceline at speeds the lateral-acceleration limit allows."""
    s = rng.uniform(0, track.length, count)
    kappa = track.curvature(s)
    limit = np.minimum(24.5, np.sqrt(0.85 * 5.886 / np.maximum(np.abs(kappa), 1e-6)))
    vx = limit * rng.uniform(0.8, 1.0, count)
    return np.column_stack((
        s, rng.uniform(-0.3, 0.3, count), rng.uniform(-0.02, 0.02, count), vx,
        np.zeros(count), vx * kappa, 3.128 * kappa, rng.uniform(-1.0, 1.0, count),
    ))


def stage_parameters(track, states, N, dt):
    progress = states[:, :1] + np.arange(N + 1) * dt * states[:, 3:4]
    return np.stack((track.curvature(progress), *track.edges(progress)), axis=-1)


def sample_thetas(count, rng):
    spec = DiagonalTerminalTheta()
    space = spec.space()
    flat = lambda values: np.concatenate([values["log_weights"], values["v_threshold"]])  # noqa: E731
    low = flat({name: box.low for name, box in space.spaces.items()})
    high = flat({name: box.high for name, box in space.spaces.items()})
    thetas = rng.uniform(low, high, (count, 4))
    thetas[0::4] = flat(spec.defaults())
    thetas[1::4] = np.r_[np.log([40.0, 20.0, 1.0]), 15.0]
    return thetas


def solve(mpc, states, track_p, theta, ctx=None):
    theta = torch.as_tensor(theta, dtype=torch.float64)
    return mpc(
        torch.as_tensor(states, dtype=torch.float64), ctx=ctx,
        params={"log_weights": theta[:, :3], "v_threshold": theta[:, 3:], "track": track_p},
    )


def autograd_gradients(mpc, states, track_p, theta):
    """Return (ctx, d[u0, V]/dtheta) with shape (batch, 3, 4), and the state plan."""
    variable = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    ctx, u0, x, _, value = solve(mpc, states, track_p, variable)
    outputs = torch.cat((u0, value), dim=1)
    gradients = [torch.autograd.grad(outputs[:, i].sum(), variable, retain_graph=True)[0]
                 for i in range(outputs.shape[1])]
    return ctx, torch.stack(gradients, dim=1).numpy(), x.detach().numpy()


def central_differences(mpc, states, track_p, theta, ctx, plan, step):
    """Central differences; NaN where a solve fails or leaves the local optimum."""
    result = np.empty((len(states), 3, 4))
    for i in range(4):
        delta = np.eye(4)[i] * step
        plus, up, xp, _, vp = solve(mpc, states, track_p, theta + delta, ctx)
        minus, um, xm, _, vm = solve(mpc, states, track_p, theta - delta, ctx)
        # Smooth branch: the midpoint of both plans is the unperturbed plan.
        jumped = np.abs((xp + xm).numpy() / 2 - plan).max((1, 2)) > SAME_OPTIMUM
        result[:, :, i] = (torch.cat((up, vp), 1) - torch.cat((um, vm), 1)).numpy() / (2 * step)
        result[(plus.status != 0) | (minus.status != 0) | jumped, :, i] = np.nan
    return result


def relative(difference, *values):
    """Max-norm error per output row, absolute below ZERO."""
    scale = np.maximum.reduce([np.abs(v).max(2) for v in values] + [np.full(difference.shape[:2], ZERO)])
    return np.abs(difference).max(2) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--phase1-tolerance", action="store_true",
                        help="test the Phase 1 driving tolerances instead (expected to fail)")
    parser.add_argument("--output", default=str(ROOT / "results" / "generated" / "leapc"))
    args = parser.parse_args()

    rng, dt = np.random.default_rng(args.seed), 0.08
    track = load_track(TRACK_ROOT, args.track, args.scale)
    states = sample_states(track, args.samples, rng)
    track_p = stage_parameters(track, states, args.N, dt)
    theta = sample_thetas(args.samples, rng)

    options = SolverOptions() if args.phase1_tolerance else SENSITIVITY_SOLVER_OPTIONS
    newton_options = replace(
        SENSITIVITY_SOLVER_OPTIONS, regularize_method="NO_REGULARIZE", levenberg_marquardt=0.0,
        globalization="FIXED_STEP", nlp_solver_max_iter=30, nlp_solver_tol_stat=1e-10,
        nlp_solver_tol_comp=1e-10, qp_solver_tol_stat=1e-12, qp_solver_tol_eq=1e-12,
        qp_solver_tol_ineq=1e-12,
    )
    mirror_options = replace(
        SENSITIVITY_SOLVER_OPTIONS, nlp_solver_tol_stat=1e-10, nlp_solver_tol_eq=1e-11,
        qp_solver_tol_stat=1e-12, qp_solver_tol_eq=1e-12, qp_solver_tol_ineq=1e-12,
        nlp_solver_max_iter=3000,
    )

    def build(name, solver_options):
        return build_diff_mpc(
            TUM_CONFIG, Path(args.output) / f"N{args.N}" / name, args.N, dt,
            solver_options=solver_options, n_batch_init=args.samples, dtype=torch.float64,
        )[0]

    mpc = build("phase1" if args.phase1_tolerance else "sensitivity", options)
    polishers = [build("newton_reference", newton_options), build("mirror_reference", mirror_options)]

    ctx, gradient, plan = autograd_gradients(mpc, states, track_p, theta)
    converged = ctx.status == 0
    with torch.no_grad():
        references = []
        for polisher, steps in zip(polishers, (STEPS, LINEAR_STEPS)):
            polished, _, polished_plan, _, _ = solve(polisher, states, track_p, theta, ctx)
            same_optimum = (polished.status == 0) & (
                np.abs(polished_plan.numpy() - plan).max((1, 2)) < SAME_OPTIMUM)
            references.append((polisher, polished, polished_plan.numpy(), same_optimum, steps))

        best = np.full(gradient.shape[:2], np.inf)
        reference = np.full(gradient.shape, np.nan)
        for solver, start, base, usable, steps in references:
            unresolved = best >= AGREEMENT
            values = [central_differences(solver, states, track_p, theta, start, base, h) for h in steps]
            for coarse, fine in zip(values, values[1:]):
                spread = np.nan_to_num(relative(coarse - fine, coarse, fine), nan=np.inf)
                spread[~usable] = np.inf
                better = (spread < best) & unresolved
                best[better], reference[better] = spread[better], fine[better]

    conclusive = (best < AGREEMENT) & converged[:, None]
    error = relative(gradient - reference, gradient, reference)
    error[(np.abs(gradient).max(2) < NEGLIGIBLE) & (np.abs(reference).max(2) < NEGLIGIBLE)] = 0.0

    print(f"\nN={args.N}, {args.samples} problems, seed {args.seed}, "
          f"tol_stat={options.nlp_solver_tol_stat:g}: {converged.sum()} converged")
    failed = 0
    for j, name in enumerate(("djerk/dtheta", "dsteer/dtheta", "dV/dtheta")):
        ok = conclusive[:, j]
        wrong = int((error[ok, j] >= TOLERANCE).sum())
        failed += wrong
        nonzero = int((np.abs(gradient[ok, j]).max(1) > ZERO).sum())
        print(f"RESULT {name:14s} conclusive {ok.sum():4d}/{args.samples}  nonzero {nonzero:4d}  "
              f"median {np.median(error[ok, j]):.1e}  max {error[ok, j].max():.1e}  failed {wrong}")
        for b in np.flatnonzero(ok & (error[:, j] >= TOLERANCE))[:10]:
            print(f"FAILED {name} sample {b}: s={states[b, 0]:.0f} vx={states[b, 3]:.1f} "
                  f"error={error[b, j]:.1e}")
    passed = failed == 0 and converged.all() and conclusive.mean() >= 0.9
    print(f"RESULT conclusive {conclusive.mean():.1%}, failed {failed}: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
