# latch_leapc — analysis of the LATCH-13 × Leap-C terminal-cost project

Snapshot: 2026-11-13, working copy at `/home/david/LATCH` (WSL2 Ubuntu, glibc 2.35, 16 cores).
Written from a full read of the code and from runs made on this machine on that date.

---

## 1. What the program is

`latch_leapc` is a Python package placed in the root of a **Leap-C** checkout. It takes the vehicle
and track setup from the **LATCH-13** Frenet NMPC (SSY226 project, built on **TUM-CONTROL**) and
rebuilds that OCP in acados through Leap-C's differentiable-MPC layer. It then tests one research
question:

> Can a **short-horizon** MPC (N=10, 0.8 s) be made to behave like the **long-horizon** LATCH MPC
> (N=38, 3.04 s) by **learning only the three terminal-cost weights**
> `P = [P_ey, P_epsi, P_v]`, with gradients that pass through the MPC solver?

Method: imitation learning. The long-horizon MPC is the teacher. It labels sampled states with its
first action, and the short-horizon MPC's `P_terminal` is trained so that its first action matches.
The gradient dAction/dP comes from Leap-C's implicit differentiation of the acados KKT system.

Where each part comes from:

| Part | Origin |
|---|---|
| Frenet single-track model with Pacejka tyres (`vendor/frenet_model.py`) | LATCH / SSY226 Frenet NMPC. The force equations are identical to TUM-CONTROL `pred_model_dynamic_stm_pacejka.py` (B. Zarrouki); only the pose kinematics are Frenet |
| Cartesian plant simulator (`vendor/plant_model.py`, `sim_stm`) | TUM-CONTROL simulator model (B. Zarrouki), LGPL-3 (`vendor/LICENSE_LATCH`) |
| Cartesian↔Frenet projection (`vendor/frenet_projection.py`) | LATCH project |
| EDGAR vehicle, Pacejka, ggv and MPC params (`data/EDGAR_Frenet/*.yaml`, `ggv.csv`) | LATCH `Config/EDGAR_Frenet` |
| Track: scaled F1TENTH Silverstone, 5891 m lap, 2232 points, v_ref 12.8–37.5 m/s | LATCH `Utils/preprocess_f1tenth_track.py` |
| OCP construction, parameter manager, differentiable solve | Leap-C (`AcadosParameterManager`, `AcadosDiffMpcTorch`) on acados v0.5.3-54 (`6111f01f2`) |
| Pipeline: teacher data, training, evaluation | new code in `latch_leapc/` |

---

## 2. Package layout

```
latch_leapc/
  __main__.py      CLI: python -m latch_leapc {all,check,smoke,baseline,collect,train,compare}
  runtime.py       environment bootstrap (acados path, temp dir, Windows DLLs), settings, fingerprint
  vehicle.py       prediction model, RK4 map, road preview, Cartesian plant adapter, projection
  controller.py    build_ocp(): LATCH OCP in acados via Leap-C; VehicleMpc wrapper with retries
  experiments.py   check_models, smoke, collect, train, rollout, metrics, plots
  settings.yaml    horizons, dt, solver tolerances, training hyperparameters
  vendor/          LATCH / TUM-CONTROL sources (see above)
  data/            Silverstone reference + EDGAR_Frenet config
latch_leapc_build/N10, N38/   generated + compiled acados solvers (forward and backward)
latch_leapc_results/          .npz rollouts, *_metrics.json, plots, teacher_data.npz, weights
```

---

## 3. The model and the OCP (`controller.build_ocp`)

**State (8):** `x = [s, e_y, e_psi, v_x, v_y, psi_dot, delta, a]`
**Input (2):** `u = [jerk, delta_dot]`
**Per-stage non-differentiable parameter `road` (4):** `[kappa, v_ref, ey_min, ey_max]`
**Global differentiable parameter `P_terminal` (3):** default `P0 = [2.8, 0.4, 2.0]`, equal to the stage weights

**Dynamics:** `DISCRETE` integrator type. `disc_dyn_expr` is an explicit RK4 with 3 substeps per
0.08 s interval, written in CasADi (`vehicle.rk4_map`). This reproduces the original ERK settings.
Leap-C's parameter-sensitivity interface requires a discrete integrator.

**Cost (EXTERNAL):**
- Stage: `0.005·(q·ξ² + r·u²)`, with `ξ = [e_y, e_psi, v_x − v_ref]`, `q = [2.8, 0.4, 2.0]` and `r = [38.1, 101.4]`
- Terminal: `0.005·(P_terminal · ξ²)`. This is the only learned quantity.
- `q_v = 2.0` was retuned on 2026-11-11 (TUM's value was 0.2).

**Constraints:**
- Hard input bounds on jerk and delta_dot.
- Soft state bounds on delta and a (`idxsbx`).
- Soft nonlinear `h`, applied to the initial, path and terminal stages:
  1. Combined acceleration circle `(a/ax_max(v))² + (v·psi_dot/ay_max(v))² ≤ 1`, using ggv lookup and `acc_min` when braking.
  2. Track bounds `e_y − ey_min ≥ w` and `ey_max − e_y ≥ w`, with `w = veh_width/2 + track_margin` (margin 0).
- Slack penalties: L1 = 106.7 and L2 = 9.9 for actuator/acceleration constraints; 1000 for track bounds.

**Solver:** full `SQP`, `GAUSS_NEWTON`, `PARTIAL_CONDENSING_HPIPM` (cond_N=N),
`MERIT_BACKTRACKING`, max 100 iterations, all tolerances 1e-7.
The original LATCH controller uses `SQP_RTI`. Here full SQP is required, because implicit
differentiation needs a converged KKT point. Closed-loop timings are therefore **not** comparable
to the RTI controller.
acados warns about GN combined with an EXTERNAL cost. With that combination it uses the exact cost
Hessian and leaves out constraint and dynamics curvature. Leap-C's backward solver uses the exact
Hessian for the sensitivities.

**Initializer:** constant-speed guess along s, with v_y and psi_dot decaying, and u=0.
In closed loop the previous `ctx` is passed back as a warm start.

**Convergence handling (`VehicleMpc.batch`):** if any batch element returns status 2
(max iterations) and the others are 0, the solve is restarted from its own iterate, up to 2 more
times. If any element still has a nonzero status, `RuntimeError` is raised and no action is used.

**Road preview (`vehicle.preview`):** fixed per solve. Nodes sit at `s + max(v_x,1)·dt·k`. Curvature
is taken at interval midpoints and the other fields at nodes. The preview lies outside the
differentiable graph.

---

## 4. Simulation loop (`experiments.rollout`)

1. Start the Frenet state on the reference at `start_fraction·L` (v = v_ref, everything else 0) and convert it to Cartesian.
2. Every 0.02 s (`simulation_dt`): MPC solve → `CartesianPlant.step` → project back to Frenet, using the previous s as a guess and handling lap wrap-around.
3. Actuation mode `consistent_jerk` (default): jerk and steering rate are held over the 0.02 s plant step, and acceleration is integrated inside RK4 as an 8th state. `latch_legacy` instead reproduces the old convention, which applied the predicted node-1 acceleration.
4. Stop on solver failure, non-finite state, |e_psi| > π/2 or v_x < 1.
5. Metrics: RMS of e_y, e_psi and speed error; minimum track margin and violation count; constraint excess (acceleration circle, steering, acceleration); solve times (mean, p95, max); `deadline_miss_samples` (solves slower than 20 ms).

Note: the MPC predicts at 0.08 s but is re-solved every 0.02 s.

---

## 5. The learning pipeline

### `collect` (teacher data)
- 128 arc-length-stratified samples. Each start state has random perturbations: e_y ±0.5, e_psi ±0.04, v ±2. Yaw rate, steering and v_y are set near steady cornering (`x[5]=v·kappa`, `x[6]=atan(3.128·kappa)`, `x[4]=1.644·x[5]`, plus noise), and a ∈ [0.1, 0.3].
- **Spatial split:** the lap is cut into 10 arc-length groups. Groups 2 and 7 are validation, group 9 is the **untouched test set**, and the rest are training. A training sample is dropped if its N=38 preview reaches into groups 2, 7 or 9 (boundary exclusion).
- A sample is kept only if both the teacher (N=38) and the untrained student (N=10) converge.
- Saved per sample: state, the teacher's first action, the student's road preview, and the split label.

### `train`
- Parameterisation: `P = P0 · exp(θ)` with θ ∈ ℝ³ and θ=0 at the start. θ is clamped to multipliers between 0.05 and 100.
- Loss: MSE between the student's first action and the teacher's, scaled by `[8, 0.322]` for (jerk, delta_dot).
- Adam, lr 0.08, batch 8, up to 30 epochs, gradient-norm clip 10, early stopping with patience 5 on validation loss.
- Writes `terminal_weights.json` (best validation θ) and `training_history.json` after every epoch.

### `compare`
- Runs `long_fixed` (N=38, P0), `short_fixed` (N=10, P0) and `short_learned` (N=10, learned P) from `start_fraction = 0.92`, which is inside test group 9.
- Raises an error if the default window (steps plus the long preview) would run past the lap end into training data.

### Fingerprint
`runtime.fingerprint` hashes `settings.yaml` and **every** `.py`, `.yaml`, `.csv` and `.json` file in the package.
`train` refuses a dataset, and `compare` refuses weights, whose fingerprint does not match. As a
result, **any code edit, including to `__main__.py` or the plotting code, invalidates existing data
and weights**, and `collect` + `train` must be run again.

### Sanity checks
- `check`: no compiler needed. Over 20 random states it compares the CasADi RK4 against a CasADi ERK integrator, the Frenet RHS against the Cartesian plant RHS, the Frenet→Cartesian→Frenet round trip, and the RK4 Jacobian against finite differences. It also checks jerk-integration timing.
- `smoke`: builds the N=10 MPC with tolerances 1e-10. It checks dValue/dP and dSteering/dP against central finite differences (rtol 5%).

---

## 6. Environment and patches

- **Current platform:** WSL2 Linux, Python 3.11.18 venv (`.venv`), torch 2.15.0+cpu, casadi 3.8.2, numpy 2.4.6. Leap-C is installed editable from this repo, and acados comes from `external/acados` (Linux `libacados.so` in `lib/`).
- **Earlier platform:** native Windows (MSVC, VS 2022, Anaconda `leapc` env). Leftovers from that setup:
  - `apply_leapc_windows_fix{,_v2}.py`, `rebuild_acados_windows_v2.cmd`, `smoke_test_leapc_windows.py`, `windows_fix_v2_manifest.json`, and `README_WINDOWS_FIX*.md` (in Chinese).
  - `payload/` and `patches/` hold the patched files.
  - The patch moves OpenMP loop variables out of 29 `for` loops in 5 acados C files and in 6 functions of `acados_solver.in.c`. It also adds an MSVC branch to `CMakeLists.in.txt`, passes a cmake builder through the batch solver, and in `leap_c/diff_mpc/utils/create_solver.py` invalidates the code-reuse cache when the solver DLL is missing.
  - These source patches **are still applied** in this Linux tree: there are real diffs in `external/acados` (5 C files and 3 template files) and in `create_solver.py` (+25 lines). The Windows-only branches are inactive on Linux. Beyond that, `create_solver.py` only resolves the export directory and creates it.
  - `external/acados/build_msvc_*` directories and `.bak` files are also still present.
- Most "modified" files in `git status` differ only by CRLF line endings, because the files were copied from Windows.
- `runtime.configure` sets `ACADOS_SOURCE_DIR`, adds acados `bin`/`lib` to PATH and points TEMP at `latch_leapc_tmp/`. It also asserts that `acados_template` is imported from the bundled acados.
- **Cross-platform reproducibility:** the Linux run reproduced the Windows metrics exactly, to about 1e-15. The examples are baseline_short and the N=38 run from start 0 for 200 steps. Only solve times differ.

---

## 7. Results so far

### 7a. Earlier full run (Windows, 2026-11-12, now in `latch_leapc_results_backup_20261113/`)

Training: 8 epochs with early stopping. Best at epoch 3: validation MSE 0.006613 (start 0.007786), **learned P = [2.265, 1.055, 4.004]** (P0 = [2.8, 0.4, 2.0]).
Over training, P_epsi and P_v rose steadily while validation loss went up after epoch 3.

Comparison (200 steps = 4 s):

| Run | N | RMS e_y [m] | RMS e_psi [rad] | RMS Δv [m/s] | min margin [m] | mean / p95 solve [ms] | deadline misses |
|---|---|---|---|---|---|---|---|
| long_fixed | 38 | 0.1117 | 0.00468 | 1.669 | 2.47 | 35.4 / 156.7 | 69 |
| short_fixed | 10 | 0.0904 | 0.00513 | 2.510 | 2.52 | 16.0 / 24.1 | 44 |
| short_learned | 10 | 0.1147 | 0.00561 | 2.252 | 2.46 | 16.8 / 32.4 | 63 |

There were no constraint violations in any run. On Windows a 500-step long-horizon baseline failed at step 287 with status 2 (max iterations).

**Important caveat:** all three of these comparison files record `start_fraction: 0.0`, and
short_fixed equals baseline_short exactly. That comparison was therefore made with an earlier code
version that started at s=0, which is the **training region**, not test group 9. Current code
defaults to 0.92. The table is not a held-out evaluation.

Reading: the learned P moved short_learned toward the teacher on speed error (2.51 → 2.25, teacher
1.67) but made it worse on lateral error. The teacher itself has larger lateral error than the
short MPC over this window. Imitation of the first action yields a mixed result on tracking metrics.

### 7b. Run on Linux, 2026-11-13 (`python -m latch_leapc all`)

| Stage | Result |
|---|---|
| check | PASS (RK4 err 5.6e-17, RHS err 0, projection 1.6e-13, Jacobian 8.0e-8) |
| smoke | PASS. dV/dP analytic `[2.3397e-3, 5.0486e-7, 4.4276e-2]` vs FD `[2.3397e-3, 5.0464e-7, 4.4276e-2]`; steering gradient matches to about 1e-8 |
| baseline_short (N=10, s=0) | identical to the Windows run; solve 11.5 ms mean, 18.6 ms p95, 5 deadline misses |
| baseline_long (N=38, s=0) | identical to Windows long_fixed; solve 23.0 ms mean, 103 ms p95, 54 deadline misses |
| collect | 121 accepted, 4 rejected (status 2), 3 boundary exclusions |
| train | **crashed in epoch 2.** Epoch 1: train 0.0285, val 0.007732 (start 0.007773), P=[2.79, 0.91, 2.00] |
| compare | not reached |

Total runtime up to the crash was about 28 s.

---

## 8. Known issues and observations

1. **Training is not robust to a single non-converged solve.** `train()` → `batch_loss` → `VehicleMpc.batch` raises `RuntimeError` when any of the 8 batch elements stays at status 2 after the two warm retries. The error is not caught, so the whole run aborts (seen on 2026-11-13: `status=[0,0,0,0,0,0,2,0]`). Whether this happens depends on P, so it shows up once θ moves away from 0. Possible fixes:
   - skip or mask the failed sample in the loss;
   - skip the batch and count it;
   - raise `nlp_max_iter` or the number of retries for training only;
   - add a P-dependent check on convergence.

   The validation set must be handled consistently whichever fix is chosen.
2. **The old comparison was not on held-out data** (see 7a).
3. **Real-time:** with full SQP, N=38 misses the 20 ms deadline often (p95 100–220 ms), and N=10 is marginal on Windows. Closed-loop timing does not represent an RTI deployment.
4. **The long-horizon MPC is not reliably convergent** (failed at step 287 in a 500-step run; 4 of 128 collect samples rejected). The teacher labels exist only where the teacher converged, so there is a selection bias.
5. **The fingerprint covers all package files**, so cosmetic edits invalidate data and weights.
6. **Weak loss signal:** validation MSE moves only from about 0.0078 to about 0.0066, so action imitation hardly separates candidate P values. P_epsi tends to grow while validation loss worsens. One option is to add a value or closed-loop objective, or to imitate several actions or the predicted trajectory instead of only u₀.
7. **The stage/terminal scaling convention should be confirmed** against LATCH. The yaml comments say "acados scales stage costs by Ts_MPC; terminal not scaled". The code uses an EXTERNAL cost with factor 0.005 on both, and the interpretation of the learned P depends on that relative scaling.
8. The teacher (N=38) has higher lateral RMS than N=10 from s=0 over 4 s. "Imitate the long horizon" therefore does not automatically mean "better tracking" on these metrics. A longer or more curved test window, or lap-time and constraint margins, may be more informative.

---

## 9. How to run

```bash
cd /home/david/LATCH && source .venv/bin/activate
python -m latch_leapc all                 # full pipeline (added 2026-11-13)
python -m latch_leapc check|smoke
python -m latch_leapc baseline --horizon short|long [--steps 200] [--start-fraction f]
python -m latch_leapc collect [--samples 128]
python -m latch_leapc train [--epochs 30]
python -m latch_leapc compare [--steps 200] [--start-fraction f]
```

`all` was added on 2026-11-13. It runs check → smoke → baseline short → baseline long → collect →
train → compare. If a baseline or comparison rollout stops early, it continues and reports a warning
and exit 1 at the end; errors from other stages still abort the run. Single commands behave as before.
Because of the fingerprint, this edit made the 2026-11-12 data and weights stale.

---

## 10. Open questions for the next step

- Which way should non-converged training samples be handled (see issue 8.1)?
- Should evaluation use full laps or several windows across test group 9, and include lap-time and constraint-margin metrics?
- Should the student be compared against an `SQP_RTI` controller for timing, keeping full SQP only for training and gradients?
- Would a different learning target (value function, multi-step actions, closed-loop cost) give a stronger signal than first-action imitation?
