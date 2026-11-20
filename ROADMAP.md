# ROADMAP: Learned terminal cost for short-horizon MPC (SSY226, group 13)

Status as of 2026-11-20. Repo: `~/Refactor`, branch `refactor/clean-architecture`
(remote `git@github.com:Trenaddictus/LATCH-13.git`). Platform: WSL2 Ubuntu 22.04, Python 3.11, Conda env `latch13`.

This file is the working plan for Claude Code. Read it together with:
- `REFACTOR_SUMMARY.md`: description of the current `latch13` code (layout, model, OCP, simulation, results)
- `LEAPC_SUMMARY.md`: Leap-C, `leap-c-lab` and `mpc-sac` APIs, the RL algorithms, and sketches for env/planner/training script

Update the **Status** field of each step when it is started or finished, and add short notes on findings.
Do not mark a step done unless its acceptance criteria are met.

---

## 0. Project goal and fixed decisions

**Goal.** Learn the terminal cost of a short-horizon MPC (N = 10, 0.8 s) with **reinforcement learning**
via Leap-C, so that it approaches the performance of a long-horizon MPC (N = 38, 3.04 s) while keeping
a much lower solve time. Tracks: F1TENTH racetracks (Silverstone first), used as well-defined reference
tracks for general autonomous driving.

**Objective (set by the teacher).** Drive fast (maximise progress / minimise lap time) while keeping
tracking error low and never leaving the track.

**Decisions already made by the group — do not change without asking:**

| # | Decision |
|---|---|
| D1 | **No imitation learning.** The N = 38 MPC is a benchmark only; nothing is trained on its outputs. The old `~/LATCH` / `latch_leapc` imitation pipeline is reference material, not the method. |
| D2 | **No reference speed.** `v_ref` is removed from the cost; speed is optimised through a progress term. |
| D3 | **The car must stay inside the track edges at all times.** Implemented as a track-edge constraint in the MPC (soft, exact L1 penalty) and as episode termination in the RL environment. |
| D4 | **RL is done with Leap-C** (core + `leap-c-lab` + `mpc-sac`). Start with SAC-ZOP, then SAC-FOP. |
| D5 | **Only the terminal cost is learned.** Stage cost weights are fixed during RL. |
| D6 | **Stage cost weights are tuned with Optuna by teammates** (not in this roadmap). Hand-pick reasonable values here; do not spend effort tuning them. |
| D7 | **Build on `latch13` modules.** Import model, track and plant code from the package; do not copy them into scripts. |

---

## 1. Current state (done)

- `latch13` package with Frenet model on top of TUM-CONTROL (`models/frenet_model.py`), OCP builder
  (`controllers/ocp.py`) and closed-loop simulation (`scripts/run_baseline.py`). See `REFACTOR_SUMMARY.md`.
- State `x = [s, ey, epsi, vx, vy, r, delta, a]`, input `u = [jerk, steering_rate]`, parameter `p = [κ]`.
- Current cost uses a **constant `v_ref`**; terminal weights are **baked in at code generation**.
- Constraints: hard bounds on `vx ∈ [1, 25]`, `delta`, `a`, inputs; nonlinear lateral-acceleration bound.
  **No track-edge constraint, no combined friction circle.**
- N = 38 baseline at `v_ref = 5 m/s`: 0 solver failures, RMS ey 0.0038 m, mean solve 13.6 ms, ≈ 4000 m
  driven (less than one 4462 m lap).
- Track boundaries are loaded from the centerline CSV **for plotting only**.
- Known housekeeping debt (see `REFACTOR_SUMMARY.md` §7.6): duplicate track code in `run_baseline.py` vs
  `tracks/f1tenth.py`, stale `tum_adapter.py` / `tum_live.py`, no tests, TUM README, stray
  `acados_ocp_SNMPC.json`.

---

## Phase 1 — Controller without learning (`~/Refactor`)

This phase is the critical path. Everything later depends on a working N = 38 lap without `v_ref`.
Make the changes **one step at a time** so the cause of any regression is clear.

### Step 1 — Track edges relative to the raceline
**Status:** done (2026-11-20)

Notes:
- One loader: `tracks/f1tenth.py` (`load_track`, `Raceline` with `curvature`, `edges`, `pose`,
  `cartesian`, `project`, `to_npz`/`from_npz`). `run_baseline.py` and both viewers use it.
  It keeps the geometry the controller already used (arc length from the XY polyline, smoothed
  finite-difference curvature), not the CSV's `s`/`kappa`; results are bit-identical to before.
- Silverstone x10: `ey_min` in [-20.33, -2.46] m, `ey_max` in [2.07, 19.53] m; corridor ≈ 22 m.
  In corners (|κ| > 0.02) the raceline is closer to the inner edge at 91 % of vertices (the rest are
  entries/exits). Reconstructed edges match the polylines to 1e-13 m. `scripts/check_track.py` checks
  and plots all of this.

**Why.** The Frenet frame follows the **raceline**, but F1TENTH gives track widths relative to the
**centerline**. The MPC needs the lateral distance from the raceline to each edge, as a function of `s`.

**Tasks.**
1. Merge the duplicate track code first: make `run_baseline.py` use `tracks/f1tenth.py`
   (one loader, one projection). Extend the `Raceline` dataclass instead of adding a second structure.
2. From the centerline CSV (`x, y, w_tr_right, w_tr_left`), scaled by `--scale`, build the left and right
   edge polylines in Cartesian coordinates.
3. For each raceline sample `s_i`, intersect the raceline normal with both edge polylines and store
   `ey_max(s_i)` (left, positive) and `ey_min(s_i)` (right, negative). Check the sign convention of `ey`
   in `frenet_model.py` and use the same one.
4. Store the arrays in the `Raceline` object with an interpolation function that wraps around the lap.
5. Extend `scripts/check_track.py` to plot the raceline, both edges and `ey_min/ey_max` over `s`.

**Acceptance.**
- `ey_min(s) < 0 < ey_max(s)` everywhere (the raceline lies inside the track).
- In corners the raceline is closer to the inner edge (asymmetric bounds).
- Edges reconstructed from `raceline + ey_min/max · normal` coincide visually with the edge polylines.

### Step 2 — Track-edge constraint in the OCP (keep `v_ref` for now)
**Status:** done (2026-11-20)

Notes:
- `p = [κ, ey_min, ey_max]`; `h = [a_lat, ey − (ey_max − w/2 − m), ey − (ey_min + w/2 + m)]` with
  one-sided bounds (±`ACADOS_INFTY`), at nodes 1…N and terminal (acados has no node-0 `h`).
  `w` = `veh_width` from the EDGAR config, `m` = `--track-margin` (default 0.3 m).
  Soft: L1 `1e4`, L2 `1e1` (`ConstraintOptions.track_slack_*`).
- **Solver change:** `globalization = FUNNEL_L1PEN_LINESEARCH`. With merit backtracking the soft
  constraints made the line search stall at its minimum step (mean 5 / max 97 SQP iterations at
  L1 = 1e3, worse at 1e5; `alpha_min = 0.2` helped but stayed weight-sensitive). Funnel gives
  3.0 / max 5 iterations independent of the slack weight.
- Logged: `max_slack` (diagnostics + trajectory), `track_margin_m` (trajectory), and
  `min_track_margin_m`, `track_violation_steps`, `max_slack` in the summary; overview has margin and
  slack panels.
- Acceptance run `results/Silverstone/N38/20261120-124740-595653` (10 000 steps, v_ref 5 m/s) vs
  `20261119-095044-061016` (before): RMS ey 0.003832 vs 0.003830 m, max |ey| 0.0339 vs 0.0339 m,
  0 failures; slack ≤ 5e-10; min plant margin 1.14 m (hairpin, s ≈ 786 m); 0 violations.
  Solve time mean / p95 12.9 / 14.8 ms vs 10.9 / 11.7 ms: same iteration count, ≈ 17 % more per
  iteration from the larger QP (two soft constraints per stage). Not pure noise; judged acceptable.

**Tasks.**
1. Extend the stage parameter from `p = [κ]` to `p = [κ, ey_min, ey_max]` in `frenet_model.py` /
   `ocp.py`. Keep `κ` first so existing indexing stays valid.
2. Add the nonlinear constraint at nodes 1…N (not node 0, because `x0` is fixed from the plant):
   `h(x, p) = [ey − (ey_max − w/2 − m),  (ey_min + w/2 + m) − ey] ≤ 0`
   - `w`: vehicle width from the EDGAR config; `m`: safety margin, start with 0.3 m, make it configurable.
   - Terminal node N needs the same constraint (`con_h_expr_e`).
3. Soften it with slacks: large L1 penalty (`zl`, `zu`, e.g. 1e3–1e4) and small L2 penalty (`Zl`, `Zu`).
   Put the weights in `ConstraintOptions`.
4. In `run_baseline.py`, set all three parameters per stage, at the same predicted positions as `κ`.
5. Log per step: max slack value over the horizon, and the plant's track margin
   `min(ey_max − w/2 − ey, ey − ey_min − w/2)` in `diagnostics.jsonl` / `trajectory.csv`.
6. Add the edges to `overview.png` and the offline viewer.

**Acceptance.**
- N = 38 at `v_ref = 5 m/s` gives the same results as before (RMS ey, solve times within noise).
- Slack is zero at every step; plant track margin is positive at every step.

### Step 3 — Remove `v_ref`, optimise speed
**Status:** not started

**Tasks.**
1. Stage cost: remove `(vx − v_ref)²`; add `−q_s · s_dot`, with `s_dot` the exact model expression
   `(vx cos epsi − vy sin epsi) / (1 − κ ey)`. Keep the `ey`, `epsi`, `vy`, `r`, `a` and input terms.
2. Terminal cost: remove the `(vx − v_ref)²` term from `QuadraticTerminalCost`; keep `ey` and `epsi`.
   (The learnable speed term is added in Phase 3.)
3. Remove `v_ref` from `build_ocp`'s signature, the CLI (`--speed`) and the configs, or keep it only as the
   initial-guess speed (rename to `v_init` if so).
4. **Parameter lookup:** stop using `s0 + j·dt·v_ref`. Use the predicted `s_k` of the previous solution,
   shifted by one step (append an extrapolated last stage). At the first step, use a constant-speed guess.
5. Initial guess / cold start: constant moderate speed (e.g. 10 m/s on the scaled track) along the
   raceline, `ey = epsi = 0`.
6. Keep `vx ≤ 25 m/s` (hard) and the lateral-acceleration constraint. Do not add the combined friction
   circle yet; add it only if step 4 shows it is needed (note the reason here if so).

**Acceptance.**
- The OCP builds and solves from a cold start at the lap start.
- On straights the car accelerates towards the speed limit; in corners it slows down.

### Step 4 — N = 38 full lap (benchmark)
**Status:** not started

**Tasks.**
1. Pick `q_s` by hand (a few trial values) so the car drives fast without pushing the constraints into
   frequent slack use. **No systematic tuning** (D6).
2. Run a full lap of Silverstone (4462 m on the scaled track; set `--steps` high enough).
3. Make sure `completed_laps` is counted correctly with the new unwrapping.

**Acceptance.**
- Full lap completed, 0 solver failures, 0 track violations (plant margin > 0), slack ≈ 0.
- Report: lap time, RMS ey / epsi, max |ey|, speed profile plot, solve time mean / p95 / max.
- Record the run directory and the chosen `q_s` here.

### Step 5 — N = 10 with default terminal cost
**Status:** not started

**Tasks.**
1. Same configuration as step 4 with `--N 10` (dt stays 0.08 s, so a 0.8 s horizon).
2. Record where and how it fails (position `s`, speed, slack, solver status), or the lap result if it
   completes.

**Acceptance.** Results documented here. This defines the gap RL should close.
(Confirm with the group whether this run is reported in the final comparison or only used as a diagnostic.)

### Housekeeping (any time during Phase 1)
- ~~Delete or fix `visualization/tum_adapter.py` and `tum_live.py`.~~ Done: replaced by
  `visualization/live.py` (`--live 1|2`, TUM-CONTROL Mode 1/2).
- Restore `tests/` with at least: model shape test, track-edge sanity test, one OCP solve test.
- Replace TUM's `README.md`; remove `acados_ocp_SNMPC.json`; decide on `results/` and `*.egg-info` in git.

---

## Phase 2 — Leap-C setup

### Step 6 — Install and smoke-test the RL stack
**Status:** not started

**Tasks.** Follow `LEAPC_SUMMARY.md` §9: clone `leap-c/leap-c-lab` and `leap-c/mpc-sac`, install with
`--no-deps` into the `latch13` env, install `gymnasium==1.1.1`, `tensordict`, `tensorboard`.
Run `pytest ~/mpc-sac/tests` and train SAC-ZOP briefly on the bundled mass-spring-damper example.

**Acceptance.** `python -c "import leap_c, leapc_lab, mpc_sac"` works; the example trains and logs.
Record the commit hashes of `leap-c`, `leap-c-lab` and `mpc-sac` here and pin them.

Note: `~/leap-c` is shared with the old `~/LATCH` project; do not edit it in place.

### Step 7 — Problem specification
**Status:** not started

Write `docs/PROBLEM_SPEC.md` and get it agreed by the group. It must fix:

1. **Terminal cost form and θ.** Proposal:
   `V_f(x_N) = exp(θ_ey)·ey_N² + exp(θ_epsi)·epsi_N² + exp(θ_v)·max(0, vx_N − v_θ)²`,
   θ = `[θ_ey, θ_epsi, θ_v, v_θ]`. Log space for the weights because they may span orders of magnitude.
   Specify bounds (`gym.spaces.Box`) and the initial value. The `max(0, ·)²` term is a smooth-enough
   penalty on terminal overspeed; it encodes "how much braking is still needed after the horizon".
2. **Observation.** Frenet state (8) + road preview **beyond** the horizon (curvature and edge
   distances sampled every Δ m for L m ahead; propose Δ = 10 m, L = 200–300 m). Finite bounds are
   required for the `scaling` extractor.
3. **Reward per env step (0.08 s).**
   `r = w_p·Δs/(v_max·Δt) − (w_ey·ey² + w_epsi·epsi²)·Δt − w_c·violation`.
   Must be consistent with the performance measure used by Optuna (D6).
4. **Termination:** plant outside the track edges, `|epsi| > π/2`, non-finite state, MPC failure after
   retry — each with a penalty. **Truncation:** time limit or end of the episode's track section.
   Truncation must not be marked as termination.
5. **Track split:** training, validation and held-out test sections along `s`; episode start states
   sampled in training sections with small perturbations of `ey`, `epsi`, `vx`.
6. **Evaluation protocol** (see step 15).

**Acceptance.** Spec agreed and committed; open questions listed in the spec are resolved.

---

## Phase 3 — RL implementation

### Step 8 — Leap-C parameterised OCP
**Status:** not started

**Tasks.**
1. New builder (e.g. `controllers/leapc_ocp.py`) using `AcadosParameterManager`:
   - non-differentiable per-stage: `[κ, ey_min, ey_max]`;
   - differentiable global: θ from step 7.
2. Reuse `build_frenet_model` and the Phase 1 cost and constraints; only the terminal cost reads θ.
3. Solver settings that worked in the old Leap-C integration: `integrator_type="DISCRETE"` with an
   explicit RK4 map, SQP, exact Hessian with `MIRROR` regularisation, `globalization_alpha_min=0.2`,
   `nlp_solver_tol_stat=1e-5`. Use a stable `export_directory` per horizon.
4. Wrap in `AcadosDiffMpcTorch`. Always build parameters via the manager (stage-varying indicator).

**Acceptance.** `dV/dθ` and `du0/dθ` match central finite differences on a few states (rel. error < 1e-3).

### Step 9 — Fixed-θ reproduction
**Status:** not started

Run the Leap-C OCP in the Phase 1 closed loop with θ fixed to the default terminal cost.

**Acceptance.** N = 38 and N = 10 results match steps 4 and 5 (same lap result, metrics within noise).

### Step 10 — `RacingEnv`
**Status:** not started

**Tasks.** `src/latch13/simulation/env.py`, a `gymnasium.Env`:
- `step(action)`: 4 TUM plant substeps of 0.02 s (jerk integrated to acceleration as in
  `run_baseline.py`), project to Frenet, unwrap `s`, compute reward/termination from the spec.
- `reset(seed, options={"mode": "train"|"val"|"test"})`: sample start state in the right section.
- `info["task"]` with violation and success flags (logged by `mpc-sac`).
- Move the closed-loop code out of `run_baseline.py` into shared functions; `run_baseline.py` and the
  env must call the same code (same plant, projection, track, termination).

**Acceptance.** A rollout with the fixed-θ MPC through the env gives the same trajectory as
`run_baseline.py` from the same start state.

### Step 11 — Planner and training script
**Status:** not started

**Tasks.**
- `controllers/planner.py`: `ParameterizedPlanner` wrapping `AcadosDiffMpcTorch`; `param_space` as a
  `gym.spaces.Dict` of θ bounds; `default_param` = θ initial value; builds per-stage parameters from the
  observation. Template: `leapc_lab/mass_spring_damper/planner.py` (see `LEAPC_SUMMARY.md` §7.3).
- A good warm-start initializer (constant-speed guess along the raceline), not the zero iterate.
- `scripts/train_rl.py`: config, env, planner, `ControllerFromPlanner`, `SacZopTrainer`, run, save θ /
  policy network. Use the `scaling` extractor.

**Acceptance.** A short training run (a few thousand steps) runs end to end without crashes, logs
reward, `success_rate` and solver statistics, and saves checkpoints.

---

## Phase 4 — Training

**Prerequisite:** the stage cost weights from Optuna are frozen and committed (D5, D6).
If they change after training, the trained θ is invalid.

### Step 12 — SAC-ZOP, constant θ
**Status:** not started
Observation without road preview, so θ is effectively global. First RL result.

### Step 13 — SAC-ZOP, road-dependent θ (main result)
**Status:** not started
Observation with road preview beyond the horizon; the network outputs θ per step.

### Step 14 — SAC-FOP
**Status:** not started
Same setup as step 13 with gradients through the MPC (`fop`, optionally `foa`). Compare
sample efficiency and final performance with ZOP. Watch `masked_samples_perc` and `zero_dudp_perc`.

For all training steps record: seed, commits (`latch13`, `leap-c`, `leap-c-lab`, `mpc-sac`), config,
training curves, best validation checkpoint.

---

## Phase 5 — Evaluation and report

### Step 15 — Comparison
**Status:** not started

Controllers: N = 10 learned (steps 12–14), N = 38 benchmark, and N = 10 default terminal cost (see
step 5 note). Evaluate on the held-out test section and on full laps, several seeds, mean ± std.

Metrics: lap time / distance reached, RMS and max ey, RMS epsi, track violations (count, worst margin),
slack usage, solver failures, solve time mean / p95 / max.

Evaluation uses the same plant and env code as training, with the learned θ / network loaded and no
learning. For constant θ, a plain acados solver with θ as parameters is enough; for road-dependent θ,
the policy network runs every step.

### Step 16 — Report
**Status:** not started
Method, setup, results, limitations (e.g. plant/model mismatch, single track, compute cost).

---

## Pitfalls to remember

- Leap-C sensitivities are only valid at converged solves; mask or penalise failed solves.
- Never apply a NaN action; terminate with a penalty instead.
- Anything RL changes must be a manager parameter; anything baked in at code generation needs a rebuild.
- `mpc-sac` installs `leap-c` from GitHub `main`; pin commits and install with `--no-deps`.
- Windows/WSL line endings: `core.autocrlf input` is set in the old repo; check new files are LF.
