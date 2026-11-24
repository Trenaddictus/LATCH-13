# ROADMAP: Learned terminal cost for short-horizon MPC (SSY226, group 13)

Status as of 2026-11-23. Repo: `~/Refactor`, branch `feature/track-edges`
(remote `git@github.com:Trenaddictus/LATCH-13.git`). Platform: WSL2 Ubuntu 22.04, Python 3.11, uv venv `~/Refactor/.venv`.

This file is the working plan for Claude Code. Read it together with:
- `docs/PROBLEM_SPEC.md`: the RL problem (θ, observation, reward, episode ending, split), draft for the group
- `STEP3_SOLVER_ISSUES.md`: solver findings behind the MPC settings (see the correction in Step 3)
- `external/leap-c/LEAPC_SUMMARY.md`: Leap-C, `leap-c-lab` and `mpc-sac` APIs, the RL algorithms, and sketches
  for env/planner/training script (written for the old integration)
- `REFACTOR_SUMMARY.md`: the `latch13` code before Step 1 (outdated; this file has the current state)

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
| D6 | **Stage cost weights are tuned with Optuna.** Originally by teammates; since 2026-11-24 done here (David): `scripts/tune_stage_weights.py`, separately for N = 38 and N = 10, see "Stage-weight tuning" before Phase 4. |
| D7 | **Build on `latch13` modules.** Import model, track and plant code from the package; do not copy them into scripts. |

---

## 1. Current state (2026-11-23)

- **Phase 1 is done.** The Frenet MPC (`controllers/ocp.py`) rewards progress (`−q_s·ṡ`, `q_s = 1`, no
  `v_ref`). Stage parameters are `p = [κ, ey_min, ey_max]`.
- **Constraints:**
  - soft track edges (margin 0.3 m) and soft lateral acceleration (5.886 m/s²);
  - hard speed 2–25 m/s, actuator and input bounds.
- **Solver:** SQP with exact Hessian + MIRROR, funnel line search, `tol_stat = 1e-3`, 1000 iterations.
- **Default terminal cost** (`QuadraticTerminalCost`) = the initial RL P since 2026-11-23, including a
  terminal `vx` weight of 0.01 (see Step 5).
- **Default stage weights** are tuned with Optuna per horizon (`TUNED_STAGE_WEIGHTS`, see "Stage-weight
  tuning"), **frozen on 2026-11-24**. `--stage-weights hand-picked` gives the old ones.
- **N = 38** drives 4 laps with 0 failures (flying laps 192.24–192.32 s).
- **N = 10** drives 3 laps with 0 failures, cruising at about 22 m/s (laps 225.9 / 225.4 / 224.2 s; with
  the hand-picked weights 238.8 / 236.1 / 236.0 s). It used to fail at s ≈ 745–775 m, before the default
  had a `vx` term.
- **Leap-C:**
  - `controllers/leapc_ocp.py`: the same MPC with a learnable diagonal terminal cost, the supervisor's
    form `x̃ᵀ diag(exp θ) x̃` over `x̃ = [ey, epsi, vx, vy, r, δ, a]` (7 θ, since 2026-11-23);
  - `controllers/leapc_fixed.py`: drives it in the runner (`--controller leapc|leapc-unshifted`).
- **Simulation:**
  - `simulation/plant.py`, `closed_loop.py`: shared by the runner and the RL env;
  - `simulation/env.py`: the RL environment.
- **Checks:** `scripts/check_track.py`, `check_sensitivities.py`, `check_env.py`, `check_planner.py`.
- **RL:** `controllers/planner.py` and `scripts/train_rl.py` (SAC-ZOP) run end to end (Step 11).
- **Next:** Step 12 (first training run). The stage weights are frozen (2026-11-24); it still waits for
  the `vx` weight bound and the other group decisions in `docs/PROBLEM_SPEC.md` §10.
- **Nothing since commit `a4c6631`** (Steps 1–2) is committed.

Starting point before Step 1 (2026-11-20), for reference:

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
**Status:** done (2026-11-20), corrected 2026-11-22

Correction (found in Step 10):
- **Interpolated edges:** between the vertices `edges(s)` was up to 0.62 m beyond the real edge
  (Silverstone, hairpin at s = 787 m). `load_track` now narrows the offsets there: at most 4 mm beyond
  between the vertices, 0.075 m at the edge's own corners.
- **False loops:** offsetting the centerline made small loops in tight corners, up to 0.43 m into the
  track. They are removed.
- **Cost:** loading takes about 2 s. `check_track.py` now fails if the interpolation passes an edge.
- **All 20 tracks load** at x10 and x1. Three are unsuitable as a second track at x10 with this margin
  (body + margin needs 1.27 m): YasMarina's raceline crosses an edge (loading warns), Hockenheim's
  comes within 0.11 m of one and Austin's within 0.95 m.
- The notes below describe the original version.

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
**Status:** done (2026-11-20). Details of the solver work: `STEP3_SOLVER_ISSUES.md`.

**Correction (2026-11-22).** The "stationarity floor of about 1e-5" below, and the same explanation in
`STEP3_SOLVER_ISSUES.md`, are wrong. The failures were mostly the 100-iteration limit.
- **Current settings:** `nlp_solver_tol_stat = 1e-3`, `nlp_solver_max_iter = 1000`, `speed_min = 2`
  (all set by David).
- **The 1000 iterations:** solver failures in off-raceline tests dropped from 4.7 % to 0, and N = 38 also
  finishes 4 laps at speed limits of 30 and 37.5 m/s. The cost is longer solves in hard cases.
- **The speed floor:** the MPC's integrator is unusable below about 1.86 m/s (the plant is fine down to
  about 0.5 m/s), so the floor protects the MPC.

Notes:
- Stage cost `−q_s·s_dot` (exact model expression, `q_s = 1`) replaces `(vx − v_ref)²`; no speed term in the
  terminal cost; `v_ref` removed from `build_ocp`. CLI: `--v-init` (start / cold-start speed, 10 m/s) and
  `--progress-weight`. Stage parameters come from the previous solution's predicted `s` (shifted one
  stage, anchored at the measured `s`); constant-speed cold-start guess along the raceline.
- Needed for SQP convergence:
  - `regularize_method = MIRROR`: the progress reward is not convex; without it the cold start fails.
  - Terminal cost `+ 10·(vy² + r² + δ²)`: the lateral states at the end of the horizon were a nearly flat
    direction, and the SQP drifted there.
  - `qp_solver_tol_* = 1e-8`, `qp_solver_iter_max = 200`, `nlp_solver_tol_stat = 1e-4`: the SQP hits a
    stationarity floor of about 1e-5 (lowered by accurate QPs), so 1e-5 caused iteration-limit failures.
- Acceptance run `results/Silverstone/N38/20261120-193956-822073` (N = 38, 3500 steps = 1.4 laps):
  lap time 194.5 s, mean / max speed 22.7 / 25 m/s, 0 solver failures, 0 track violations (closest
  approach to an edge 1.43 m), max |ey| 0.36 m, RMS ey 0.07 m, slack ≈ 1e-12.
  Accelerates to 25 m/s on straights, brakes to 11–20 m/s in corners.
- Open (not blocking Step 4): solves take 6.1 SQP iterations on average, but 193 of 3500 need 20 or more
  (max 62). Solve time mean / p95 / max is 27 / 85 / 249 ms, and 198 solves exceed the 80 ms MPC step,
  all in braking zones. Fine for simulation, not real-time. Whether `tol_stat = 1e-4` is accurate
  enough for Leap-C sensitivities is to be checked in Step 8.

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
**Status:** done (2026-11-21), rerun 2026-11-22. Chosen `q_s = 1` (the default of `--progress-weight`).

**Current result with the tuned N = 38 stage weights (2026-11-24):** laps 194.32 / 192.32 / 192.24 /
192.32 s, 0 failures, RMS ey 0.066 m, closest approach 0.82 m ("Stage-weight tuning").

**Result with the hand-picked stage weights** (2026-11-23: default terminal cost = initial RL P with `vx`,
`a` weights 0.01; speed_min 2, 1000 SQP iterations, corrected edges and exit check; 10 000 steps):
- 0 solver failures, never off the track; closest approach 0.78 m; max slack 5e-4.
- Laps 194.48 / 192.32 / 192.64 / 192.48 s; RMS ey 0.073 m, max |ey| 0.52 m. Still 25 m/s on straights.
- Before the `vx` term (2026-11-22): laps 194.64 / 192.48 / 192.40 / 192.64 s, RMS ey 0.070 m, closest
  approach 0.83 m.
- The numbers further down are from earlier settings and the old exit check.

Hand trials of `q_s` (first lap from the standing start, N = 38, otherwise the settings below; runs in
`results/Silverstone/N38/qs_trials/`):

| `q_s` | Result | Lap 1 | RMS / max ey | Mean SQP iterations | Solves ≥ 20 iterations |
|---|---|---|---|---|---|
| 0.5 | full lap | 195.20 s | 0.059 / 0.33 m | 3.72 | 1 |
| **1** | full lap | 194.64 s | 0.067 / 0.36 m | 4.40 | 34 |
| 2 | full lap | 194.32 s | 0.087 / 0.46 m | 5.43 | 102 |
| 5 | solver failure at s = 1416 m | — | 0.164 / 0.85 m | 5.60 | 40 |

- Lap time barely depends on `q_s` (0.9 s between 0.5 and 2), because speed is limited by the
  constraints.
- Higher `q_s` worsens tracking and solver convergence, and `q_s = 5` fails.
- Slack stays at most 1.5e-5 and the car body at least 1.42 m from an edge in all trials.
- `q_s = 1` was kept as the middle value, already benchmarked over 4 laps.
- `q_s = 0.5` would be about 0.3 % slower but more robust for the solver; an option if solver load
  matters in Phase 3.

Final settings (after Step 3): soft lateral acceleration (L1 1e3, L2 10), `nlp_solver_tol_stat = 1e-3`,
QP tolerances 1e-8. With a hard lateral limit, the SQP's linearised constraint made the QP infeasible
near the limit (QP failure on lap 2); with `tol_stat = 1e-4` it stalled just above the tolerance on lap 2.
See `STEP3_SOLVER_ISSUES.md` §8.

Benchmark run `results/Silverstone/N38/20261120-202857-508420` (`q_s = 1`, 10 000 steps, 4.15 laps):
- **0 solver failures**, 0 track violations; closest approach to an edge 1.37 m; max slack 0.004.
- **Lap times: 194.64 s** (lap 1, standing start at 10 m/s), **192.40 s / 192.48 s** (flying laps 2 / 3).
- Per lap: RMS ey 0.063–0.074 m, max |ey| 0.36–0.43 m, mean speed 22.9–23.2 m/s.
  Speed: 25 m/s on straights, 11–20 m/s in corners.
- SQP iterations mean 4.4, max 50. Solve time mean / p95 / max 21 / 56 / 215 ms;
  1.7 % of solves exceed the 80 ms MPC step.
- `completed_laps` and `lap_time_s` (first lap) come from the unwrapped progress.

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
**Status:** done for `q_s = 1` (2026-11-20), rerun 2026-11-22. Rerun if Step 4 picks a different `q_s`.
Group decision: this failure is reported as the N = 10 result (no workaround for non-converged solves).

**Current result with the tuned N = 10 stage weights (2026-11-24),** 10 000 steps, 0 failures each:
acados 225.92 / 225.36 / 224.16 s, Leap-C shifted 226.40 / 225.60 / 226.40 s, Leap-C unshifted
223.92 / 222.24 / 221.92 s; RMS ey 0.045–0.051 m, max |ey| ≤ 0.85 m ("Stage-weight tuning").

**Result with the hand-picked stage weights (2026-11-23), changed by the new default terminal cost.**
With the terminal `vx` and `a` weights at 0.01 (same settings as Step 4's current result), **N = 10 no
longer fails**:

| Run (4000 steps unless noted) | Result | Lap 1 | Max speed | Closest to an edge |
|---|---|---|---|---|
| acados, v-init 10, 10 000 steps | 3 laps, 0 failures | 238.80 s (then 236.08, 236.00) | 20.6 m/s | 0.57 m |
| acados, v-init 12 | 0 failures | 237.52 s | 20.6 m/s | 0.59 m |
| acados, v-init 15 | 0 failures | 236.48 s | 20.6 m/s | 0.56 m |
| Leap-C shifted, v-init 10 | 0 failures | 238.40 s | 20.6 m/s | 0.58 m |
| Leap-C unshifted, v-init 10 | 0 failures | 238.88 s | 20.6 m/s | 0.59 m |

- **The gap RL should close is now lap time:** about 236 s against N = 38's 192.4 s (23 % slower). The
  `vx` weight holds N = 10 near 20 m/s, which is slow but safe.
- **Before (2026-11-22, no `vx` term):** no lap. Left the track at step 813 / 586 (v-init 10 / 12), or
  solver failure at step 714 / 723 (v-init 15 / Leap-C unshifted), all at s ≈ 745–775 m.
- **For Step 15:** "N = 10 with the default terminal cost" now means this safe, slow controller. If the
  failing variant without the `vx` term is also wanted as a baseline, it needs its own option.

The notes below are from earlier settings and the old exit check.

Notes (run `results/Silverstone/N10/20261120-200930-506250`; repeated with the final Step 4 settings in
`results/Silverstone/N10/20261120-211703-228323`, with the same result: step 397, s = 705 m, residual 3.9e-3):
- **Result: no lap.** `solver_status_2` at step 397, s = 705 m (of 4462 m), after 31.8 s.
- Up to s ≈ 630 m it drives like N = 38 (25 m/s, |ey| < 0.05 m).
- **Late braking** for the corner sequence at s ≈ 650–700 m (radius 48 → 32 m): N = 10 starts braking at
  s = 642 m, N = 38 at s = 623 m. The 0.8 s horizon covers only about 20 m at 25 m/s.
  N = 10 carries 3–4 m/s more speed into the corner (speed / ey: s = 675 m 19.3 / +2.2 vs N = 38
  15.2 / +0.3; s = 690 m 15.3 / +6.5 vs 13.0 / +0.3).
- **It runs 10 m wide** (ey = +10.0 at s = 705 m, 9.4 m/s). It stays on the track: 4.6 m to the edge at
  the failure, minimum 1.61 m, 0 violations, slack ≈ 1e-12.
- **The solver fails during the recovery.** Both attempts (warm start and cold restart) reach the same
  sensible plan: brake to 7.5 m/s, steer back toward the raceline, lateral acceleration at 5.89 m/s² at
  every node. But stationarity sticks at 3.7e-3 to 5e-3, so both stop at the iteration limit. The
  steps before already needed 42–67 iterations. This is the same weakness as in Step 3
  (`STEP3_SOLVER_ISSUES.md`), in a more extreme state.
- Over the driven part: RMS ey 2.15 m, max |ey| 10.06 m, RMS epsi 0.071 rad. Solve time mean / p95 / max
  5.5 / 8.2 / 77.6 ms, about 5× faster than N = 38.
- Diagnostics, not adopted:
  - soft lateral acceleration: same failure at s = 703 m, slack 0.01;
  - applying failed solves and continuing: the car slows to about 1 m/s, weaves, and grazes an edge at
    s = 827 m; many failed solves are QP failures with garbage plans.
- Change made for this step: the cold-start guess now starts from the measured state (speed, ey, epsi,
  vy, r, δ, a) and blends the deviations back to the raceline with a 1 s time constant. It replaces the
  constant-speed guess on the raceline, which gave garbage restarts (residual 1e4) far off the line.
  At the lap start the two are the same.

**Tasks.**
1. Same configuration as step 4 with `--N 10` (dt stays 0.08 s, so a 0.8 s horizon).
2. Record where and how it fails (position `s`, speed, slack, solver status), or the lap result if it
   completes.

**Acceptance.** Results documented here. This defines the gap RL should close.
(Confirm with the group whether this run is reported in the final comparison or only used as a diagnostic.)

### Housekeeping (any time during Phase 1)
- ~~Delete or fix `visualization/tum_adapter.py` and `tum_live.py`.~~ Done: replaced by
  `visualization/live.py` (`--live 1|2`, TUM-CONTROL Mode 1/2).
- ~~Merge the duplicate track code in `run_baseline.py` and `tracks/f1tenth.py`.~~ Done in Step 1.
- Restore `tests/` with at least: model shape test, track-edge sanity test, one OCP solve test.
  Partly covered by the check scripts (`check_track.py`, `check_sensitivities.py`, `check_env.py`), which
  are not pytest tests.
- Replace TUM's `README.md`; remove `acados_ocp_SNMPC.json`; decide on `results/` and `*.egg-info` in git.

---

## Phase 2 — Leap-C setup

### Step 6 — Install and smoke-test the RL stack
**Status:** done (2026-11-21)

**What was done** (this differs from the original plan, which assumed a Conda env `latch13` and a shared
`~/leap-c`):
- **Environment:** the uv venv `~/Refactor/.venv`. Install packages with `uv pip install` (the venv has
  no pip). acados comes from Leap-C's build, `external/leap-c/external/acados` (`ACADOS_SOURCE_DIR` is set
  in `.venv/bin/activate`).
- **Leap-C repos as git submodules, pinned:**

  | Package | Path | Commit |
  |---|---|---|
  | `leap-c` | `external/leap-c` | `217e1c8` |
  | `leap-c-lab` | `external/leap-c-lab` | `1e69692` |
  | `mpc-sac` | `external/mpc-sac` | `d38726d` |

  `leap-c-lab` and `mpc-sac` are installed editable with `--no-deps`.
- **Dependencies installed:**
  - `torch 2.15.0+cpu` (CPU build; `mpc-sac` defaults to the CPU and the MPC runs on the CPU);
  - `gymnasium 1.1.1`, `tensordict 0.15.6`, `tensorboard 2.19.0`, `pytest`.
- **Deliberately not pinned:** `scipy 1.17.1` and `PyYAML 6.0.3` were kept, against the package pins
  1.15.2 and 6.0.2. All tests pass with them.

**Checks:**
- `import leap_c, leapc_lab, mpc_sac` and `from leap_c.torch import AcadosDiffMpcTorch` work.
- `pytest external/mpc-sac/tests`: **88 passed** (98 s), including short SAC-ZOP runs on cartpole, chain,
  mass-spring-damper and pointmass.
- **Smoke training:** SAC-ZOP on mass-spring-damper for 20 000 steps (validation every 5000).
  - Ran to completion. It wrote CSV and TensorBoard logs, the config, `git.txt` and checkpoints.
  - Validation score per episode −4.2 to −5.7 (too short to show learning; the script default is
    200 000 steps). No violations; MPC inference about 0.9 ms per step.

### Step 7 — Problem specification
**Status:** in progress (2026-11-21): draft in `docs/PROBLEM_SPEC.md`, waiting for the group decisions listed in its §10.
New for the group (2026-11-22): the proposed split and validation score of Step 10.
**Decided 2026-11-24 (David), spec §10:**
- reward weights as in spec §5;
- failure penalty P = 100 (was 10);
- time limit 30 s;
- track split of spec §7;
- validation/test score with the time bonus;
- θ bounds with `vx` [0.001, 0.05];
- a terminal `vx²` weight is acceptable under D2.

Still open: the observation details, the start distribution and the evaluation protocol.
**θ changed 2026-11-23** to the supervisor's form (spec §3): `V_f = x̃ᵀ diag(exp θ) x̃`,
`x̃ = [ey, epsi, vx, vy, r, δ, a]`, 7 log-weights, initial P = diag(40, 20, 0.01, 10, 10, 10, 0.01),
bounds [1e-4, 1000] (the supervisor's 0.01 lower bound would put two initial values on the bound).
The overspeed term and `v_threshold` are gone. Task 1 below is the original proposal.
**`vx` bound (measured 2026-11-23, spec §3):** the `vx` weight acts as a cruise speed setting (0.01 → 20.5 m/s,
0.02 → 10.8 m/s), and from 0.1 the car stops. Proposal for the group: [0.001, 0.05] for that weight.
**Applied 2026-11-24** (David): `DiagonalTerminalTheta` has per-weight bounds, [0.001, 0.05] for `vx` and
[1e-4, 1000] for the other six.

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
**Status:** done (2026-11-22); rechecked 2026-11-23 for the 7-weight θ. Again 2 gradient exceptions of
the same two kinds, plus 2 of 1440 reference solves that did not converge (below). Reviewed and accepted
by David (2026-11-23).

**Recheck with the 7-weight θ (2026-11-23):**
- **Equivalence:** at `phase1_theta`, u0 matches the Phase 1 acados MPC to 7.7e-11 (median 1e-13) and the
  plan to 3.3e-10, on 163 recorded states (N = 10 and 38), all converged.
- **Gradient check, 6 seeds:**

  | Seeds | Conclusive | Failed (rel. error >= 1e-3) | Base solves not converged |
  |---|---|---|---|
  | 1, 2, 3 | 2146 / 2160 | 0 | 1 (seed 2) |
  | 4, 5, 6 | 2133 / 2160 | 2 (seed 5) | 1 (seed 4) |

- **The two failures, both at the initial θ:**
  - #16: steering rate pinned at its limit, true gradient 0, reference noise about 1e-6.
  - #44: steering-rate limit only just active, so the derivative jumps. The one-sided values are about
    3e-4 and 1e-3 per unit of log-weight, too small to matter.
- **Non-converged solves:** at the tight gradient tolerance a few cold solves fail (0–4 of 480 in a
  separate test), independent of the `vy, r, δ` weights. At the driving tolerance 0 of 1440 failed.
- All dV/dtheta checks pass. The script reports FAIL for seeds 2, 4 and 5 because of these four cases.

Notes (written for the earlier 4-parameter θ):
- `controllers/leapc_ocp.py`: diagonal terminal P (spec §3, stage A), then theta = `log_weights` (3) +
  `v_threshold` (1), both global and differentiable; `track` = [kappa, ey_min, ey_max] per stage. Reuses
  `build_ocp`; `ocp.py` and `run_baseline.py` are unchanged. Since 2026-11-23: `log_weights` (7) only.
- **Settings differ from task 3:** DISCRETE RK4 (4 substeps, required for sensitivities) with the Phase 1
  solver settings (funnel line search, MIRROR), not the old `alpha_min` settings. Leap-C generates the
  exact-Hessian backward solver itself.
- **Equivalence:** with the speed term off, u0 and cost match Phase 1 to 1e-9 or better (median ~1e-13)
  on 159 recorded states (N = 10 and 38). Solves are faster: about 1.6× on the closed-loop
  timer (medians; acados' own timer has spurious 2–3 s entries).
- **Gradient check:** `python scripts/check_sensitivities.py --seed K` (240 problems, 720 checks per
  seed, N = 10, tight preset, through torch autograd).

  | Seeds | Role | Conclusive | Failed (rel. error >= 1e-3) |
  |---|---|---|---|
  | 1, 2, 3 | used to develop the test | 2159 / 2160 | 0 |
  | 4, 5, 6 | held out | 2151 / 2160 | 2 |

  The two failures: seed 5 #40 (jerk pinned at its bound, true gradient ~0, reference noise 1e-6) and
  seed 6 #224 (steering-rate limit only just active, so nearly non-differentiable; error 0.4 %).
  All dV/dtheta checks pass. The test's reference rules were corrected during development; the 1e-3
  threshold was not.
- **SAC-FOP must use `SENSITIVITY_SOLVER_OPTIONS`.** At the driving tolerance (1e-3), 36 % of nonzero
  du0/dtheta miss 1e-3 and 4 % are off by more than 10 %. The preset costs ~2.2x solve time.
  SAC-ZOP can keep the driving tolerance.
- **Vanishing gradients:** du0/dtheta is zero in 66 % of sampled problems (u0 pinned at a limit). At the
  initial theta the speed parameters move u0 in only 10 of 180 samples, so FOP gets almost no gradient
  towards "brake earlier" at the start. This supports ZOP first (D4).
- **Several local optima** exist, so warm starts can change which solution, and which gradient, is found.
- Gradients are untested at N = 38 (not learned) and on states far from the raceline.


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
**Status:** done (2026-11-22)

**With the tuned stage weights (2026-11-24),** N = 38, 10 000 steps, 0 failures in all three: laps
194.32 / 192.32 / 192.24 / 192.32 s (acados), 194.32 / 192.40 / 192.32 / 192.32 s (Leap-C shifted),
194.40 / 192.32 / 192.24 / 192.32 s (unshifted). N = 10: see Step 5.

**With the hand-picked stage weights** (rerun 2026-11-23 with the new default terminal cost, which
`phase1_theta` now reproduces and which equals the initial RL θ; N = 38, 10 000 steps, 0 failures in
all three):

| | acados | Leap-C, shifted | Leap-C, unshifted |
|---|---|---|---|
| Laps 1 / 2 / 3 / 4 [s] | 194.48 / 192.32 / 192.64 / 192.48 | 194.48 / 192.32 / 192.24 / 192.40 | 194.32 / 192.48 / 192.40 / 192.48 |
| RMS ey / max ey [m] | 0.0730 / 0.520 | 0.0783 / 0.526 | 0.0740 / 0.523 |
| Closest to an edge [m] | 0.782 | 0.855 | 0.787 |

- **Equivalence:** u0 matches the acados MPC to 5.6e-11 (163 recorded states, N = 10 and 38).
- **N = 10, same three controllers:** 0 failures; lap 1 238.80 / 238.40 / 238.88 s (Step 5).
- **Earlier today, before the `vx` term was in the default** (7-weight θ at the Phase 1 weights): N = 38
  laps 194.64 / 192.48 / 192.40 / 192.64 s (acados); N = 10 left the track at step 813 or failed at 723.

The table further down is from earlier settings.

Run: `python scripts/run_baseline.py --controller leapc --N 38 --steps 10000` (also `--N 10`;
`--controller leapc-unshifted` uses the warm start of RL training). Runs in
`results/Silverstone/N{10,38}/step9/`. The default `--controller acados` is bit-identical to the
Step 4 / 5 runs.

| | acados | Leap-C, shifted | Leap-C, unshifted |
|---|---|---|---|
| N = 38: laps 1 / 2 / 3 [s] | 194.64 / 192.40 / 192.48 | 194.64 / 192.48 / 192.40 | 194.56 / 192.40 / 192.56 |
| N = 38: failures in 10 000 steps | 0 | 0 | 0 |
| N = 38: RMS ey / max ey [m] | 0.0690 / 0.426 | 0.0726 / 0.455 | 0.0715 / 0.437 |
| N = 38: solve mean / p95 [ms] | 21.2 / 52.1 | 11.4 / 28.7 | 10.2 / 26.4 |
| N = 10: failure | step 397, s = 705 m | step 397, s = 705 m | step 395, s = 703 m |
| N = 10: solve mean / p95 [ms] | 4.4 / 6.2 | 2.9 / 4.5 | 2.4 / 5.6 |

- **The only difference is the warm start:** Leap-C sets node 0 of the guess to the measured state. With
  that one change the acados controller reproduces Leap-C to 1e-11 (N = 10) and 5e-6 over 10 000 steps
  (N = 38).
- **"Within noise" was measured:** the acados controller from 15 start speeds within 10 ± 0.002 m/s (all 15
  drove 4 laps without failure) spans RMS ey 0.0676-0.0733 m, max ey 0.395-0.465 m, flying laps
  192.32-192.64 s. All eight compared Leap-C metrics lie inside, for both warm starts.
- A 5 % RMS threshold guessed before the runs was missed (N = 38 shifted +5.2 %; N = 10 unshifted
  -5.4 %, from stopping 2 steps earlier). It was narrower than the benchmark's own spread.
- **For Step 15:** one run is not a benchmark. RMS ey varies by about 8 % between indistinguishable
  starts, so report mean ± std over several starts for N = 38 too.


Run the Leap-C OCP in the Phase 1 closed loop with θ fixed to the default terminal cost.

**Acceptance.** N = 38 and N = 10 results match steps 4 and 5 (same lap result, metrics within noise).

### Step 10 — `RacingEnv`
**Status:** done (2026-11-22). Reward, penalty, time limit, split, score and start distribution are
placeholders until the group decides them.

Run the acceptance test: `python scripts/check_env.py` (`--steps-long 10000` for four N = 38 laps;
28 checks).

What was built:
- `simulation/plant.py`, `simulation/closed_loop.py`: moved out of `run_baseline.py`. The move alone left
  11 reference runs identical in every value except timings.
- `Raceline.body_margin`: exit check on the real edge polylines, with heading and body length. The
  runner and the env both use it. `track_margin_m` in older runs is not comparable.
- `simulation/env.py`: the environment. It differs from the original tasks below in four ways:
  - the split is a constructor argument (mpc-sac passes no mode on validation resets);
  - an MPC failure must arrive as a non-finite action (the trainer passes nothing else to `step`);
  - the road in the horizon is given at the raceline's vertices, which is exact (a 1 m grid still had
    up to 0.18 m edge error);
  - episodes end below 3 m/s. This is in the env only, not in the runner.

Fixed before Step 11 (2026-11-22):
- **Track edges:** see the correction in Step 1.
- **Split (proposed, not yet agreed):** validation is corners 12+13 (3930–4430 m), test stays corners
  5+6 (1300–1850 m), training is 1850–3930 m and 4430–1095 m over the lap start.
- **Preview:** training ends before its 200 m preview reaches a held-out corner. It still sees the
  straight before corners 12+13 (radius above 200 m).
- **Score (proposed, not yet agreed):** in val and test, finishing a section adds the time left before
  the 30 s limit as progress at full speed. Without it the summed reward barely sees time. The training
  reward is unchanged.

Default controllers from 50 random starts per split (same starts for both).

With the tuned stage weights (2026-11-24):

| Split | N = 38 failures | N = 10 failures | N = 38 score | N = 10 score |
|---|---|---|---|---|
| train | 0 | 0 | 307.2 | 274.6 |
| val | 0 | 0 | 337.5 | 292.2 |
| test | 0 | 0 | 307.3 | 262.0 (19 of 50 hit the 30 s limit) |

With the hand-picked stage weights (2026-11-23, `vx` term in the default):

| Split | N = 38 failures | N = 10 failures | N = 38 score | N = 10 score |
|---|---|---|---|---|
| train | 0 | 0 | 307.2 | 261.4 |
| val | 0 | 0 | 336.7 | 280.7 |
| test | 0 | 0 | 306.9 | 248.0 (29 of 50 hit the 30 s limit) |

- **Validation still separates them**, now through the time bonus: N = 10 scores 17 % lower because it
  cruises at about 20 m/s.
- **Before the `vx` term** (2026-11-22), N = 10 failed in 13 / 24 / 34 of 50 starts and scored 259.3 /
  202.1 / 171.1.

Still open:
- The MPC's edge constraint ignores heading and body length (the exit check does not).
- SAC-FOP: non-finite actions must not reach its buffer, and it drops failed solves from updates (Step 14).

**Tasks** (original plan; where the implementation differs, the notes above apply).
`src/latch13/simulation/env.py`, a `gymnasium.Env`:
- `step(action)`: 4 TUM plant substeps of 0.02 s (jerk integrated to acceleration as in
  `run_baseline.py`), project to Frenet, unwrap `s`, compute reward/termination from the spec.
- ~~`reset(seed, options={"mode": "train"|"val"|"test"})`~~: the split is a constructor argument instead
  (mpc-sac passes no mode on validation resets); `reset` samples the start state in that split.
- `info["task"]` with violation and success flags (logged by `mpc-sac`).
- Move the closed-loop code out of `run_baseline.py` into shared functions; `run_baseline.py` and the
  env must call the same code (same plant, projection, track, termination).

**Acceptance.** A rollout with the fixed-θ MPC through the env gives the same trajectory as
`run_baseline.py` from the same start state.

### Step 11 — Planner and training script
**Status:** done (2026-11-23). Real training waits for the group decisions below (Phase 4 prerequisite).

Run:
- `python scripts/check_planner.py`: 8 checks, about 2 minutes.
- `python scripts/train_rl.py --steps 3000 --val-freq 1500 --val-rollouts 2`: short training run,
  about 1 minute.
- `--constant-theta` gives one θ for the whole track (Step 12). An existing `--output` directory is
  continued from its last checkpoint.

What was built:
- `controllers/planner.py` (`RacingPlanner`): the Leap-C MPC driven only from the env observation, with
  s = 0 at the car.
  - **Road data and warm start:** stage parameters from the previous plan, shifted one stage as in
    Phase 1. The warm start is that plan moved back by its predicted progress of one step.
  - **New episodes:** if the observation does not continue the previous plan (state jump > 0.5), it
    starts cold. This handles mpc-sac's validation, which keeps the context across episodes.
  - **Failed solves:** return a non-finite action, so the env ends the episode as `mpc_failure`.
  - **Horizon check:** it refuses horizons longer than the observed road (30 m), so N = 38 is refused.
- `scripts/train_rl.py`: SAC-ZOP with the settings of mpc-sac's examples.
  - **Actor:** residual (starts at the initial θ), `scaling` extractor, float64.
  - **Validation:** on the validation split.
  - **Checkpoints:** at every validation (`ckpts/<step>_*.ckpt`); pick the best with `val_log.csv`. In
    "best" mode a run that never beats its first validation saves nothing.
  - **Run record:** `run_config.json` holds the settings and the commits of this repo and all
    submodules. They are read from files: the script runs no git command and records no uncommitted
    changes.
  - **Resuming:** two workarounds for mpc-sac's resume, described in the code.

Results:
- **Same controls as the runner:** fed the runner's own states, the planner gives the controls of
  `run_baseline.py --controller leapc-unshifted` over 3000 steps (v-init 10, 12, 15).
  - 99.9 % of steps are within 4e-6 of the control range, the worst within 1.4e-4 (one step at the
    hairpin).
  - Not bitwise, because the warm start is moved by the predicted, not the measured, progress.
- **Own closed loop:** the same ending, distance within 0.1 % and RMS ey within 2 %. Its states drift
  apart from the runner's over a long run (2 m in s after 3000 steps), as runs from slightly different
  start states do (Step 9).
- **How the test changed:** it first compared closed-loop states (limit 1e-6, then 1e-4). That only
  worked while N = 10 failed within about 700 steps. All limits were set after seeing the results.
- **Validation context:** 3 cold starts in 3 episodes, with results identical to a fresh context.
- **Checks:** a batch of observations gives the same actions as one at a time; a forced failure ends
  the episode as `mpc_failure`.
- **Training:** 3000 steps take about 45 s. Logs: return, length, violation, success, losses, θ and
  actions per step, solve time, success and retry rate. Checkpoints: about 15 MB per validation
  (networks and optimizer state, float64), plus one replay-buffer file that grows with the run (about
  1.9 KB per step, so about 370 MB at 200 000 steps). Resuming a stopped run tested from 3000 to 4500 steps.

Found, to settle before Step 12:
- **The `vx` weight dominates:** at the initial bounds, 57 % of the range stops the car. The fresh actor
  explores the whole range and its deterministic θ is already up to 3× off. Proposal [0.001, 0.05], in
  spec §3. The other six weights barely matter at ×3 or ÷3.
- **Exploration:** even with the narrower bound, check how wide the actor's initial noise is in θ.
  mpc-sac's default covers most of the bounds.

Notes from Steps 8–10 (all handled above):
- **Warm start across episodes:** mpc-sac's `validate()` never resets the planner context between
  episodes. The planner must detect a new episode and start cold (found by reading the code).
- **Relative s:** the observation has s = 0 at the car, so rebase the warm start and the stage-parameter
  lookup, which uses the previous plan. Decode with `planner_state` and `planner_road` from `env.py`.
- **Failed solve:** return a non-finite action, so the env ends the episode (ZOP only, see Step 14).
- **Precision:** the buffer is float32 by default, the MPC needs float64.
- **Actor:** `actor.residual = True` to start at the default θ; `actor.mlp.hidden_dims = None` for a
  constant θ (Step 12).
- **Validation starts** are the same at every checkpoint (`episode_rollout` uses a fixed seed).

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

### Stage-weight tuning (D6)
**Status:** done. **Frozen on 2026-11-24** by David ("for now"), so that RL training can start.
Changing them later invalidates every trained θ (Phase 4). Record the commit that contains them here
once it is made.
- **N = 38:** the best of the first study (trial 99), robust over 4 laps with all three controllers.
- **N = 10:** trial 84 of the first study, which is also trial 2 of the stricter study. It is not the
  best score in either study: both best-scoring sets failed outside the laps they were tuned on, and
  this one passed every check (below).

**Stricter N = 10 study** (`stage_weights_N10_robust`, 100 trials, 52 finished all 4 laps):
- **Best score:** trial 31, 225.47. **Trial 2 (old 84):** 226.54, third, with RMS ey 0.05 / 0.04 m
  instead of 0.09 / 0.13 m. The hand-picked weights scored 238.74; old trial 57 failed.
- **Checks of the two candidates** (10 000 steps per controller; 2 laps from 8, 12 and 20 m/s with both
  warm starts):

  | | Trial 31 | Old trial 84 (chosen) | Hand-picked |
  |---|---|---|---|
  | acados / Leap-C shifted | failure in lap 3 | 0 failures, 224.2–226.4 s | 0 failures |
  | Leap-C unshifted (RL) | failure after 501 steps | 0 failures, 221.9–223.9 s | 0 failures |
  | Other starts (6 runs) | 0 of 6 finish | 6 of 6, 222.0–228.0 s | 6 of 6, 235.1–239.4 s |
  | RMS ey / max \|ey\| | 0.2–0.7 m / up to 2.9 m | 0.035–0.052 m / ≤ 0.85 m | 0.037–0.058 m / ≤ 0.77 m |

- **Result:** with old trial 84, N = 10 is about 12 s (5 %) per lap faster than with the hand-picked
  weights, at the same tracking accuracy.
- **Lesson:** even the stricter objective found an overfitted best. Check candidates outside the tuning
  laps before taking a best score.

**Checks with both tuned defaults (2026-11-24):**
- `check_env.py`: 28 pass.
- `check_planner.py`: 11 pass, after another change of its limits (third time, after seeing results).
  - At the hairpin apex, planner and runner once stop at different local solutions for one step
    (steering rate 0.101 vs 0.052 rad/s), identical again one step later.
  - RMS ey of the two closed loops differed by 6.5 % at v-init 15, within the 8 % spread of Step 9.
  - New limits: at most 0.1 % of steps above 1e-3 of the control range; RMS ey within 10 %.
- **Gradient check, seeds 1 and 4** (N = 10 stage cost changed):
  - seed 1: 2 failures, both with a control pinned at its limit (true and analytic gradient 0, reference
    noise about 4e-5);
  - seed 4: 0 failures, 2 of 240 solves not converged.
  - The same kinds as accepted before.
- **The `vx`-weight sweep** gives the same pattern with the tuned weights: the lowest weight that
  survives all five situations is again 0.01 (now cruising at 21.9 m/s); 0.05 crawls at 5.2 m/s, and
  0.1 and above stop. The proposal [0.001, 0.05] stands.

**Defaults since 2026-11-24:**
- `build_ocp` uses `default_stage_weights(N)`: the tuned sets for N = 10 and N = 38, the hand-picked set
  for other horizons.
- `--stage-weights hand-picked` gives the old weights; `--stage-weights FILE` any other set.
- With the tuned N = 38 default: laps 194.32 / 192.32 / 192.24 / 192.32 s (acados), 0 failures, closest
  approach 0.82 m; Leap-C shifted and unshifted are the same within 0.08 s per lap.

**Why the first N = 10 result was rejected:**
- Made the default, its weights failed:
  - acados: solver failure early in lap 4;
  - Leap-C shifted: the same;
  - **Leap-C unshifted** (RL's warm start): failure after 474 steps, in lap 1.
- All three at the hairpin. The car brakes down to 2.4 m/s while still decelerating and cannot stay
  above the hard 2 m/s floor (the known jerk-limited braking problem).
- The study only scored lap 1 with the shifted warm start, so it could not see this.

**Stricter objective** (`--objective robust`, now the default):
- **Each trial drives** 2 laps from 10 m/s with the shifted warm start, and 2 laps from 15 m/s with the
  unshifted one (RL's).
- **Score:** the mean of (lap time + 50 × RMS ey) over the 4 laps if all finish, otherwise 1000 + the
  metres missing.
- **The unshifted warm start is copied with the acados solver,** because a Leap-C build takes 72 s per
  trial. It keeps the previous plan unshifted and puts the measured state at its first node. Checked
  against the real Leap-C runner: the same failure at step 474 (s = 716 m) for the first tuned weights,
  and the same lap 238.88 s for the hand-picked ones.
- **Queued first in the study:** the hand-picked weights, then the first study's trials 57 and 84.

**First studies** (objective `lap1`: lap 1 from 10 m/s, shifted warm start only):

**Results** (`results/optuna/stage_weights_N{10,38}/`; logs `results/optuna_N{10,38}.log`):

| | Hand-picked (trial 0) | Best trial |
|---|---|---|
| N = 38 score / lap / RMS ey | 197.82 / 194.48 s / 0.067 m | 197.20 / 194.32 s / 0.058 m (trial 99) |
| N = 10 score / lap / RMS ey | 240.46 / 238.80 s / 0.033 m | 224.32 / 219.36 s / 0.099 m (trial 57) |
| N = 10 finished laps | | 67 of 100 trials |

Best weights (progress 1 in both):
- **N = 38:** ey 16.1, epsi 24.3, vy 0.145, yaw rate 0.232, acceleration 0.079, jerk 0.0052,
  steering rate 0.337.
- **N = 10:** ey 182, epsi 99.1, vy 0.0149, yaw rate 0.266, acceleration 0.0128, jerk 0.0027,
  steering rate 0.238.

Check from 5 start speeds (8, 10, 12, 15, 20 m/s; tuned only at 10), one lap each:

| | Hand-picked: laps, RMS ey | Tuned: laps, RMS ey | Mean score hand / tuned |
|---|---|---|---|
| N = 38 | 192.80–195.04 s, 0.061–0.074 m | 192.64–195.12 s, 0.058–0.072 m | 197.45 / 197.13 |
| N = 10 | 235.84–238.80 s, 0.023–0.055 m | 217.44–221.60 s, 0.099–0.146 m | 239.29 / 226.36 |

- **N = 38: no real gain.** About 0.2 s per lap, within the run-to-run spread. The hand-picked
  weights were already near the best this objective finds.
- **N = 10: about 18 s faster from every start, always finishing, but 3–6× the tracking error.**
  - The time comes from a higher cruise speed: 23.0 instead of 20.6 m/s at most, median 22.2 instead
    of 20.0. It gains on every straight and most corners, and loses 0.7 s at corners 3+4.
  - Max |ey| on lap 1 is 1.12 m instead of 0.38 m, and mean |jerk| 1.03 instead of 0.43 m/s³.
  - Which weight raises the speed is not yet isolated. The lower jerk weight correlates most with lap
    time over the 67 finished trials (+0.80); acceleration is also lower.
- **Correction:** the earlier note that the stage weights cannot change N = 10's speed was wrong. They
  shift the balance against the terminal `vx` weight.

**For the group:**
- **Overlap with RL:** tuned this way, the N = 10 stage weights already take part of the speed gain that
  RL was meant to provide (D5: RL learns the terminal cost). RL would then start from a faster but less
  accurate N = 10.
- **The tracking weight decides the outcome:** at 50 s per metre of RMS ey, 18 s of lap time outweighs
  0.07–0.11 m more RMS ey. A larger tracking weight would prefer more accurate trials, e.g. trial 84:
  225.9 s and RMS ey 0.025 m, which is faster and more accurate than the hand-picked weights.
- **Options:** keep this result; raise the tracking weight and rerun; or tune N = 10 only on tracking
  and smoothness, with its speed left to RL.

Run: `python scripts/tune_stage_weights.py --N 38 --trials 100 --workers 8` (and `--N 10`). Running it
again with the same study adds trials. Results go to `results/optuna/stage_weights_N<N>/`
(`trials.csv`, `best_stage_weights.json`).

Set-up (decided by David 2026-11-24):
- **Separate weights per horizon:** N = 38 and N = 10 each get their own tuned stage weights, so both
  are at their best in the comparison. RL for N = 10 uses the N = 10 weights.
- **What is tuned:** the 7 stage weights other than progress (log-uniform, about 1/100 to 10× the
  hand-picked values). The progress weight stays 1, because scaling all weights together barely changes
  the plan. The terminal cost stays at the initial P (D5: RL learns it).
- **Objective:** lap time from the standing start (10 m/s), plus 50 s per metre of RMS ey. An unfinished
  lap scores 1000 + the metres missing. Lower is better. The tracking weight is a placeholder.
- **Trial 0** is the hand-picked weights, as a reference.
- **Each trial** compiles its own solver (3.5 s) and drives one lap: about 25 s for N = 10, 80 s for N = 38.
- **Checks:** `evaluation/lap.py` gives the runner's lap times exactly (238.8 s for N = 10, 194.48 s for
  N = 38). A weights file reproduces its trial in `run_baseline.py --stage-weights FILE` (249.44 s for
  the test trial).
- **Using the result:** `run_baseline.py --stage-weights FILE`, `train_rl.py --stage-weights FILE`.
  Compiled solvers for other weights get their own build folder.

Known limits:
- **The objective differs from the RL reward** (lap time instead of summed progress), by David's choice.
  This differs from what D6 originally asked; the group should know.
- **One lap from one start per trial.** Lap times vary by about ±0.3 s between nearly identical starts
  (Step 9), so small differences between trials are partly noise.

## Phase 4 — Training

**Prerequisite:** the stage cost weights from Optuna are frozen and committed (D5, D6), one set per
horizon. If they change after training, the trained θ is invalid. Frozen on 2026-11-24; the commit is
still to be made.

### Step 12 — SAC-ZOP, constant θ
**Status:** first runs done (2026-11-24). Result: unstable. Constant θ learns "maximum speed", which
fails at the hairpin.

**Validation** (10 fixed episodes; the initial θ scores 296.6 on them):

| Step (×1000) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0 | 297 | 36 | 301 | 293 | 195 | 259 | 31 | 30 | 36 | 283 | 38 |
| Seed 1 | 297 | 82 | 317 | 278 | 315 | 316 | 316 | 315 | 31 | 32 | 317 |
| Seed 2 | 297 | 312 | 321 | 316 | 316 | 31 | 305 | 316 | **323** | 315 | 318 |

- **Two kinds of checkpoint:** good ones score 315–323, about 7–9 % above the initial θ, with all 10
  episodes successful. Bad ones score 30–80: all 10 episodes leave the track or stop below 3 m/s.
- **Best checkpoints:** seed 2 at 80 000 (323), seed 1 at 40 000–70 000 (315–317), seed 0 at 20 000 (301).
- **The `vx` weight sits at its lower bound (0.001) in all three seeds:** the actor wants maximum speed.
- **The other six weights end far apart between seeds,** e.g. epsi 0.00025 / 281 / 544, vy 567 / 0.08 /
  908. They barely affect the result (spec §3), so they drift.
- **Training failures are at the hairpin:** 60 / 36 / 45 `low_speed` and 7 / 1 / 3 `left_track` per seed,
  in 326–340 episodes. Most end at s = 700–900 m (corners 3+4, in the training split), e.g. seed 0: 54 of
  67 between 700 and 800 m.
- **Interpretation:** a constant θ cannot slow down for the hairpin only. Fast enough to gain on the
  rest of the track means too fast for the hairpin; the spec predicted this (§3). The validation
  section (corners 12+13) is milder, so good checkpoints do well there. Road-dependent θ (Step 13)
  should be able to brake only where needed.
- **Validation collapses** come and go between checkpoints, so the best checkpoint must be chosen by
  validation (`val_log.csv`), not by taking the last one.

**Rerun with the Step 13 second-round settings (step12b), started 2026-11-24**, so that Step 12 and 13
differ only in constant vs road-dependent θ:
- **Settings:** `--constant-theta`, 4 seeds × 200 000 steps, tracking weight 1.0, validation with the
  hairpin.
- **Runs:** `results/rl/step12b_constant_seed{0,1,2,3}`.

**Results (2026-11-24).** Step 12b vs 13b, best checkpoint of each seed by validation, one start each
(`results/compare/step12b_*` and `step13b_*`; references: N = 38 194.40 s / 27.04 s, N = 10 default
224.16 s / 31.12 s):

| Seed | 12b lap | 12b test | 13b lap | 13b test |
|---|---|---|---|---|
| 0 | 203.36 s | 29.92 s | 202.64 s | 28.88 s |
| 1 | 203.12 s | 29.20 s | 208.40 s | 28.64 s |
| 2 | fails at 700 m (MPC failure) | 28.96 s | 210.56 s | 32.56 s |
| 3 | **200.80 s** | 28.96 s | fails at 1679 m | fails |
| Laps finished | 3 of 4 | 4 of 4 | 3 of 4 | 3 of 4 |
| Mean of finished | 202.43 s | 29.26 s | 207.20 s | 30.03 s |
| Max \|ey\| on the lap | 0.82–1.21 m | 0.59–0.96 m | 1.10–1.78 m | 0.65–0.75 m (seeds 0, 1) |

- **With identical settings, constant θ is at least as good as road-dependent θ.**
  - It finishes as many laps (3 of 4), the test section more often (4 of 4 vs 3 of 4), with faster
    mean times and smaller deviations.
  - The best lap of all is a constant θ (seed 3, 200.80 s, 79 % of the gap to N = 38 closed).
- **So most of the second round's improvement came from the reward and validation changes**, not from
  making θ road-dependent.
- **What the constant θ learned:**
  - all four seeds: the `vx` weight at its lower bound (full speed) and a very high ey weight (720–960,
    initial 40);
  - other weights differ between seeds.
  - A high terminal ey weight forces every plan to end on the raceline, which makes the car brake for
    corners by itself.
- **Validation is misleading:** Step 12b never scored above the initial θ (best 241–248 vs 259.6), yet
  its controllers lap 21–23 s faster than the default. The validation score (7 of 10 episodes at the
  hairpin, tracking-weighted) does not rank controllers the way lap time does. Choosing checkpoints by
  it is questionable.
- **Still one start per evaluation:** the differences between 12b and 13b are partly within run-to-run
  noise. Step 15 needs several starts.

**Comparison with `scripts/compare.py`** (new 2026-11-24, like LATCH's `compare`). Best checkpoint, seed 2
at 80 000:

| Start | N = 38 default | N = 10 default | N = 10 learned (constant θ) |
|---|---|---|---|
| Test section (s = 1300 m, 20 m/s) | 27.04 s, RMS ey 0.093 m | 31.12 s, RMS ey 0.041 m | 30.56 s, RMS ey 0.297 m, max \|ey\| 1.28 m |
| Full lap from s = 0, 10 m/s | 194.40 s | 224.16 s | **stops below 3 m/s at s = 797 m** (hairpin) |

- **On the test section** the learned θ is 0.56 s faster than the default but much less accurate.
- **On the lap** it drives 25 m/s like N = 38, brakes too late for the hairpin, runs 6.3 m wide and stops.
- **The test section** (corners 5+6) does not contain the hairpin. Validation and test can therefore look
  good while a full lap fails; report full laps in Step 15 as well.
- **In `run_baseline.py`, which has no 3 m/s stop** (`--learned`, below), the same policy survives the
  hairpin. It drops to 2.5 m/s for 23 steps at s = 796–801 m, then drives on at 25 m/s. It reached
  s = 2418 m in 1500 steps with no failure, max |ey| 6.3 m. So the env's 3 m/s termination is what ends
  these episodes. Whether that stop is too strict is a design question for the group: it was added
  because the MPC's hard 2 m/s floor can make solves fail.

**Running a learned constant θ in the runner** (new 2026-11-24):
`python scripts/run_baseline.py --N 10 --controller leapc-unshifted --learned results/rl/<run>
[--learned-step K]`.
- **What it loads:** the θ of a `--constant-theta` run, by default the best validation checkpoint, and
  writes it to `learned_theta.json`.
- **What it refuses:** road-dependent runs (θ changes every step; use `compare.py`), a different N,
  different stage weights, and `--controller acados`.
- **Shared code:** policy loading is in `latch13/learning/policy.py`, used by `run_baseline.py`,
  `compare.py` and `train_rl.py` (which takes its actor settings from there).
- **Runs:** `results/rl/step12_constant_seed{0,1,2}` (logs `results/rl_step12_constant_seed*.log`).
- **Settings:** `train_rl.py --constant-theta --steps 100000 --val-freq 10000 --val-rollouts 10`, seeds 0–2,
  P = 100.
- **Code state:** not yet committed (frozen weights, `vx` bound, penalty and logging are from 2026-11-24;
  the newest commit holds 2026-11-23).
- **Per-episode logs** (new 2026-11-24): `episodes_train.csv` and `episodes_val.csv` in each run folder,
  one row per episode with start and end position and speed, the termination, distance and return.
- **Speed:** the second pilot did 20 000 steps in 4 minutes (about 12 ms per step, alone on the machine).
θ constant over the whole track: set `actor.mlp.hidden_dims = None`, so the actor is a single learned
vector (`train_rl.py --constant-theta`). Removing the road preview alone is not enough, because θ would
still vary with the car's state. First RL result.

Before starting:
- **θ bounds:** set on 2026-11-24 (`vx` [0.001, 0.05]).
- **Initial exploration, measured 2026-11-24** (fresh actor, 2000 samples, 3 initialisations):
  - **vx weight:** sampled over almost the whole new range (5–95 %: 0.0012–0.046). 35–46 % of samples are
    ≤ 0.007 (too fast for the hairpin at N = 10) and 21–26 % are ≥ 0.03 (crawling).
  - **Other six weights:** sampled over all 7 decades (about 1e-4 to 850).
  - **Deterministic start:** vx weight 0.009–0.014. Validation at step 0 scored 194, against about 292
    for the exact initial θ.
  - **Cause:** mpc-sac's default initial noise (std about 1 before the tanh squashing).
  - **Option, not applied:** cap it, e.g. `log_std_max = -1`.
  - **Decision (David):** pilot with the default first.

**Pilot, 2026-11-24** (`results/rl/pilot_default_noise_seed0`):
- **Setup:** 20 000 steps, road-dependent θ (the `train_rl.py` default, not `--constant-theta`), default
  exploration noise, seed 0, validation every 5000 steps on 5 fixed episodes.

| Step | 0 | 5000 | 10 000 | 15 000 | 20 000 |
|---|---|---|---|---|---|
| Validation score | 200.2 | 278.9 | **311.6** | 307.5 | 296.1 |
| Validation episode length (steps) | 375 | 328 | 280 | 299 | 309 |

- **Baseline:** the exact initial θ on the same 5 episodes scores 295.9 (length 313). The step-0 score
  is lower because of the fresh network's offset.
- **Validation:** the best checkpoint (10 000) is 5 % above the initial θ. It then falls back to its
  level. One seed and 5 episodes, so this is not yet evidence of learning beyond noise.
- **Training episodes:** success fell from 100 % (first 5000 steps) to 78 % (last 3000). No track
  violations and an MPC success rate of 1.0, so the failures are other terminations, most likely
  `low_speed` (not logged separately).
- **θ drift (mean of the sampled θ):**
  - the `vx` weight drops to 0.0013, near its lower bound 0.001, so the actor asks for maximum speed;
  - the ey, r and δ weights collapse from 40 / 10 / 10 to about 0.03 / 0.02 / 0.001, so almost no
    terminal lateral cost is left (Step 3 found the vy, r and δ terms necessary for SQP convergence; no
    solver failures appeared here);
  - the epsi weight drops to about 3.
- **Exploration collapses quickly:** the noise std falls from 0.39 to 0.007 and the entropy from 4.7 to
  0.2 by 20 000 steps. The initial width was not the problem.
- **Likely reason for the falling success: the placeholder reward.** A failure costs P = 10, about 12
  steps of full-speed progress, while going faster earns progress on every step. Trading some failures
  for speed may pay. To raise with the group together with P and the reward weights.
- **Termination type:** logged since 2026-11-24 (`end_<reason>` columns in `train_log.csv` and
  `val_log.csv`). A 2000-step test showed `left_track` endings, so the pilot's failures need not have
  been `low_speed`.
- **Changed after the pilot (2026-11-24):** failure penalty 10 → 100. The pilot ran with 10.

**Second pilot, 2026-11-24** (`results/rl/pilot_penalty100_seed0`): the same settings with P = 100 and
termination logging. It ran on code not yet committed (the newest commit holds the state of 2026-11-23).

| Step | 0 | 5000 | 10 000 | 15 000 | 20 000 |
|---|---|---|---|---|---|
| Validation score, P = 100 | 200.2 | 278.9 | 290.7 | 316.8 | **320.7** |
| Validation score, first pilot (P = 10) | 200.2 | 278.9 | 311.6 | 307.5 | 296.1 |
| Validation episode length, P = 100 | 375 | 328 | 318 | 288 | 287 |

- **Validation:** at 20 000 steps it is 8 % above the initial θ (295.9), still rising, and every
  validation episode ends successfully. One seed and 5 episodes.
- **Identical up to step 5000:** both pilots use the same seed, and the penalty only matters once an
  episode fails.
- **Training episodes still fail more over time:** success falls from 100 % to about 80 %, as in the
  first pilot.
  - **How they end** (last 3000 steps): about 14–18 % `low_speed`, 3 % `left_track`, no MPC failures.
  - So most failures are N = 10 braking down below 3 m/s, most likely at the hairpin after arriving
    faster. Not confirmed: the place of the failure is not logged.
- **θ:**
  - the `vx` weight again goes near its lower bound (0.0019): maximum speed;
  - the lateral weights first collapse as before, then partly recover (at 19 000 steps: ey 0.44,
    r 0.89, vy 0.50, a 0.55, δ 0.013).
- **Exploration** settles at a std of about 0.04 (first pilot: 0.007).

### Step 13 — SAC-ZOP, road-dependent θ (main result)
**Status:** in progress (2026-11-24): a working controller was found, but it is not yet reliable across
seeds and not yet fully evaluated.
- **First round** (settings of Step 12): no seed laps (below).
- **Second round** (tracking weight 1.0, validation with the hairpin, 200 000 steps): seed 0 laps in
  202.64 s and handles the held-out test section; 2 of 4 seeds lap ("Second-round results").
- **Open:**
  - **Commit:** the runs record commit `cf94fb5`, but ran on later, uncommitted changes (reward,
    validation sections, penalty, `vx` bound, logging). Commit them so the runs can be reproduced.
  - **Reliability:** 1 of 4 seeds clearly good.
  - **Evaluation:** several starts, mean ± spread over seeds (Step 15).
  - **Comparison with Step 12:** done (Step 12, "step12b"). With the same settings, constant θ is at
    least as good, so road-dependence has not shown an advantage yet.

**Validation** (10 fixed episodes on corners 12+13; initial θ 296.6, Step 12 best 323):

| Step (×1000) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0 | 200 | 162 | 320 | 320 | 313 | **327** | 317 | 158 | 317 | 320 | 310 |
| Seed 1 | 265 | 315 | 300 | 215 | 310 | 305 | 276 | 319 | **322** | 314 | 314 |
| Seed 2 | 291 | **325** | 313 | 94 | 94 | 94 | 314 | 314 | 314 | 315 | 96 |

**Best checkpoints in `compare.py`** (results in `results/compare/step13_seed*_{lap,test}`):

| | Test section (corners 5+6, held out) | Full lap from s = 0 |
|---|---|---|
| N = 38 default | 27.04 s | 194.40 s |
| N = 10 default | 31.12 s | 224.16 s |
| Step 12 seed 2 (constant θ) | 30.56 s, RMS ey 0.30 m | stops at the hairpin (797 m) |
| Step 13 seed 0 (step 50 000) | **left the track** after 366 m, up to 9.8 m wide | stops at the hairpin (736 m) |
| Step 13 seed 1 (step 80 000) | **left the track** after 365 m | gets through the hairpin, leaves the track at 1665 m (corners 5+6) |
| Step 13 seed 2 (step 10 000) | **29.44 s**, RMS ey 0.30 m | stops before the hairpin (712 m) |

- **Training failures** are again almost all at the hairpin (s = 630–950 m), mostly `low_speed`.
  - In the last quarter of training: 7 %, 6 % and 17 % of episodes for seeds 0, 1, 2.
  - Step 12 seeds ended at 13–26 %.
- **The road-dependent θ does not generalise to corners 5+6.** It never trained there; they are the test
  section. Two of three seeds leave the track there, though they validate well on corners 12+13.
- **The hairpin is still not solved,** although it is in the training split. Only seed 1 gets through
  it on a lap.
- **Validation cannot see either problem:** it contains neither the hairpin nor corners 5+6, so the
  "best" checkpoint can fail both. Seed 2's best is from step 10 000, barely trained.

**Possible next steps (to decide):**
1. Raise the tracking weight (spec §5): speed now dominates the reward about 15:1.
2. Reconsider the 3 m/s stop at the hairpin (the car recovers in the runner).
3. Make validation include harder corners, or validate on a full lap, so the checkpoint choice sees
   the failures.
4. Train longer or with more seeds once 1–3 are settled.

**Second round (step13b), started 2026-11-24** with David's choice of 1, 3 and 4 (2 skipped for now):
- **Reward tracking weight 0.1 → 1.0** (`RewardOptions.tracking` in `env.py`). This is the RL reward only;
  the frozen Optuna stage weights of the MPC are unchanged.
- **Validation also covers the hairpin:** a second section, s = 450–1000 m, next to corners 12+13. 7 of the
  10 fixed validation episodes start there (468–509 m), 3 at corners 12+13. The hairpin is also in the
  training split; the test section stays held out.
- **New validation baseline** (initial θ, new reward and sections): 259.6, all 10 episodes successful.
  Not comparable with the earlier 296.6.
- **Runs:** 4 seeds × 200 000 steps, `results/rl/step13b_road_seed{0,1,2,3}`. About 0.5 GB of memory each.
- `check_env.py` updated (the validation hairpin may overlap training; the test section may not
  overlap either): 28 pass.

**Second-round results (2026-11-24):**

| Seed | Validation (every 10 000 steps; initial θ 259.6) | Best |
|---|---|---|
| 0 | 188 94 165 −174 −327 −186 −230 260 −337 −205 236 223 114 205 255 266 218 223 261 263 265 | **266.2** at 150 000 |
| 1 | 238 194 −225 195 −21 −91 184 −229 −115 −29 −118 −258 −218 167 158 −125 −89 62 181 188 160 | 238.0 at 0 (never better) |
| 2 | 34 127 231 256 257 139 244 −83 18 40 132 39 −18 −217 258 244 263 51 36 17 −76 | 262.7 at 160 000 |
| 3 | 236 75 132 146 131 160 216 244 227 188 106 169 120 91 221 199 169 77 189 216 147 | 243.8 at 70 000 |

- **Negative scores** are validation episodes stopping below 3 m/s at the hairpin.
- **Training failures:** all at the hairpin, all `low_speed` (3 `left_track` for seed 1).
  - Last quarter of training: 4 / 4 / 21 / 1 % of episodes for seeds 0–3.
  - 55–75 % of training episodes end at the 30 s limit.

**Best checkpoints in `compare.py`** (`results/compare/step13b_seed*_{lap,test}`):

| | Full lap from s = 0 | Test section (corners 5+6, held out) |
|---|---|---|
| N = 38 default | 194.40 s, RMS ey 0.063 m | 27.04 s, RMS ey 0.093 m |
| N = 10 default | 224.16 s, RMS ey 0.038 m | 31.12 s, RMS ey 0.041 m |
| **Seed 0 (150 000)** | **202.64 s**, RMS ey 0.18 m, max 1.78 m, no failure | **28.88 s**, RMS ey 0.16 m, max 0.65 m |
| Seed 2 (160 000) | 210.56 s, RMS ey 0.18 m, max 1.69 m | 32.56 s, RMS ey 0.19 m |
| Seed 3 (70 000) | stops at s = 1679 m (corners 5+6) | stops below 3 m/s |

- **Seed 0 closes 72 % of the N = 10 → N = 38 lap-time gap**
  ((224.16 − 202.64) / (224.16 − 194.40)) and 55 % on the held-out test section. It gets through the
  hairpin and the unseen corners 5+6.
- **Accuracy:** much better than the first round (RMS ey 0.9–3.4 m there), but still about 4× worse than
  N = 10 default.
- **Not robust across seeds:** 2 of 4 seeds give a controller that laps; 1 of 4 is clearly better than
  the default on both. One start per evaluation so far; Step 15 needs several starts and laps.
- **Runs:** `results/rl/step13_road_seed{0,1,2}` (logs `results/rl_step13_road_seed*.log`).
- **Settings:** as in Step 12 but without `--constant-theta`: 100 000 steps, validation every 10 000 on
  10 episodes, P = 100, the 3 m/s stop kept so it matches Step 12.
- **Evaluate** with `compare.py --learned <run> --start test` and `--start lap`.
Observation with road preview beyond the horizon; the network outputs θ per step.

### Step 14 — SAC-FOP
**Status:** not started
Same setup as step 13 with gradients through the MPC (`fop`, optionally `foa`). Compare
sample efficiency and final performance with ZOP. Watch `masked_samples_perc` and `zero_dudp_perc`.

**Note: `zero_dudp_perc` is inverted in `mpc-sac` `d38726d`** (`rl/sac_fop.py:304` logs
`1 - zero_grads.mean()`), so it is the share of *nonzero* gradients: a low value is the bad case.
Use `SENSITIVITY_SOLVER_OPTIONS` for this step (Step 8).

For all training steps record: seed, commits (`latch13`, `leap-c`, `leap-c-lab`, `mpc-sac`), config,
training curves, best validation checkpoint.

---

## Phase 5 — Evaluation and report

### Step 15 — Comparison
**Status:** not started. Tool ready: `scripts/compare.py` (2026-11-24).
- **Policies in git:** `policies/` holds the best checkpoint of each Step 12b and 13b run (5.5 MB; see
  its README), so the group can compare without retraining.
  Example: `python scripts/compare.py --start lap --live --learned policies/step12b_constant_seed3
  policies/step13b_road_seed0`.
- **Several learned runs:** `--learned` takes any number of runs, each driven as its own controller;
  `RUN:STEP` picks a checkpoint.
- **What it does:** drives N = 38 default, N = 10 default and N = 10 learned one after another from the
  same start, through the RL env (same plant, exit check and endings).
- **Options:** `--live 1|2` for the live view, `--start test|lap`, `--learned RUN [--step K]` (default:
  the best validation checkpoint).
- **Output** in `results/compare/<time>/`: `comparison.png` (speed, ey, time and edge margin over s),
  `summary.csv` and one CSV per run.
- **Controllers:** N = 38 runs the Leap-C MPC with RL's warm start (like `run_baseline.py --controller
  leapc-unshifted`); both N = 10 runs use the RL planner.

Controllers: N = 10 learned (steps 12–14), N = 38 benchmark, and N = 10 default terminal cost (see
step 5: since 2026-11-23 this is the safe, slow controller with terminal `vx` weight 0.01, which equals
the RL starting point). Evaluate on the held-out test section and on full laps, several seeds, mean ± std.

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
