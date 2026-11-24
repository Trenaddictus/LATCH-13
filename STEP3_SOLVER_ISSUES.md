# Step 3 solver issues: SQP convergence with a progress reward

Status as of 2026-11-20. Repo `~/Refactor`, branch `feature/track-edges` (Steps 1–2 pushed in `a4c6631`).
The Step 3 changes described here are **uncommitted** in the working tree.
Context: `ROADMAP.md` (Phase 1, Step 3) and `REFACTOR_SUMMARY.md`.

**Summary:**
- **What works:** the Step 3 formulation drives correctly.
- **What fails:** the acados SQP solver does not converge reliably, so closed-loop runs stop with `solver_status_2` (iteration limit).
- **What's undecided:** the problem is understood in part, and no fix has been chosen. The group is to discuss it before going on.

---

## 1. What Step 3 changed

| | Before (Steps 1–2) | Now (Step 3) |
|---|---|---|
| Speed in the stage cost | `w_vx·(vx − v_ref)²`, constant `v_ref` | `−q_s·ṡ`, with `ṡ = (vx cos epsi − vy sin epsi)/(1 − κ ey)` (the exact model expression) |
| Terminal cost | `40 ey² + 20 epsi² + 2 (vx − v_ref)²` | `40 ey² + 20 epsi²` |
| Horizon parameters `[κ, ey_min, ey_max]` | looked up at `s0 + j·dt·v_ref` | looked up at the previous solution's predicted `s`, shifted one stage and anchored at the measured `s` |
| Command line | `--speed` | `--v-init` (start speed and cold-start guess, default 10 m/s), `--progress-weight` (q_s, default 1) |
| Hessian regularisation | none | `regularize_method = MIRROR` |

Everything else is unchanged:
- **Model and plant:** TUM's single-track model with Pacejka tyres, in Frenet coordinates; the TUM plant at 0.02 s; MPC step 0.08 s.
- **Stage weights:** ey 20, epsi 10, vy 0.1, r 0.1, a 0.05, jerk 0.01, steering rate 0.05.
- **Hard limits:** `vx ∈ [1, 25]`, lateral acceleration within ±5.886 m/s², and the TUM actuator limits.
- **Soft track-edge constraint:** L1 1e4, L2 10.
- **Solver:** SQP, exact Hessian, HPIPM, ERK (RK4, 4 steps), funnel line search, Levenberg–Marquardt 1e-2, `tol_stat` 1e-5, 100 iterations maximum.

**Why `MIRROR`:** the progress reward is not convex. Without regularisation, the very first (cold-start) solve fails with status 3 after 34 iterations; with `MIRROR` it converges in 8.

## 2. What works

With `tol_stat` relaxed to 1e-4, N = 38 behaves the way Step 3 intends:

| t | s | vx | |
|---|---|---|---|
| 0 s | 0 m | 10.0 m/s | cold start |
| 8 s | 151 m | 25.0 m/s | at the speed limit on the straight |
| 32 s | 710 m | 15.5 m/s | braking for the corner sequence |
| 40 s | 832 m | 17.1 m/s | 24 m hairpin, lateral acceleration 4.9 m/s² |
| 48 s | 1018 m | 25.0 m/s | back at the limit |

It covered 1450 m with no track violations; the car body stayed at least 1.44 m from the edges, and the slack stayed around 1e-9. The Step 3 acceptance criteria for behaviour are met: the controller solves from a cold start, accelerates on straights and slows down in corners.

## 3. The problem

- **With the default settings** (`tol_stat` 1e-5), N = 38 stops at **step 48** (s ≈ 54 m, on the first straight) with `solver_status_2`. The cold restart fails too.
- **The best variant found** reaches step 817 (s ≈ 1425 m), still short of a lap (4462 m).

There are three distinct effects.

### Effect 1: a flat lateral direction at the end of the horizon (confirmed)
- **Where:** the first failure, step 48, the moment the plan first reaches 25 m/s at the end of the horizon.
- **What happens:** the stationarity residual falls to about 3e-5 within 5 iterations, then slowly *increases* with full steps (α = 1) up to 6.8e-4 at 100 iterations.
- **What drifts:** comparing iterates 5 and 100 shows changes only in lateral states at the last stages. The steering rate at stage 36 of 38 goes from −0.0008 to −0.067 rad/s, and vy, r and δ follow. Speed and acceleration barely move.
- **Interpretation:** the end-of-horizon lateral states are barely penalised (the terminal cost has only ey and epsi). With a progress reward the speed costate is large, so the Hessian is indefinite in those directions.
- **Fix:** **terminal weights on vy, r and δ remove it.** With weight 10 the first failure moves from step 48 to 326, and the drift pattern disappears. With exact Newton steps (`NO_REGULARIZE`) this problem fails (status 3), so this direction really is indefinite. It's a property of the problem, not of the regularisation.

### Effect 2: general slowness from near-bang-bang plans (confirmed, partial)
- **What happens:** with a progress reward, acceleration and jerk sit at their limits for whole stretches, and typical solves take about 7 iterations.
- **Fix:** **larger acceleration/jerk weights** (0.5 / 0.1 instead of 0.05 / 0.01) cut typical solves to about 4 iterations and remove most slow ones. They don't remove the failures.

### Effect 3: failures when a tight corner enters the horizon at speed (cause: the `MIRROR` regularisation)
- **Where:** once effects 1 and 2 are mitigated, almost every failure happens at about 25 m/s, just before a 35–44 m corner (s ≈ 590–600 m and s ≈ 1425–1448 m). That's the moment the plan must switch from flat out to braking.
- **What is not the cause:** the hard lateral-acceleration limit. Making it soft changes nothing; its slack is never used, and the failures happen at the same steps.
- **What a longer solve shows:** re-solving the failing problem (step 323, s = 590 m) with 500 iterations, it **does converge** after 134 iterations, linearly (about ×0.96 per iteration, all full steps). The result is a sensible gentle braking plan.
- **Changing Levenberg–Marquardt** (1e-2, 1e-3, 0) has no effect.
- **The same saved problem with other regularisations:**

| Regularisation | Result |
|---|---|
| `MIRROR` | converges in 134 iterations (linear) |
| **`NO_REGULARIZE`** (exact Hessian) | **converges in 9 iterations** |
| `PROJECT_REDUC_HESS` | fails, 500 iterations (residual 0.23) |
| `GERSHGORIN_LEVENBERG_MARQUARDT` | fails, 500 iterations (residual 1.2e-3) |

- **Interpretation:** near the solution the problem is fine and exact Newton steps work. `MIRROR` modifies each stage Hessian block wherever it is locally indefinite, even where the problem as a whole doesn't need it. The steps are then no longer Newton steps, and convergence becomes slow and linear.
- **But the exact Hessian is not usable on its own:**
  - closed loop with `NO_REGULARIZE` fails at the cold start (step 0), even with the terminal lateral weights;
  - in a hybrid (below) it fails on about 11% of warm-started solves.

## 4. Everything tested

All tests are N = 38, Silverstone ×10, unless noted. "First failure" is the closed-loop step where the solver and its cold restart both failed.

| Variant | First failure | Notes |
|---|---|---|
| Default (`MIRROR`, `tol_stat` 1e-5) | step 48 | effect 1 |
| `tol_stat` 1e-4 | step 816 (s = 1448 m) | 115 solves over 8 iterations, p95 141 ms |
| QP tolerances 1e-9, QP max iterations 200 | step 48 | |
| Levenberg–Marquardt 0 | step 48 | |
| `MERIT_BACKTRACKING` / `FIXED_STEP` line search | step 48 | |
| `PROJECT` / `CONVEXIFY` | step 48 / 50 | |
| `exact_hess_dyn = 0` | step 0 | |
| Discrete RK4 dynamics (`integrator_type = DISCRETE`) | step 48 | |
| Speed limit 60 m/s instead of 25 | step 120 (31 m/s) | not specific to the 25 m/s limit |
| Soft speed limit | step 48 | |
| Hard track-edge constraints | step 48 | the track slacks aren't the cause |
| vy and r stage weights ×10 | step 49 | |
| `q_s = 0.1` | step 200 | delays it |
| Terminal vy, r, δ weight 1 / 10 | step 140 / 326 | fixes effect 1 |
| Terminal weight 10 + `tol_stat` 1e-4 | step 448 | |
| Acceleration/jerk 0.5 / 0.1 (± terminal weights) | step 323 (s = 590 m) | effect 2 better, ~4 iterations typical |
| Acceleration/jerk 2.0 / 0.5 | step 429 | mean speed drops to 17.3 m/s |
| Soft lateral acceleration (L1 1e3), alone / + terminal / + all | 48 / 543 / 323 | slack never used, so not the cause |
| `NO_REGULARIZE` + terminal weights (± smoother weights) | step 0 | cold start fails |
| Convex speed term `(vx − 25)²` (w = 1) instead of the progress reward, + terminal weights (± smoother weights) | step 321 (s = 590–591 m) | typical 3–4 iterations; **same failure location as the progress reward** |
| Speed limit 20 m/s (+ terminal weights) | step 212 (s = 316 m) | abrupt failure on a gentle corner, residual 8.6e-5 |
| N = 50 (4 s horizon), 25 m/s (+ terminal weights) | step 290 (s = 529 m) | abrupt failure, residual 2.2e-5 |
| QP tolerances 1e-8 or 1e-10, QP max iterations 200 (+ terminal weights) | step 797 (s = 1411 m) | residual 1.27e-5 at failure; 1e-8 and 1e-10 identical |
| **QP tolerances 1e-8 + `tol_stat` 1e-4 (+ terminal weights)** | **none in 3500 steps** | **full lap, 194.5 s**; see section 7 |
| **Hybrid:** `MIRROR` for cold starts, exact Hessian when warm, `MIRROR` fallback; + terminal + smoother weights | **step 817 (s = 1425 m)** | exact succeeded on 730/817 warm solves; 87 fallbacks; mean 4.1 iterations; solve mean 57, p95 199, max 1454 ms |

For reference, before Step 3 (with `v_ref` tracking) all of this converged in 3 iterations, with at most 5.

## 5. Current state of the code

- **The Step 3 changes are uncommitted.**
- **In `ocp.py`:**
  - `MIRROR`;
  - terminal weights 10 on vy, r and δ;
  - soft lateral acceleration (L1 1e3, L2 10);
  - `nlp_solver_tol_stat = 1e-3`;
  - `qp_solver_tol_* = 1e-8`, `qp_solver_iter_max = 200`.
- **In `run_baseline.py`:** the cold-start guess starts from the measured state.
- **Stage weights are unchanged:** acceleration/jerk 0.05 / 0.01.
- **Result:** N = 38 drives 4+ laps without failures (section 8).
- **The other variants in section 4** were runtime patches in temporary scratch scripts outside the repo.

## 6. Same failure location in every formulation

With the terminal weights in place, every variant fails at **s ≈ 590–601 m at about 25 m/s**:
- the progress reward, with or without smoother acceleration/jerk weights;
- soft or hard lateral acceleration;
- the convex speed term `(vx − 25)²` (suggested as a convex substitute for the progress reward).

That is where the corner sequence at s ≈ 650–700 m (radius 53 → 31 m) comes into the 3.04 s horizon (about 70 m at 25 m/s). So the cost formulation is probably not the cause.

Hypothesis (not tested):
- **The required braking:** slowing from 25 m/s to the lateral limit of that corner (≈ 14 m/s at a 31–34 m radius) needs about 61 m at the 3.5 m/s² braking limit.
- **What that does to the plan:** when the corner enters the horizon, the plan is right at the edge of what's physically possible.

**Tested, and refuted:** with a 20 m/s speed limit or a longer horizon (N = 50), the solver fails *earlier* (s = 316 m / 529 m), abruptly, on gentle corners, with 4–7 iterations in the steps before.

The failing plan at step 326 is very tightly constrained, but legitimate:
- lateral acceleration at the limit for 22 consecutive nodes;
- braking at its limit at the same time;
- speed at its limit early in the horizon;
- steering rate at its limit near both ends;
- the car running about 10 m wide.

That made that failure slow, but it isn't the general cause.

**The common pattern** is that every failing solve ends with a stationarity residual of 2e-5 to 1.3e-4, just above the 1e-5 tolerance: a residual floor rather than divergence. Tight QP tolerances lower that floor (section 7).

## 7. First full lap: QP tolerance 1e-8 and `tol_stat` 1e-4

Settings:
- the current code (progress reward, terminal weights 10 on vy/r/δ, `MIRROR`);
- `qp_solver_tol_{stat,eq,ineq,comp} = 1e-8`, `qp_solver_iter_max = 200`;
- `nlp_solver_tol_stat = 1e-4`.

Result: **N = 38 drove 3500 steps (6347 m, 1.4 laps) with no solver failure.**

| | |
|---|---|
| Lap time | **194.5 s** |
| Mean / max speed | 22.7 / 25 m/s |
| Failures / recovered solves | 0 / 0 |
| Track violations, closest approach to an edge | 0, 1.43 m |
| Max lateral error, max slack | 0.36 m, 4e-12 |
| Iterations per solve | 2–4: 2344 · 5–9: 598 · 10–19: 365 · ≥ 20 (max 62): 193 |
| Solve time mean / p95 / max | 27 / 83 / 245 ms (MPC step 80 ms) |

- **Without the tight QP tolerances,** `tol_stat` 1e-4 alone failed at s = 1448 m. With them but with `tol_stat` 1e-5, the run failed at s = 1411 m (residual 1.27e-5). Both changes are needed.
- **This is fine for Steps 4–5** (simulation, not real time).
- **Still open:** about 5 % of solves exceed the 80 ms step, and some need 20–62 iterations. Whether `tol_stat` 1e-4 is accurate enough for Leap-C sensitivities is also still to be checked (Step 8).

## 8. Final settings: soft lateral acceleration and `tol_stat` 1e-3

Two more failures showed up on a second lap with the section 7 settings:

1. **QP failure (status 4) at the first SQP iteration**, lap 2, lap position 3841 m, braking at a
   lateral acceleration of 5.66 m/s².
   - Re-solving that exact state: with a hard lateral limit the QP fails again; with a **soft** limit
     (L1 1e3, L2 10) it solves in 13 iterations with **zero slack**.
   - So the problem is feasible. The SQP's *linearised* lateral-acceleration constraint makes the first
     QP infeasible near the limit, and a soft constraint keeps the QP feasible.
   - → Lateral acceleration is now soft by default.
2. **Iteration limit with residual 1.2e-4**, lap 2, lap position 749 m (approaching the hairpin), with
   `tol_stat` 1e-4. This is the same residual floor as in section 6; 1e-4 is not a reliable margin.
   - → `nlp_solver_tol_stat = 1e-3`.

**Result:** run `results/Silverstone/N38/20261120-202857-508420`, N = 38, 10 000 steps, **4.15 laps with
0 solver failures**.

| | |
|---|---|
| Lap times | 194.64 s (standing start), 192.40 s, 192.48 s |
| Iterations | mean 4.4, max 50 |
| Solve time mean / p95 / max | 21 / 56 / 215 ms (1.7 % over the 80 ms step) |
| Track | closest approach to an edge 1.37 m, max slack 0.004 |

**N = 10 with the same settings** fails exactly as before: step 397, s = 705 m, after braking 19 m late
and running 10 m wide; residual 3.9e-3. That is the Step 5 result.

**Remaining caveats:**
- **Real time:** 1.7 % of N = 38 solves exceed 80 ms.
- **Leap-C accuracy:** whether `tol_stat` 1e-3 is accurate enough for Leap-C sensitivities is to be
  checked in Step 8.
- **Recovery from large deviations:** N = 10 cannot be solved to the tolerance in the recovery
  manoeuvre at s ≈ 705 m.

## 9. Options to discuss (status after section 8)

Options 1 and 2 below are adopted: terminal weights, plus the tolerance and soft-constraint settings
instead of accepting failed solves. The others remain relevant for real-time performance and for
Phase 3.




1. **Keep the two confirmed changes:**
   - terminal weights on vy, r and δ (fixes effect 1);
   - smoother acceleration/jerk weights (helps effect 2).

   Both change the cost, so they need agreeing with D5/D6 (stage weights are owned by the Optuna tuning). Hand-picked reasonable values are allowed for Phase 1.
2. **Accept nearly converged solves:** when the solver stops at the iteration limit with a small residual (e.g. below 1e-3), apply its first control and log it instead of stopping. This is common in real-time MPC and unblocks Steps 4–5. However, Leap-C (Phase 3) needs converged solves for valid sensitivities, so it is not a fix for Phase 3.
3. **A smarter regularisation strategy:** the exact Hessian works near the solution, and `MIRROR` is needed far from it. A better switching rule than the hybrid above might work, for example deciding by residual instead of trying exact first. Open-ended.
4. **Reformulate the progress reward:** for example a terminal-only reward `−q·s_N` (Mayer form), or a time-optimal formulation in `s`. Not yet tested.
5. **Ask someone who knows acados well** (supervisor, or the acados forum). The minimal case is the saved step-323 problem: `MIRROR` takes 134 iterations, `NO_REGULARIZE` takes 9.

Option 2 would unblock Steps 4–5 (N = 38 full lap, N = 10 comparison), but the question has to be settled before ROADMAP Step 8 (the Leap-C OCP).
