# Problem specification: learning the terminal cost of a short-horizon MPC with RL

**Status:** DRAFT for group discussion (2026-11-21, §3 revised 2026-11-22 and 2026-11-23). On 2026-11-24
David decided the reward, penalty, time limit, split, validation score and θ bounds (§10); the other items
are still open.
- Items marked **Planned** are the current working plan, still to be confirmed by the group.
- Items marked **Proposal** are suggestions.
- Items marked **Decide** need a group decision.

When agreed, this file is the reference for ROADMAP Steps 8–16.

Related files:
- `ROADMAP.md`: plan and status (Phase 1 and Step 6 are done).
- `REFACTOR_SUMMARY.md`: code structure.
- `STEP3_SOLVER_ISSUES.md`: solver findings that shape several choices below.

---

## 0. Summary

| Item | Proposal |
|---|---|
| What is learned (θ) | **Planned:** the supervisor's `V_f = x̃ᵀ P x̃` with `x̃ = [ey, epsi, vx, vy, r, δ, a]`; a diagonal P first (stage A, 7 log-weights), a full matrix P later (stage B, 28 parameters) |
| Who outputs θ | A neural network (SAC actor) that sees the car state and the road ahead |
| Observation | 7 Frenet states (no absolute `s`) + road preview 10–200 m ahead (curvature, edge distances), 67 numbers |
| Reward per step (0.08 s) | normalised progress − tracking error − constraint violation |
| Episode ends | **terminated:** off track, spun, non-finite state, MPC failure · **truncated:** time limit |
| Training data | Silverstone sections; held-out test section with two tight corners |
| Algorithms | SAC-ZOP first (no MPC gradients), then SAC-FOP (gradients through the MPC) |
| Main comparison | N = 10 learned vs. N = 38 benchmark vs. N = 10 default, on lap time, failures, solve time |

---

## 1. The problem in one page

**Setup** (Phase 1, all working and recorded in `ROADMAP.md`):
- **Vehicle:** TUM-CONTROL's EDGAR model, a dynamic single-track model with Pacejka tyres, in Frenet
  coordinates.
- **Track:** the F1TENTH Silverstone raceline scaled ×10, so a 4462 m lap.
- **Plant:** the TUM simulator at 0.02 s.
- **MPC:** acados, step 0.08 s.
- **Cost:** a progress reward `−q_s·ṡ` (no reference speed).
- **Constraints:** soft track edges, soft lateral acceleration (5.89 m/s²), speed ≤ 25 m/s, actuator
  limits.

**Results with identical settings** (updated 2026-11-23: the default terminal cost now has `vx` and `a`
weights of 0.01, the initial P of §3):

| | N = 38 (3.04 s horizon) | N = 10 (0.8 s horizon) |
|---|---|---|
| Laps | 4 laps, 0 failures | 3 laps, 0 failures |
| Flying lap | **192.3–192.6 s** | **236.0–236.1 s** (cruises at about 20 m/s) |
| Solve time mean / p95 | 21 / 56 ms (earlier measurement) | **5 / 7 ms** (earlier measurement) |

So N = 10 is now safe but about 23 % slower. Without the `vx` term (the earlier default), it drives
faster but fails. That earlier behaviour is described below.

**Why N = 10 failed without the `vx` term:**
- At 25 m/s it sees only about 20 m ahead.
- For the corner at s ≈ 650–700 m (radius 48 → 31 m) it starts braking at s = 642 m; N = 38 starts at
  623 m.
- It arrives 3–4 m/s too fast and runs 10 m wide (still on track), and the solver then fails during the
  recovery.

**Idea:** the *terminal cost* `V_f(x_N)` is the MPC's estimate of "how good is the state at the end of
the horizon". With a better terminal cost, N = 10 can behave as if it saw further: for example, a high
terminal speed should be penalised when a tight corner follows just after the horizon. We **learn** that
terminal cost with **reinforcement learning (RL)**, so the short MPC gets close to N = 38's lap time
while keeping its 5 ms solves.

**Fixed decisions** (ROADMAP §0, do not change without the group):
- **D1:** no imitation of N = 38.
- **D2:** no reference speed.
- **D3:** the car must stay inside the track.
- **D4:** RL with Leap-C, SAC-ZOP then SAC-FOP.
- **D5:** only the terminal cost is learned.
- **D6:** stage weights are tuned with Optuna by teammates.
- **D7:** build on the `latch13` modules.

---

## 2. How RL with an MPC works here (explainer)

**The MPC is the policy's last layer.** Ordinary RL has a neural network output the action directly.
Here the network outputs **MPC parameters θ** (the terminal-cost weights), and the MPC then computes the
action (jerk, steering rate) by solving its optimisation problem with those θ:

```
observation ──► actor network ──► θ ──► MPC (acados, N = 10) ──► action u0 ──► plant (TUM simulator)
   ▲                                                                              │
   └──────────────────────────── next observation, reward ◄─────────────────────────┘
```

The MPC keeps all constraints and the vehicle model, so the learned controller stays interpretable and
constraint-aware. It is not safe by construction: track edges and lateral acceleration are soft
constraints, and solves can fail. RL only shapes what the MPC aims for at the end of
its short horizon.

**SAC (Soft Actor-Critic):**
- The actor proposes θ with some randomness (exploration).
- Critic networks learn the value `Q` of state–θ pairs from experience.
- The actor is updated towards θ with higher `Q`.

**SAC-ZOP vs SAC-FOP** (both in `mpc-sac`):
- **ZOP (zero-order):** the MPC is treated as a black box. The critic learns directly how good each θ
  is. No derivatives of the MPC are needed, so it's robust and simple. Start here.
- **FOP (first-order):** the update back-propagates through the MPC, using its **sensitivities**
  `∂u0/∂θ` from Leap-C's `AcadosDiffMpcTorch`. Potentially more sample-efficient, but it needs accurate,
  converged solves.

**Leap-C's pieces we'll use:**

| Piece | Role |
|---|---|
| `AcadosParameterManager` | declares which OCP parameters exist, which are learnable (θ) and which are just data (curvature, edges) |
| `AcadosDiffMpcTorch` | the MPC as a PyTorch module (batched solves + sensitivities) |
| `ParameterizedPlanner` / `ControllerFromPlanner` (leap-c-lab) | wrap our OCP so `mpc-sac` can use it |
| `SacZopTrainer` / `SacFopTrainer` (mpc-sac) | the training loops |
| a `gymnasium.Env` (ours, ROADMAP Step 10) | the plant simulation, reward and termination |

The worked example in `leap-c-lab/leapc_lab/mass_spring_damper/` (planner, OCP, env) is our template;
it trained successfully in Step 6.

---

## 3. What is learned: terminal cost and θ

**Changed 2026-11-23 to the supervisor's form.** The earlier proposal (an overspeed term with a learned
threshold `v_θ`, and fixed `vy, r, δ` terms) is dropped.

**Phase 1 (fixed) terminal cost:**
`V_f = 40·ey_N² + 20·epsi_N² + 10·(vy_N² + r_N² + δ_N²)`, and since 2026-11-23 also
`+ 0.01·(vx_N² + a_N²)`, so that it equals the initial P below.
The `vy, r, δ` terms were added in Step 3 because without them the SQP drifted in a nearly flat lateral
direction at the end of the horizon (`STEP3_SOLVER_ISSUES.md`, effect 1).

**Planned:** learn a **diagonal P first**, and try a **full matrix P later in the project**.

**Form** (the same in both stages, only P changes):

```
V_f(x_N; θ) = x̃ᵀ P(θ) x̃
x̃ = [ ey_N, epsi_N, vx_N, vy_N, r_N, δ_N, a_N ]     (the terminal state without s)
```

### Stage A: diagonal P (first, implemented)

`P = diag(exp θ)`, θ = the 7 log-weights, all learnable and global (the same at all nodes of one solve).

| θ entry | Weight on | Bounds | Initial weight |
|---|---|---|---|
| `θ_1` | `ey_N²` | [1e-4, 1000] | 40 (as Phase 1) |
| `θ_2` | `epsi_N²` | [1e-4, 1000] | 20 (as Phase 1) |
| `θ_3` | `vx_N²` | [1e-4, 1000] | 0.01 (new) |
| `θ_4` | `vy_N²` | [1e-4, 1000] | 10 (as Phase 1) |
| `θ_5` | `r_N²` | [1e-4, 1000] | 10 (as Phase 1) |
| `θ_6` | `δ_N²` | [1e-4, 1000] | 10 (as Phase 1) |
| `θ_7` | `a_N²` | [1e-4, 1000] | 0.01 (new) |

- **Bounds:** the supervisor proposed [0.01, 1000]. The lower bound is 1e-4 because an initial value on a
  bound cannot be learned (see below).
- **Log-weights** keep P positive and let the weights span orders of magnitude.
- **The `vx` entry only pulls speed down.** There is no threshold, so any positive weight rewards a lower
  terminal speed. This is not a reference speed (D2), but it does slow the car down.
- **Measured effect of the initial θ** (closed loop, Leap-C, warm start of RL training, from the lap
  start at 10 m/s):

  | | Old default (no `vx`, `a` terms) | Initial θ (= default since 2026-11-23) |
  |---|---|---|
  | N = 10 | leaves the track / solver failure at s ≈ 764 m | **full lap, 238.9 s**, cruising at ~20 m/s |
  | N = 38 | 4 laps, 194.6 s first lap | 4 laps, 194.3 s first lap, still 25 m/s on straights |

  So the initial N = 10 controller is safe but about 23 % slower than N = 38.
  `QuadraticTerminalCost` in `ocp.py` now uses these weights too, so the non-learning default and the
  RL starting point are the same controller. The RL task becomes mainly
  "go faster without failing", rather than "stop failing".
- **Solver robustness across θ:** at the driving tolerance, 0 of 1440 solves with θ drawn at random
  inside the bounds failed, including very small `vy, r, δ` weights.

**Measured range of the `vx` weight (2026-11-23).**
- **Setup:** N = 10, the other six weights at their initial values, 250 steps (20 s), one run per weight.
- **Starts:** a straight (s = 1850 m, 20 m/s), before the hairpin (450 m, 25 m/s), in the hairpin
  (790 m, 11 m/s), before corners 5+6 (1250 m, 25 m/s) and before corners 12+13 (3950 m, 25 m/s).

The `vx` weight acts like a **cruise speed setting**: in every situation the car settles at the same
speed for a given weight.

| `vx` weight | Speed it settles at | Effect |
|---|---|---|
| 1e-4 to 0.003 | 25 m/s (limit) | identical results: the term has no effect |
| 0.005 to 0.007 | about 25 m/s | still too fast for the hairpin (and for 12+13 at 0.005); those runs end below 3 m/s |
| 0.01 (initial) | 20.5 m/s | lowest weight that survives all five situations |
| 0.015 | 14 m/s | |
| 0.02 | 10.8 m/s | about the slowest corner speed on the track (11 m/s) |
| 0.03 / 0.05 / 0.07 | 7.4 / 4.6 / 3.4 m/s | crawling |
| 0.1 and above | stops | ends below 3 m/s within about 7 s in every situation |

**Bounds [0.001, 0.05] for the `vx` weight** (proposed 2026-11-23, applied in the code 2026-11-24, still
to be confirmed by the group) (the other six keep [1e-4, 1000]; changing them
by a factor of 3 barely changed the result).
- **Lower bound 0.001:** everything below 0.003 behaves the same. 0.001 leaves some room below it, so
  "no speed term" is not on the bound.
- **Upper bound 0.05:** covers braking to below the slowest corner speed with margin. From about 0.1 the
  car stops, which can only end the episode.
- **Effect on exploration:** the log-range shrinks from 16 to 3.9, so far fewer explored θ simply stop
  the car. With [1e-4, 1000], 57 % of that range (0.1 and above) stops the car.
- **Depends on the stage cost:** the `vx` weight works against the progress reward (`q_s = 1`) and the
  stage weights. Repeat this sweep when the Optuna stage weights are frozen.
- **Relation to D2:** a weight that sets a cruise speed is close to choosing a speed. The road-dependent
  θ (Step 13) would choose it per situation. The group should confirm this is acceptable under D2.
- **Not tested:** other horizons, other combinations of the six other weights, more than one run per
  weight.
- **Repeated with the tuned N = 10 stage weights (2026-11-24):** the same pattern. 0.01 is again the
  lowest weight that survives all five situations (cruising at 21.9 m/s), 0.05 crawls (5.2 m/s), 0.1
  and above stop.

**Two settings are required to start at the initial θ** (measured with `mpc-sac` `d38726d`):
1. **Set `actor.residual = True`.** The default is `False`, which starts θ at the middle of its bounds:
   with these bounds that is a weight of 0.32 on every entry, not the initial P above.
2. **Keep every initial value strictly inside its bounds.** The actor squashes its output with `tanh`,
   so a value on a bound gets almost no gradient and no exploration. This is why the lower bound is 1e-4
   and not 0.01.

A freshly initialised network adds a small offset, so the start is close to the initial θ, not identical.
For the exact reproduction test in ROADMAP Step 9, set θ directly to the default weights
(`phase1_theta` in `controllers/leapc_fixed.py`, which since 2026-11-23 equals the initial θ).

**Constant or road-dependent** (both use a diagonal P):
- **Step 12, constant:** one P for the whole track. Set `actor.mlp.hidden_dims = None`, which makes the
  actor a single learned vector. Only removing the road preview is not enough, because θ would still
  vary with the car's state. A constant P cannot tell a straight from a hairpin.
- **Step 13, road-dependent:** the network outputs a new θ every 0.08 s from the state and road preview.

### Stage B: full matrix P (later)

```
P = L·Lᵀ,   L lower triangular (7 × 7),   L_ii = exp(l_i),   L_ij = l_ij for i > j
```

- **θ:** 7 diagonal entries `l_i` and 21 off-diagonal entries `l_ij`, so 28 parameters. The Cholesky
  form keeps P positive definite for any θ, which learning the matrix entries directly would not.
- **Start from the diagonal:** `l_i` = half the stage A log-weight and all `l_ij = 0` give exactly the
  same P.
- **Examples of what the new terms add:**

| Coupling | What it can express | Can it be constant? |
|---|---|---|
| `ey`–`epsi` | being off the line is less bad when already heading back to it | yes, it treats left and right the same |
| `ey`–`vx` | being fast is worse on one side of the raceline | no, its sign must flip between left and right corners |
| `epsi`–`vx` | being fast is worse when pointing one way | no, for the same reason |

- **Consequence:** couplings like these need a road-dependent θ, so stage B makes most sense after
  Step 13 works.
- **Comparison:** same seeds, reward and track split as stage A. Choose between A and B on the validation
  section, not the test section.
- **Costs and risks:** the actor and critic change size, so stage B needs new training. Monitor MPC
  failures and iteration counts.

### Implementation (Step 8)

- `controllers/leapc_ocp.py` writes the terminal cost as `x̃ᵀ L Lᵀ x̃` with `L = diag(exp(θ/2))`. Stage B
  then only fills the lower triangle of L (and rebuilds the solver).
- `AcadosParameterManager`: θ (`log_weights`, 7) is learnable and global; `[κ, ey_min, ey_max]` are
  non-learnable and per stage.

**Decide:**
- (a) Confirm the plan: diagonal first, full matrix (28 parameters) later.
- (b) Is a terminal `vx²` weight, which always favours lower speed, acceptable under D2?
- (c) The bounds: [0.001, 0.05] for the `vx` weight (measured above), [1e-4, 1000] for the others
  instead of the supervisor's [0.01, 1000], and the initial values.

---

## 4. Observation

**Proposal: 67 numbers.**
- **State (7):** `ey, epsi, vx, vy, r, δ, a`. The absolute progress `s` is **left out**: it grows without
  bound and would make the policy memorise the track position instead of reacting to the road.
- **Road preview (60):** curvature `κ`, `ey_min` and `ey_max` sampled every **Δ = 10 m** from **10 to
  200 m ahead** of the car.
  - **Why 200 m:** braking from 25 m/s to the 11 m/s of the slowest corner at 3.5 m/s² takes about
    72 m. Over a 0.8 s horizon at 25 m/s (about 20 m), a decision about braking must be made roughly 90 m
    before the apex, so 200 m gives margin.
  - **Distance-based, not time-based:** a fixed distance grid means the same road always looks the same,
    whatever the speed.

**Bounds** (needed by `mpc-sac`'s `scaling` extractor, which maps each entry to [0, 1]):

| | Bounds |
|---|---|
| `ey` | [−22, 22] m |
| `epsi` | [−π/2, π/2] |
| `vx` | [0, 26] m/s |
| `vy` | [−5, 5] m/s |
| `r` | [−2, 2] rad/s |
| `δ` | [−0.62, 0.62] rad |
| `a` | [−3.6, 2.6] m/s² |
| `κ` | [−0.05, 0.05] 1/m (Silverstone max 0.042) |
| `ey_min` / `ey_max` | [−22, 0] / [0, 22] m |

**Correction needed (found in Step 9): these 67 numbers are enough for the network, not for the MPC.**
The planner receives only the observation, so it must also find the MPC's road data inside the horizon
there. Interpolating the 10 m grid gives up to 2.7 m edge error, against a 0.3 m safety margin. Add a
dense section from 0 m to just beyond the horizon (about 25 m for N = 10). Even a 2 m grid still has
0.31 m, so use 1 m or pass the raceline's own vertices. The absolute `s` can stay out, because the
dynamics don't depend on it (the planner can use s = 0 at the car).

**Decide:**
- (a) Δ and preview length.
- (b) Include `ey_min/ey_max` (the raceline position on the track) or only `κ`.
- (c) Also give the actor the previous θ (smoother θ over time)?

---

## 5. Reward

**Proposal, per environment step (Δt = 0.08 s):**

```
r = w_p · Δs / (v_max·Δt)                ← progress, normalised to [0, 1] (Δs: progress this step, v_max = 25 m/s)
  − w_e · (ey² + c·epsi²) · Δt           ← tracking error
  − w_c · (constraint violation)         ← e.g. lateral-acceleration excess in the plant, edge proximity
```

- **Changed 2026-11-24 (David): `w_e = 1.0`** instead of 0.1. With 0.1, speed outweighed tracking about
  15:1 and the Step 13 policies drove up to 10 m off the raceline.
- **Start values:** `w_p = 1`, `w_e = 0.1` (1 m of error for 1 s costs 0.1, small next to progress), `c`
  chosen so 0.1 rad counts like about 1 m, `w_c = 1` per m/s² of excess.
- **Discount:** `γ = 0.99` per step means an effective horizon of about 1/(1−γ) = 100 steps ≈ 8 s. That
  is long enough to "see" that braking now pays off in the next corner.
- **Objective:** the teacher's objective (drive fast, low tracking error, stay on track) is expressed
  here, not in the MPC cost. ROADMAP D6 originally asked this reward to match the performance measure of
  the Optuna stage-weight tuning.
- **The Optuna measure (chosen by David, 2026-11-24)** is lap time from the standing start + 50 s per
  metre of RMS ey, with unfinished laps scored worse than any finished one. It expresses the same aims
  (fast, accurate, on track) but is not the same number as this reward.

**Validation and test score (proposed 2026-11-22, not yet agreed; implemented as a placeholder):**
finishing a section adds `w_p ·` (steps left before the time limit), i.e. the time saved counts as
progress at full speed. Without it the summed reward over a fixed distance barely depends on time.
The training reward has no such term.

**Decide:**
- (a) The weights, and whether the different Optuna measure is acceptable.
- (b) Whether tracking error is measured to the raceline (current) or not at all (pure lap time).

---

## 6. Episode ending: termination vs truncation

This distinction matters for RL:
- **Termination:** the episode ends because of the agent (a failure). There is no future value, and the
  critic learns that this state is final.
- **Truncation:** the episode is cut for practical reasons (time limit). The future still counts, and it
  must **not** be marked as terminated, or the critic learns wrong values.

**Proposal:**

| Event | Type | Penalty |
|---|---|---|
| Vehicle body crosses a track edge (plant margin < 0; the `left_track` check in `run_baseline.py`) | terminated | −P |
| `|epsi| > π/2` (spun / going backwards) | terminated | −P |
| Non-finite plant state | terminated | −P |
| MPC failure after the cold-restart retry | terminated | −P |
| Time limit (30 s = 375 steps, decided 2026-11-24) or end of the episode's track section | **truncated** | none |

- **P = 100** (decided 2026-11-24; first proposal 10).
  - Continuing an episode is worth about 85 in discounted reward (about 0.85 per step, γ = 0.99), so
    P = 10 added little to a crash.
  - The pilot (ROADMAP Step 12) traded crashes for speed: training success fell from 100 % to 78 %.
- **The MPC failure case matters most early on:**
  - In Phase 1, N = 10 never left the track; it failed by **MPC failure** at s ≈ 705 m after braking too
    late.
  - So early in training most episodes will end by MPC failure in the first braking zone, and the
    penalty for that is the main learning signal.
- **Leap-C detail:** sensitivities are only valid at converged solves. `mpc-sac` reports
  `masked_samples_perc` (failed solves excluded from the update). This should stay low.

---

## 7. Track split and episode starts

**Silverstone corners that need braking** (radius < 150 m; N = 38 minimum speed; everything else is
taken at 25 m/s):

| Corner | s [m] | Min radius | Direction | N = 38 min speed |
|---|---|---|---|---|
| 3 + 4 | 631–943 | 31 m, 24 m (hairpin) | right, left | 12.9, 11.0 m/s |
| 5 + 6 | 1435–1755 | 42 m, 28 m | left, right | 15.7, 12.4 m/s |
| 10 | 3010–3101 | 61 m | right | 18.0 m/s |
| 11 | 3772–3916 | 66 m | right | 19.3 m/s |
| 12 + 13 | 4140–4391 | 48 m, 46 m | left, right | 16.4, 15.3 m/s |

Only **five braking zones** per lap, so the split is a trade-off.

**Proposal (revised 2026-11-22, not yet agreed; implemented as a placeholder):**
- **Training:** 1850–3930 m (corners 10, 11) and 4430–1095 m over the lap start (corners 3+4). It ends
  early enough that the 200 m preview never reaches a held-out corner.
- **Validation** (checkpoint selection): corners 12+13, s = 3930–4430 m. The first draft used corner 10,
  where untrained N = 10 already never fails, so it could not tell controllers apart.
- **Test** (held out, never trained on): corners 5+6, s = 1300–1850 m. These are two tight corners in a
  row, a hard case not seen in training.
- **Full laps** are reported as well, clearly labelled as including training sections.
- **Alternative (Decide):** use another F1TENTH track as the test track. That gives cleaner
  generalisation, at the cost of a second track setup.

**Episode start states:**
- `s`: uniform in the training sections, 100–200 m before a braking zone or anywhere.
- `ey ~ U(±0.5 m)`, `epsi ~ U(±0.05 rad)`.
- `vx`: the N = 38 benchmark speed at that `s`, ± 2 m/s.

Using N = 38's *speed profile* only to choose realistic starting speeds is not imitation (nothing is
trained on N = 38's actions), but **confirm with the group that this is compatible with D1**. Alternative:
start at a random speed in [10, 25] m/s.

---

## 8. Evaluation protocol (ROADMAP Step 15)

**Controllers:**
1. N = 10 learned: SAC-ZOP constant θ (Step 12), SAC-ZOP road-dependent θ (Step 13), SAC-FOP (Step 14).
2. N = 38 benchmark (default terminal cost).
3. N = 10 default terminal cost (the Step 5 result; since 2026-11-23 the safe, slow controller with
   `vx` weight 0.01, i.e. the RL starting point).

**Settings:** identical MPC settings for all; ≥ 3 training seeds per learned controller; mean ± std.

**Metrics:**
- lap time, or distance reached before failure;
- termination type;
- RMS / max `ey`, RMS `epsi`;
- track violations (count, worst margin), slack use;
- MPC failures;
- solve time mean / p95 / max, and the fraction of solves above the 80 ms step.

**Where:** the held-out test section (many start states) and full laps from the standing start.

---

## 9. Known risks from Phase 1

1. **Solver robustness.** N = 38 needed specific settings to finish laps (`STEP3_SOLVER_ISSUES.md`):
   `MIRROR` regularisation, terminal `vy/r/δ` weights, QP tolerance 1e-8, soft lateral acceleration,
   stationarity tolerance 1e-3. RL will run millions of MPC solves from unusual states; MPC failures
   will be frequent early and must be handled cleanly (termination + penalty).
2. **Sensitivity accuracy (SAC-FOP), measured in Step 8.**
   - At the driving tolerance (1e-3), 4 % of the nonzero `∂u0/∂θ` are off by more than 10 %. FOP
     therefore needs the tight solver preset, at about 2.2× the solve time.
   - Often `∂u0/∂θ` is zero because the action sits at a limit. With the 7-weight θ (2026-11-23), the
     jerk gradient is nonzero in 77–88 and the steering gradient in 30–61 of 240 sampled problems per
     seed. ZOP does not depend on these gradients.
   - The driving-tolerance numbers above were measured with the earlier θ and not rechecked.
3. **Leap-C solver settings.** The roadmap suggested settings from the old integration (discrete RK4,
   `MIRROR`, `globalization_alpha_min = 0.2`). Phase 1 found the funnel line search works better here.
   Step 8 therefore uses the Phase 1 settings.
4. **Model = plant.** The MPC model and the TUM plant use identical equations and parameters, so results
   are optimistic. Optionally evaluate later with perturbed plant tyre parameters.
5. **Real time.** Even the benchmark N = 38 exceeds 80 ms in 1.7 % of solves; N = 10 is about 5× faster.
   Report timing, but the main claim is about closed-loop performance.

---

## 10. Open questions for the group (checklist)

- [ ] θ: confirm the supervisor's form, diagonal P first (7 log-weights) and full matrix P later (28)?
- [x] Is a terminal `vx²` weight, which always favours lower speed, acceptable under D2? **Yes**
  (2026-11-24).
- [x] Bounds: `vx` weight [0.001, 0.05], the others [1e-4, 1000], initial P as in §3. **Decided**
  (2026-11-24).
- [ ] Observation: drop `s`? Preview 10–200 m every 10 m? Include edge distances?
- [x] Reward weights `w_p = 1, w_e = 0.1, c = 100, w_c = 1` as in §5. **Kept** (2026-11-24); the Optuna
  measure differs (lap time + 50 s/m RMS ey), which is accepted.
- [x] Termination penalty P and time limit: **P = 100, 30 s** (2026-11-24).
- [x] Track split as in §7 (revised 2026-11-22). **Decided** (2026-11-24); no second track for now.
- [x] Validation/test score with the time bonus (§5). **Decided** (2026-11-24).
- [ ] Start speeds from the N = 38 profile: compatible with D1?
- [ ] Evaluation: number of seeds, test protocol?

---

## Glossary

| Term | Meaning |
|---|---|
| Frenet coordinates | position along the raceline `s`, lateral offset `ey` (left positive), heading error `epsi` |
| Horizon N | number of MPC steps (0.08 s each): N = 10 → 0.8 s, N = 38 → 3.04 s |
| Terminal cost `V_f` | cost on the last predicted state; stands in for everything after the horizon |
| θ | learnable MPC parameters (here: terminal-cost parameters) |
| SQP | the iterative method acados uses to solve the MPC optimisation |
| Slack | how much a soft constraint is violated in the plan (≈ 0 in all Phase 1 runs) |
| SAC | Soft Actor-Critic, the RL algorithm |
| ZOP / FOP | zero-order (no MPC derivatives) / first-order (back-propagate through the MPC) |
| Sensitivities | derivatives of the MPC solution w.r.t. θ, needed by FOP |
| Termination / truncation | episode ends because of a failure / because of a time or section limit |
| Extractor | preprocessing of observations; `scaling` maps each entry to [0, 1] using the bounds |
