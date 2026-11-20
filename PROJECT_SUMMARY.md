# Project summary: Learning-Based Terminal Costs for Short-Horizon MPC (SSY226, group 13)

Status as of 2026-11-19. Based on the planning report and `LATCH_LEAPC_ANALYSIS.md` (version 2026-11-18).

---

## 1. Aim

Investigate whether a short-horizon nonlinear MPC with a **learned terminal cost** can approach the closed-loop performance of a long-horizon MPC at lower computational cost, for autonomous trajectory tracking on a full-scale F1 track.

Comparison metrics: lap completion and lap time, tracking error (e_y, e_psi, Δv), constraint satisfaction (track bounds, friction circle) and solve time.

---

## 2. Implementation: `latch_leapc`

A Python package in a Leap-C checkout (`~/LATCH`) that combines:

| Component | Source |
|---|---|
| Vehicle plant simulator (Cartesian single-track, Pacejka tyres, EDGAR parameters) | TUM-CONTROL |
| Frenet prediction model (same force equations, Frenet pose kinematics) | Group (LATCH) |
| Cartesian ↔ Frenet projection, track preprocessing | Group (LATCH) |
| Differentiable MPC (acados + implicit differentiation) | Leap-C |
| Data collection, training, evaluation, live viewer | Group (`latch_leapc/`) |

**Track:** F1TENTH Silverstone scaled to full size (5891 m lap, v_ref 12.8–37.5 m/s).

**MPC formulation**
- State `[s, e_y, e_psi, v_x, v_y, psi_dot, delta, a]`, input `[jerk, delta_dot]`; plant at 0.02 s, MPC step 0.08 s.
- Stage cost: quadratic in `ξ = [e_y, e_psi, v_x − v_ref]` plus input penalties.
- Terminal cost: `P_terminal · ξ²` with `P = [P_ey, P_epsi, P_v]`, **the only learned quantity**.
- Soft constraints: friction circle (ggv) and track bounds.
- Solver: full SQP, exact Hessian (MIRROR regularisation), HPIPM, merit backtracking.

**Horizons:** long N = 38 (3.04 s, teacher/benchmark), short N = 10 (0.8 s, student).

**Learning method:** imitation learning. The N = 38 MPC labels sampled states with its first action; `P` for N = 10 is trained (Adam, `P = P0·exp(θ)`) so that its first action matches, with gradients through the solver. The data is split spatially along the lap: training, validation (groups 2, 7) and a held-out test section (group 9).

---

## 3. Results so far

**Solver robustness.** Three solver issues were found and fixed: line-search floor, Gauss-Newton Hessian stall, and stationarity tolerance. Before the fixes no controller drove more than about 13 s.

**Baselines (full lap from s = 0)**

| | N = 38 | N = 10 |
|---|---|---|
| Lap | **complete, 202.5 s** | spins out in the hairpin at s ≈ 914 m |
| RMS e_y | 0.56 m | 7.8 m (until failure) |
| Track violations | 0 | 122 samples |
| Mean solve time | 12–16 ms | 5 ms |
| Solves > 20 ms | 14–21 % | < 1 % |

N = 10 fails because it sees only about 25 m ahead at 30+ m/s, enters the hairpin at 26 m/s instead of 15 m/s and runs off the track.

**Learning (best run, 1024 samples, `multiplier_max` = 10000)**
- Learned P = [2.23, **1930**, 9.13] vs default [2.8, 0.4, 2.0]. The learned cost weights terminal heading error very heavily, forcing the short horizon to end aligned with the track.
- Validation loss 0.00785 → 0.00291. More data alone (128 → 1024) gave no improvement; raising the weight cap did.

**Closed loop, held-out test section (8 s)**

| | N = 38 | N = 10 fixed | N = 10 learned | Gap closed |
|---|---|---|---|---|
| RMS e_y | 2.16 m | 6.22 m | 4.67 m | 38 % |
| RMS e_psi | 0.090 rad | 0.383 rad | 0.244 rad | 47 % |
| RMS Δv | 2.69 m/s | 4.79 m/s | 3.49 m/s | 62 % |
| Off-track samples | 0 | 72 | 22 | |
| Mean solve time | 11.3 ms | 7.9 ms | 5.3 ms | |

On the full lap, the learned N = 10 gets through the s ≈ 914 m hairpin that plain N = 10 cannot, but leaves the track at s ≈ 1060 m and stops at s ≈ 1293 m.

**Conclusion so far:** learning three constant terminal weights gives a clear but partial improvement. The short MPC with the learned cost is still clearly worse than N = 38 and cannot complete a lap, at roughly half the solve time.

---

## 4. Deviations from the planning report

| Planning report | Current implementation | Comment |
|---|---|---|
| Learn V_f with reinforcement learning (reward on lap time and tracking) | Imitation of the long MPC's first action | Simpler and stable; RL or a value-based target is still open |
| Compare quadratic vs small neural network | Only a quadratic with 3 learned constant weights | The network variant is the natural next step |
| Long horizon chosen by analysis | N = 38 (TUM-CONTROL default 3.04 s) | Justified in practice: completes a full lap with no violations |
| Evaluate lap time | Full-lap results only for N = 38; test section limited to ~8 s | Evaluation needs extending |

---

## 5. Known limitations

1. **Constant terminal weights** cannot encode the road beyond the horizon ("hairpin ahead, brake now"). This is the most likely cause of the remaining gap.
2. **Distribution shift:** training states lie near the reference, but in closed loop the student reaches states (too fast, too wide) that the data barely covers.
3. **Objective mismatch:** only the first action is imitated, while performance is measured in closed loop.
4. **Thin evaluation:** one test window (~8 s), one seed, one training run per setting.
5. **Timing:** the controllers run full SQP, and N = 38 is not real-time (14–21 % of solves above 20 ms). These timings are not representative of an RTI deployment.
6. **Reproducibility:** any edit to the package invalidates data and weights (fingerprint), and `latch_leapc/` is not yet committed to a shared repository.

---

## 6. Proposed next steps

1. **Road-dependent terminal cost:** make P (or a terminal speed/heading target) a function of the road preview beyond the horizon via a small network. This corresponds to the planning report's "quadratic vs neural network" question and to V_θ(x_N, c_N).
2. **Value-function target:** fit `P·ξ²` at stage 10 to the cost N = 38 still pays over stages 10–38, as an alternative to action imitation.
3. **Closed-loop data (DAgger):** label the states the student actually reaches with teacher actions and retrain.
4. **Evaluation:** several seeds, several test windows and full laps; report mean ± std, lap time and constraint margins.
5. **Timing:** add an SQP_RTI variant for real-time comparison, keeping full SQP for training and gradients.
6. **Housekeeping:** commit `latch_leapc/` to a shared group repository.

---

## 7. Timeline status

| Week | Planned phase | Status |
|---|---|---|
| 46 (9 Nov) | Literature review, baseline MPC setup | Done: Frenet NMPC, full-lap N = 38 baseline |
| 47–48 | Leap-C integration | Largely done: differentiable MPC, training and evaluation pipeline working |
| 48–50 | Fine-tuning | Next: road-dependent terminal cost, value target, evaluation |
| 51 | Final report | Draft paper due 14 Dec |

---

## 8. How to run

```bash
cd ~/LATCH && source .venv/bin/activate
python -m latch_leapc all                                         # full pipeline
python -m latch_leapc baseline --horizon long --steps 10200       # full lap, N = 38
python -m latch_leapc compare --steps 400                         # held-out test section
python -m latch_leapc compare --start-fraction 0 --steps 10200    # full lap (not held out)
# add --live --live-every 25 to watch a run
```
Environment: WSL2 Ubuntu 22.04, Python 3.11 venv in `~/LATCH/.venv`, acados from `~/LATCH/external/acados`.
