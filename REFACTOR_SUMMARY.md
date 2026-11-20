# Refactor summary: `latch13` (clean-architecture rewrite)

Status as of 2026-11-19. Repo: `~/Refactor`, branch `refactor/clean-architecture` (main branch: `main`).

This file replaces `PROJECT_SUMMARY.md` as the description of the **current code**. `PROJECT_SUMMARY.md` describes the earlier `latch_leapc` integration in `~/LATCH`. Its project aim, research questions and learning results still apply, but its code layout, commands and numbers do not apply to this repo.

---

## 1. What this repo is

This repo is a clean rebuild of the project's controller stack. TUM-CONTROL's vehicle model and simulator are reused **unchanged** (as a git submodule), and a thin layer of our own code turns the model into a **Frenet-coordinate NMPC** that tracks an F1TENTH raceline.

**Leap-C is not integrated yet.** The next step is to redo the Leap-C integration on top of this code (see §7).

The repo started as a fork of TUM-CONTROL, which is why the git history and `README.md` are still TUM's. The refactor commits are:

| Commit | Change |
|---|---|
| `16b988f` | TUM-CONTROL added as a submodule, Conda env |
| `3a726b2` | Removed duplicated TUM-CONTROL and legacy Frenet code |
| `28b0423` | F1TENTH racetracks added (now a submodule) |
| `5d0a46b` | `src/latch13` package; imports TUM model dynamics and simulator from the submodule |
| `41a9834` | Frenet model and OCP rewritten; simulation saves data and plots only afterwards |

---

## 2. Layout

```
Refactor/
├── external/                      # git submodules (run `git submodule update --init`)
│   ├── TUM-CONTROL/               # bzarr/TUM-CONTROL: prediction model, plant, EDGAR config
│   └── f1tenth_racetracks/        # f1tenth/f1tenth_racetracks: raceline + centerline CSVs
├── src/latch13/                   # our package (pip install -e .)
│   ├── models/frenet_model.py     # TUM model → Frenet AcadosModel
│   ├── controllers/ocp.py         # OCP definition + acados solver build
│   ├── tracks/f1tenth.py          # Raceline dataclass, load_raceline, project_pose
│   ├── visualization/offline.py   # post-run viewer (slider over saved predictions)
│   ├── visualization/live.py      # optional live view (--live 1|2) using TUM-CONTROL Mode 1/2 plotting
│   ├── learning/  evaluation/  simulation/   # empty placeholders
├── scripts/
│   ├── run_baseline.py            # closed-loop MPC + TUM plant simulation, the main entry point
│   └── check_track.py             # sanity check of track loading/projection
├── results/<track>/N<N>/<timestamp>/   # one directory per run
├── pyproject.toml                 # package "latch13", Python 3.11
└── environment.yml                # Conda env "latch13"
```

TUM-CONTROL is imported through its own top-level module names (`Prediction_Models`, `Vehicle_Simulator`, `Utils`). `run_baseline.py` puts `external/TUM-CONTROL` on `sys.path`. EDGAR parameters come from `external/TUM-CONTROL/Config/EDGAR/` (`veh_params_pred.yaml`, `pacejka_params.yaml`, `sim_main_params.yaml`).

---

## 3. Model: `models/frenet_model.py`

`build_frenet_model(vehicle_yaml, tire_yaml) -> AcadosModel`

- Calls TUM's `pred_stm` (dynamic single-track model with Pacejka tyres) and **reuses its physical derivatives** for `vx, vy, r, delta, a`. Only the pose kinematics are replaced with Frenet ones:
  - `s_dot = (vx cos epsi − vy sin epsi) / (1 − κ ey)`
  - `ey_dot = vx sin epsi + vy cos epsi`
  - `epsi_dot = r − κ s_dot`
- **State** `x = [s, ey, epsi, vx, vy, r, delta, a]` (8)
- **Input** `u = [jerk, steering_rate]` (2, TUM order and symbols)
- **Parameter** `p = [κ]` (track curvature, one per stage)
- TUM's numeric metadata is copied onto the model (bounds such as `jerk_min/max`, `acc_min/max`, `delta_f_min/max`, `delta_f_dot_min/max`). `model.constraint.alat(x)` exposes TUM's lateral-acceleration expression for the Frenet state.

---

## 4. OCP: `controllers/ocp.py`

- `build_ocp(config_dir, x0, N=38, dt=0.08, v_ref=8.0, *, stage_weights, terminal_cost, constraints, solver_options) -> AcadosOcp`. This builds the OCP without generating code.
- `build_mpc(...)` calls `build_ocp`, then generates and compiles the solver and returns an `AcadosOcpSolver`.

**Cost** (`EXTERNAL`, integrated by acados):
- Stage cost (`StageCostWeights`): `20 ey² + 10 epsi² + 1 (vx − v_ref)² + 0.1 vy² + 0.1 r² + 0.05 a² + 0.01 jerk² + 0.05 steering_rate²`
- Terminal cost: a pluggable `TerminalCost` protocol, `f(x, p, v_ref) -> scalar CasADi MX`.
  - Default `QuadraticTerminalCost`: `40 ey² + 20 epsi² + 2 (vx − v_ref)²` (twice the stage weights)
  - `zero_terminal_cost` turns the terminal cost off.
  - **Weights are baked in when the solver is generated.** Changing them requires rebuilding the solver. They are not acados parameters.
- `v_ref` is a **single constant** that is baked into the cost. There is no speed profile.

**Constraints:**
- Hard state bounds at nodes 1..N: `vx ∈ [1, 25]`, plus TUM's `delta` and `a` limits.
- Hard input bounds: TUM's jerk and steering-rate limits.
- Nonlinear constraint `h = a_lat(x)` within TUM's `lat_acc_min/max` at nodes 0..N. It can optionally be softened (`ConstraintOptions.lateral_slack_*`).
- **Missing compared with the old `latch_leapc`:** no track-boundary constraint on `ey` and no combined friction-circle (ggv) constraint.

**Solver** (`SolverOptions`): full SQP, exact Hessian, `PARTIAL_CONDENSING_HPIPM`, ERK (RK4 with 4 substeps per 0.08 s interval), `MERIT_BACKTRACKING`, Levenberg–Marquardt 1e-2, max 100 iterations, `tol_stat` 1e-5, other tolerances 1e-6.

---

## 5. Simulation: `scripts/run_baseline.py`

```bash
python scripts/run_baseline.py --track Silverstone --N 38 --speed 5 --scale 10 --steps 1000
python scripts/run_baseline.py --replay results/Silverstone/N38/<timestamp>   # view a saved run, no acados needed
python scripts/run_baseline.py --steps 2000 --live          # TUM Mode 1 live view (--live 2: + speed, deviation, g-g)
python scripts/run_baseline.py --steps 2000 --live --live-rate 1   # paced to real time
# other flags: --dt 0.08 --terminal-cost {quadratic,zero} --max-error 8 --plot-every 5 --no-plot --output DIR
```

- **Plant:** TUM's `PassengerVehicleSimulator`, a 7-state Cartesian model `[X, Y, yaw, vx, vy, r, delta]` with input `[a, steering_rate]` at `Ts = 0.02 s`. It is a separate model from the MPC's prediction model.
- **MPC step:** 0.08 s, so 4 plant substeps per solve. The MPC's jerk is integrated into an ideal acceleration state outside the plant, with the midpoint acceleration fed to each substep.
- **Each step:** fix `x0`, then set the per-stage curvature `κ(s0 + j·dt·v_ref)` (predicted progress at constant reference speed), then solve. If the solve fails, the script cold-restarts once from the same state. It then applies `u0`, steps the plant, projects the Cartesian state back to Frenet (`project`), unwraps `s` across laps and shifts the warm start.
- **Stopping:** the run stops on the step limit, a solver failure, `|ey| > max_error`, or Ctrl+C.
- **Outputs** per run: `config.json`, `track.npz`, `trajectory.csv`, `predictions.npy` (a stream of `(N+1)×11` snapshots), `diagnostics.jsonl`, `summary.json`, `solver/` (the generated acados code) and `overview.png`.
- **Track:** the F1TENTH raceline scaled by `--scale` (default 10), giving a Silverstone lap of **4462 m**. Curvature comes from finite differences with Gaussian smoothing. Boundaries come from the centerline CSV and are used only for plotting.
- **Duplicate track code:** `run_baseline.py` has its own track loader and projection (dict-based). It does not use `tracks/f1tenth.py`, which only `check_track.py` uses. These should be merged into one.

---

## 6. Results so far (N = 38 only)

The latest runs (`results/Silverstone/N38/20261118-124814-…` and `…-132112-…`) used N = 38, v_ref = 5 m/s, the quadratic terminal cost and 10 000 steps (800 s simulated):

| Metric | Value |
|---|---|
| Solver failures | 0 |
| RMS / max lateral error | 0.0038 m / 0.034 m |
| Mean / p95 / max solve time | 13.6 / 15.5 / 30.5 ms |
| Distance covered | ≈ 4000 m (< 1 lap of 4462 m, so `completed_laps = 0`) |

Earlier runs stopped with solver status 4 or 2 before the solver options were tuned.

**Not done yet:**
- No N = 10 run in this repo.
- No race-speed run. The old project ran at 12.8–37.5 m/s, and `speed_max` here is 25 m/s.
- No full lap.

---

## 7. Next step: redo the Leap-C integration here

Goal: as in `PROJECT_SUMMARY.md`, learn the terminal cost of a short-horizon MPC (N = 10) so that it imitates or approaches the long-horizon MPC (N = 38). This time the integration should build on the `latch13` modules above, not on a separate package.

Gaps between the current code and what Leap-C needs:

1. **Learnable parameters.** The terminal weights must become acados/Leap-C parameters, not constants baked in at code generation. For example, extend `p` from `[κ]` to `[κ, P_ey, P_epsi, P_v]` (or use Leap-C's parameter manager), so gradients flow and weights change without rebuilding the solver. The `TerminalCost` protocol is the natural place to plug this in.
2. **Speed reference.** `v_ref` is a constant baked into the cost. A per-stage `v_ref` parameter, like `κ`, is needed for a speed profile and for race-speed experiments.
3. **Missing constraints.** The old setup had soft track-boundary and friction-circle constraints. These need adding here for comparable results.
4. **Environment interface.** The closed loop in `run_baseline.py` (plant step, projection, curvature lookup, warm start) is script code. Leap-C needs it as a reusable environment or simulator API, probably in `src/latch13/simulation/`.
5. **Code layout.** `learning/` and `evaluation/` are empty and are where the training (teacher labelling, imitation loss, data split) and the comparison code will go.
6. **Housekeeping.**
   - Merge the duplicate track code.
   - Restore tests (`tests/test_frenet_model.py` was removed in `41a9834`, but `pyproject.toml` still points pytest at `tests/`).
   - Replace TUM's `README.md`.
   - Remove the stray `acados_ocp_SNMPC.json` at the repo root, which is TUM's generated file with `/home/baha/...` paths.
   - Decide whether `results/` and `src/latch13.egg-info/` belong in git.

---

## 8. Setup

```bash
git submodule update --init --recursive
conda env create -f environment.yml && conda activate latch13
pip install -e .                                   # installs latch13 from src/
pip install -e <acados_root>/interfaces/acados_template
export ACADOS_SOURCE_DIR=<acados_root>             # run_baseline.py can auto-discover and set LD_LIBRARY_PATH
```

Platform: WSL2 Ubuntu, Python 3.11.

**Dependency mismatch:** `requirements.txt` is TUM's old pinned list (`numpy==1.24.2`, PyQt), but `environment.yml` uses `numpy=1.26`. `pyproject.toml` declares no dependencies. Treat `environment.yml` as authoritative.
