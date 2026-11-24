# Trained terminal-cost policies (ROADMAP Steps 12 and 13)

The best checkpoint (by validation score) of each training run, with just the
files the scripts need: `run_config.json`, `val_log.csv` and `ckpts/<step>_pi.ckpt`.
The full runs (replay buffers, all checkpoints, logs) stay in `results/rl/`,
which is not in git.

| Folder | θ | Seed | Checkpoint | Full lap | Test section (corners 5+6) |
|---|---|---|---|---|---|
| `step12b_constant_seed0` | constant | 0 | 60 000 | 203.36 s | 29.92 s |
| `step12b_constant_seed1` | constant | 1 | 170 000 | 203.12 s | 29.20 s |
| `step12b_constant_seed2` | constant | 2 | 160 000 | fails at the hairpin (700 m) | 28.96 s |
| `step12b_constant_seed3` | constant | 3 | 60 000 | **200.80 s** | 28.96 s |
| `step13b_road_seed0` | from the road | 0 | 150 000 | 202.64 s | 28.88 s |
| `step13b_road_seed1` | from the road | 1 | 30 000 | 208.40 s | 28.64 s |
| `step13b_road_seed2` | from the road | 2 | 160 000 | 210.56 s | 32.56 s |
| `step13b_road_seed3` | from the road | 3 | 70 000 | fails at 1679 m | fails |
| *N = 38 default* | | | | *194.40 s* | *27.04 s* |
| *N = 10 default* | | | | *224.16 s* | *31.12 s* |

**Measurements:** one start each, 2026-11-24, with `scripts/compare.py`. A full lap starts at s = 0 m and
10 m/s; the test section starts at s = 1300 m and 20 m/s.

**Training settings:** all runs used 200 000 steps, tracking weight 1.0, failure penalty 100, and
validation on corners 12+13 and the hairpin. The stage weights were the frozen Optuna ones. Details and
discussion: `ROADMAP.md`, Steps 12 and 13.

## Compare them

```bash
source .venv/bin/activate

# Full lap, live: N = 38, N = 10 default, then the learned controllers one after another
python scripts/compare.py --start lap --live \
    --learned policies/step12b_constant_seed3 policies/step13b_road_seed0

# Held-out test section, with TUM's Mode 2 panels
python scripts/compare.py --live 2 \
    --learned policies/step12b_constant_seed3 policies/step13b_road_seed0

# All eight, without the live view (table and plot only)
python scripts/compare.py --start lap --learned policies/step1*
```

Results (plot, table, one CSV per controller) go to `results/compare/<time>/`.

**First run:** this compiles the solvers, which takes a few minutes.

**Constant-θ policies** (`step12b_*`) also run in the normal runner:

```bash
python scripts/run_baseline.py --N 10 --steps 3000 --controller leapc-unshifted \
    --learned policies/step12b_constant_seed3 --live
```
