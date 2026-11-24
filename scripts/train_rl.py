"""Train the terminal cost of the short-horizon MPC with SAC-ZOP (ROADMAP steps 11-13).

The actor outputs theta (7 terminal log-weights, docs/PROBLEM_SPEC.md section 3)
from the env observation, starting at the initial theta (residual actor); the
MPC turns theta into jerk and steering rate. Validation runs on the validation
split; the best checkpoint is kept.

    python scripts/train_rl.py --steps 200000            # road-dependent theta (Step 13)
    python scripts/train_rl.py --constant-theta          # one theta for the whole track (Step 12)

An existing --output directory continues from its last checkpoint. Run with the
venv activated. The env settings (reward, penalty, split) were decided on
2026-11-24 (docs/PROBLEM_SPEC.md section 10).
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from leapc_lab.planner import ControllerFromPlanner  # noqa: E402
from mpc_sac.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig  # noqa: E402

from latch13.controllers.ocp import default_stage_weights, load_stage_weights, weights_tag  # noqa: E402
from latch13.controllers.planner import RacingPlanner, RacingPlannerConfig  # noqa: E402
from latch13.learning.policy import actor_config  # noqa: E402
from latch13.simulation.env import EnvOptions, RacingEnv  # noqa: E402
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
SUBMODULES = ("leap-c", "leap-c-lab", "mpc-sac", "TUM-CONTROL", "f1tenth_racetracks")


def trainer_config(args):
    """SAC-ZOP settings; the values follow mpc-sac's example scripts."""
    cfg = SacZopTrainerConfig()
    cfg.seed = args.seed
    cfg.train_steps = args.steps
    cfg.train_start = 0
    cfg.val_freq = args.val_freq
    cfg.val_num_rollouts = args.val_rollouts
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 0
    cfg.val_render_mode = None
    cfg.val_report_score = "cum"
    # Every validation keeps its policy (ckpts/<step>_*.ckpt); pick the best with val_log.csv.
    # "best" would save nothing in a run that never beats its first validation.
    cfg.ckpt_modus = "all"
    cfg.batch_size = 64
    cfg.buffer_size = 1_000_000
    cfg.gamma = 0.99
    cfg.tau = 0.005
    cfg.lr_q = 1e-3
    cfg.lr_pi = 1e-3
    cfg.lr_alpha = 1e-3
    cfg.init_alpha = 0.02
    cfg.update_freq = 4

    cfg.log.verbose = True  # includes the MPC statistics (solving time, success and retry rate)
    cfg.log.interval = args.log_interval
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = True
    cfg.log.wandb_logger = False

    cfg.critic_mlp.hidden_dims = (256, 256, 256)
    # Defined in latch13.learning.policy, so checkpoints can be rebuilt by the other scripts.
    cfg.actor = actor_config(args.constant_theta)
    return cfg


def commit_of(repository):
    """Commit checked out in a repository, read from its files (no git command is run)."""
    git = repository / ".git"
    if git.is_file():  # a submodule: ".git" names the real git directory
        git = (repository / git.read_text().split("gitdir:", 1)[1].strip()).resolve()
    head = git / "HEAD"
    if not head.is_file():
        return "unknown"
    value = head.read_text().strip()
    if not value.startswith("ref:"):
        return value
    ref = value.split(None, 1)[1]
    if (git / ref).is_file():
        return (git / ref).read_text().strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return f"unknown ({ref})"


def source_versions():
    """Commits of this repository and the submodules; uncommitted changes are not recorded."""
    versions = {"Refactor": commit_of(ROOT)}
    versions.update({name: commit_of(ROOT / "external" / name) for name in SUBMODULES})
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--steps", type=int, default=200_000, help="training steps (0.08 s each)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=10, help="MPC horizon")
    parser.add_argument("--constant-theta", action="store_true", help="one theta for the whole track (Step 12)")
    parser.add_argument("--val-freq", type=int, default=10_000, help="validate every this many steps")
    parser.add_argument("--val-rollouts", type=int, default=10, help="validation episodes")
    parser.add_argument("--log-interval", type=int, default=1000, help="log every this many steps")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64",
                        help="networks and replay buffer; the MPC always solves in float64")
    parser.add_argument("--stage-weights", metavar="FILE",
                        help="stage weights from a JSON file (scripts/tune_stage_weights.py), or "
                             "'hand-picked'; default: the tuned weights of this horizon")
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--output", help="run directory; an existing one is continued")
    args = parser.parse_args()

    kind = "constant" if args.constant_theta else "road"
    output = Path(args.output).resolve() if args.output else (
        ROOT / "results" / "rl" / f"sac_zop_{kind}_N{args.N}" /
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    )
    continued = output.exists()
    output.mkdir(parents=True, exist_ok=True)

    track = load_track(TRACK_ROOT, args.track, args.scale)
    env_options = EnvOptions()
    # One row per episode: where it started and ended, how it ended.
    train_env = RacingEnv(TUM_CONFIG, track, mode="train", options=env_options,
                          episode_log=output / "episodes_train.csv")
    val_env = RacingEnv(TUM_CONFIG, track, mode="val", options=env_options,
                        episode_log=output / "episodes_val.csv")
    stage_weights = (load_stage_weights(args.stage_weights) if args.stage_weights
                     else default_stage_weights(args.N))
    planner_cfg = RacingPlannerConfig(N=args.N, dt=train_env.dt, near_vertices=env_options.near_vertices,
                                      stage_weights=stage_weights)
    # One compiled solver per horizon, reused across runs; acados rebuilds it when the OCP changes.
    code = f"N{args.N}_w{weights_tag(stage_weights)}"
    planner = RacingPlanner(TUM_CONFIG, ROOT / "results" / "generated" / "leapc" / "planner" / code, planner_cfg)
    cfg = trainer_config(args)

    dtype = getattr(torch, args.dtype)
    trainer = SacZopTrainer(
        cfg=cfg, val_env=val_env, output_path=output, device="cpu", dtype=dtype,
        train_env=train_env, controller=ControllerFromPlanner(planner), extractor_cls="scaling",
    )
    (output / "run_config.json").write_text(json.dumps({
        "args": vars(args), "trainer": asdict(cfg), "env": asdict(env_options),
        "planner": {**asdict(planner_cfg), "dtype": str(planner_cfg.dtype)},
        "track_length_m": track.length, "versions": source_versions(),
    }, indent=2, default=str), encoding="utf-8")
    saved = sorted(int(path.name.split("_")[0]) for path in output.glob("ckpts/*_trainer_state.ckpt"))
    if continued and saved:
        # mpc-sac names checkpoints by step and looks them up by its current step.
        trainer.state.step = saved[-1]
        trainer.load(output)
        # run() validates once more at the start and appends that score; without this, the
        # next scheduled validation (and checkpoint) would be skipped.
        trainer.state.scores.pop()
        print(f"Continuing {output} at step {trainer.state.step}")
    print(f"SAC-ZOP, {kind} theta, N={args.N}, {args.steps} steps -> {output}")
    score = trainer.run()
    print(f"Done. Validation score (sum over validations): {score:.1f}. Results: {output}")


if __name__ == "__main__":
    main()
