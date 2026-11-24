"""Tune the MPC stage-cost weights with Optuna, separately per horizon (ROADMAP D6).

Each trial builds the Phase 1 MPC with sampled stage weights, drives laps from
a standing start at s = 0 and scores them. With ``--objective robust``
(default) it drives

* 2 laps from 10 m/s with the Phase 1 warm start (shifted), and
* 2 laps from 15 m/s with the warm start of Leap-C and RL training (unshifted),

    score = mean over the 4 laps of (lap time [s] + tracking_weight * RMS ey of its run [m])
            if all 4 laps are finished,
    score = 1000 + metres short, summed over both runs, otherwise.

``--objective lap1`` is the first study's objective: lap 1 from 10 m/s, shifted
warm start only. Its best N = 10 weights failed in lap 3 and with the unshifted
warm start, which is why the robust objective exists.

Lower is better. The progress weight stays at 1: scaling all stage weights
together barely changes the plan, so it would only add a redundant direction.
The terminal cost stays at its default (the initial P that RL starts from).
The objective and the search ranges are placeholders until the group agrees on
them (docs/PROBLEM_SPEC.md section 10).

    python scripts/tune_stage_weights.py --N 38 --trials 100 --workers 6
    python scripts/tune_stage_weights.py --N 10 --trials 100 --workers 6

Running it again with the same --study adds trials to that study. Results go to
results/optuna/<study>/: trials.csv and best_stage_weights.json, which
``run_baseline.py`` and ``train_rl.py`` read with --stage-weights. Run with the
venv activated.
"""

import argparse
import json
import multiprocessing
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TUM-CONTROL"))

from latch13.controllers.ocp import StageCostWeights  # noqa: E402
from latch13.evaluation.lap import drive_lap  # noqa: E402
from latch13.tracks.f1tenth import load_track  # noqa: E402

TUM_CONFIG = ROOT / "external" / "TUM-CONTROL" / "Config" / "EDGAR"
TRACK_ROOT = ROOT / "external" / "f1tenth_racetracks"
# Log-uniform search ranges: about 1/100 to 10 times the hand-picked values of Phase 1.
SEARCH_SPACE = {
    "ey": (0.1, 200.0),
    "epsi": (0.1, 100.0),
    "vy": (1e-3, 1.0),
    "yaw_rate": (1e-3, 1.0),
    "acceleration": (5e-4, 0.5),
    "jerk": (1e-4, 0.1),
    "steering_rate": (5e-4, 0.5),
}
UNFINISHED = 1000.0  # score offset of a lap that is not finished
# (warm start, start speed [m/s], laps) driven per trial
OBJECTIVES = {
    "lap1": (("shifted", 10.0, 1),),
    "robust": (("shifted", 10.0, 2), ("unshifted", 15.0, 2)),
}


def score(results, runs, track_length, tracking_weight):
    if all(result.finished for result in results):
        laps = [time + tracking_weight * result.rms_ey for result in results for time in result.lap_times]
        return sum(laps) / len(laps)
    return UNFINISHED + sum(max(0.0, laps * track_length - result.distance)
                            for result, (_, _, laps) in zip(results, runs))


def objective(args, track):
    runs = OBJECTIVES[args.objective]

    def evaluate(trial):
        weights = StageCostWeights(progress=1.0, **{
            name: trial.suggest_float(name, low, high, log=True) for name, (low, high) in SEARCH_SPACE.items()
        })
        build = ROOT / "results" / "generated" / "optuna" / args.study / f"trial_{trial.number}"
        results = []
        try:
            for warm_start, v_init, laps in runs:
                results.append(drive_lap(TUM_CONFIG, track, args.N, build, v_init=v_init, laps=laps,
                                         warm_start=warm_start, stage_weights=weights))
                if not results[-1].finished:
                    break  # unfinished either way; skip the remaining runs
        finally:
            shutil.rmtree(build, ignore_errors=True)
        results += [None] * (len(runs) - len(results))
        for (warm_start, v_init, laps), result in zip(runs, results):
            prefix = f"{warm_start}_{v_init:g}_"
            for key, value in (asdict(result) if result else {"end": "not run"}).items():
                trial.set_user_attr(prefix + key, value)
        # The first run also under the plain names, for trials.csv and the summary.
        for key, value in asdict(results[0]).items():
            trial.set_user_attr(key, value)
        driven = [result for result in results if result is not None]
        if len(driven) < len(runs):  # a run failed: count the skipped runs as not driven at all
            missing = sum(laps * track.length for (_, _, laps) in runs[len(driven):])
            return score(driven, runs, track.length, args.tracking_weight) + missing
        return score(driven, runs, track.length, args.tracking_weight)
    return evaluate


def open_study(args, seed):
    folder = ROOT / "results" / "optuna" / args.study
    folder.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(folder / "journal.log")))
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=args.startup)
    return optuna.create_study(study_name=args.study, storage=storage, sampler=sampler,
                               direction="minimize", load_if_exists=True), folder


def worker(args, index, trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    track = load_track(TRACK_ROOT, args.track, args.scale)
    study, _ = open_study(args, args.seed + index)
    study.optimize(objective(args, track), n_trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--N", type=int, required=True, help="MPC horizon to tune for")
    parser.add_argument("--trials", type=int, default=100, help="trials added in this call (all workers)")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes")
    parser.add_argument("--study", help="study name; default stage_weights_N<N>")
    parser.add_argument("--tracking-weight", type=float, default=50.0,
                        help="seconds of lap time per metre of RMS ey (placeholder)")
    parser.add_argument("--startup", type=int, default=20, help="random trials before TPE takes over")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="robust",
                        help="robust: 2 laps shifted from 10 m/s + 2 laps unshifted from 15 m/s; "
                             "lap1: lap 1 from 10 m/s only (the first study)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track", default="Silverstone")
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()
    args.study = args.study or f"stage_weights_N{args.N}"

    study, folder = open_study(args, args.seed)
    if not study.trials:  # the hand-picked Phase 1 weights as the first trial, for reference
        study.enqueue_trial({name: getattr(StageCostWeights(), name) for name in SEARCH_SPACE})
    shares = [args.trials // args.workers + (i < args.trials % args.workers) for i in range(args.workers)]
    processes = [multiprocessing.Process(target=worker, args=(args, i, n)) for i, n in enumerate(shares) if n]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    study, _ = open_study(args, args.seed)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    best = study.best_trial
    weights = asdict(StageCostWeights(progress=1.0, **best.params))
    (folder / "best_stage_weights.json").write_text(json.dumps({
        "stage_weights": weights, "N": args.N, "score": best.value, "trial": best.number,
        "lap": best.user_attrs, "trials_completed": len(complete),
        "objective": {"name": args.objective, "runs": OBJECTIVES[args.objective],
                      "tracking_weight_s_per_m": args.tracking_weight, "unfinished_offset": UNFINISHED},
        "search_space": SEARCH_SPACE, "written": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    reference = study.trials[0]
    print(f"{args.study} ({args.objective}): {len(complete)} trials. Hand-picked weights: score "
          f"{reference.value:.2f}. Best: trial {best.number}, score {best.value:.2f}")
    for name, trial in (("hand-picked", reference), ("best", best)):
        runs = [f"{w} from {v:g} m/s: {trial.user_attrs.get(f'{w}_{v:g}_end')} "
                f"{trial.user_attrs.get(f'{w}_{v:g}_lap_times')}" for w, v, _ in OBJECTIVES[args.objective]]
        print(f"  {name}: " + "; ".join(runs))
    print(json.dumps(weights, indent=2))
    print(f"Saved {folder / 'best_stage_weights.json'}")


if __name__ == "__main__":
    main()
