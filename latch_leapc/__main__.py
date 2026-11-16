"""Run from the patched Leap-C root: python -m latch_leapc COMMAND.

Use `python -m latch_leapc all` to run the whole pipeline:
check -> smoke -> baseline (short, long) -> collect -> train -> compare.
"""
import argparse

from .runtime import configure

PIPELINE = ["check", "smoke", "baseline_short", "baseline_long", "collect", "train", "compare"]


def run_baseline(exp, settings, horizon, a, viewer=None):
    # result = exp.rollout(settings, settings[horizon+"_horizon"], a.steps,
    #                      a.start_fraction or 0., verbose=a.verbose)
    # name = "baseline_"+horizon
    name = "baseline_"+horizon
    if viewer is not None:
        viewer.begin(name, settings[horizon+"_horizon"])
    result = exp.rollout(settings, settings[horizon+"_horizon"], a.steps,
                         a.start_fraction or 0., verbose=a.verbose, viewer=viewer)
    exp.save_rollout(settings, name, result)
    exp.plot_rollouts(settings, {name: result}, name+".png")
    return not result["metrics"]["failure"]


def run_compare(exp, settings, a, parser, viewer=None):
    learned = exp.load_weights(settings)
    # Default comparison starts within untouched spatial test group 9.
    fraction = .92 if a.start_fraction is None else a.start_fraction
    ref = exp.reference()
    if a.start_fraction is None:
        speed_max = ref.v_ref.max() + 5.
        max_distance = speed_max*(a.steps*settings["simulation_dt"]
                                  + settings["long_horizon"]*settings["prediction_dt"])
        if fraction*ref.L + max_distance >= ref.L:
            parser.error("Default test interval would cross into training data. Reduce --steps or choose an explicit --start-fraction.")
    results = {}
    for name, N, P in [("long_fixed", settings["long_horizon"], None),
                       ("short_fixed", settings["short_horizon"], None),
                       ("short_learned", settings["short_horizon"], learned)]:
        # result = exp.rollout(settings, N, a.steps, fraction, P, a.verbose)
        if viewer is not None:
            viewer.begin(name, N)
        result = exp.rollout(settings, N, a.steps, fraction, P, a.verbose, viewer=viewer)
        results[name] = result
        exp.save_rollout(settings, name, result)
    exp.plot_rollouts(settings, results, "comparison.png")
    return not any(x["metrics"]["failure"] for x in results.values())


def main():
    parser = argparse.ArgumentParser(description="LATCH-13 vehicle / Leap-C terminal-cost starter")
    parser.add_argument("command", choices=["all", "check", "smoke", "baseline", "collect", "train", "compare"])
    parser.add_argument("--steps", type=int, default=200, help="closed-loop simulation steps, each 0.02s")
    parser.add_argument("--horizon", choices=["short", "long"], default="short")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--start-fraction", type=float, default=None, help="track arc-length fraction in [0,1)")
    parser.add_argument("--verbose", action="store_true", help="print CMake/compiler output")
    parser.add_argument("--live", action="store_true",
                        help="show car, MPC prediction and track live during baseline/compare rollouts")
    parser.add_argument("--live-every", type=int, default=1,
                        help="with --live: redraw every k simulation steps (larger = faster)")
    a = parser.parse_args()
    if a.steps < 1 or a.samples is not None and a.samples < 20 or a.epochs is not None and a.epochs < 1:
        parser.error("steps/epochs must be positive; use at least 20 teacher samples.")
    if a.start_fraction is not None and not 0 <= a.start_fraction < 1:
        parser.error("--start-fraction must be in [0,1).")
    settings = configure(require_solver=a.command != "check")
    from . import experiments as exp
    viewer = None
    if a.live:
        if a.command not in ("all", "baseline", "compare"):
            parser.error("--live applies to baseline, compare and all.")
        from .live import LiveView
        viewer = LiveView(settings, exp.reference(), every=a.live_every)

    if a.command == "all":
        warnings = []
        for i, stage in enumerate(PIPELINE, 1):
            print(f"\n=== [{i}/{len(PIPELINE)}] {stage} ===", flush=True)
            if stage == "check":
                exp.check_models(settings)
            elif stage == "smoke":
                exp.smoke(settings, a.verbose)
            elif stage.startswith("baseline_"):
                # if not run_baseline(exp, settings, stage.split("_")[1], a):
                if not run_baseline(exp, settings, stage.split("_")[1], a, viewer):
                    warnings.append(f"{stage} stopped early; see saved metrics.")
            elif stage == "collect":
                exp.collect(settings, a.samples or settings["training"]["samples"], a.verbose)
            elif stage == "train":
                exp.train(settings, a.epochs or settings["training"]["epochs"], a.verbose)
            elif stage == "compare":
                # if not run_compare(exp, settings, a, parser):
                if not run_compare(exp, settings, a, parser, viewer):
                    warnings.append("A comparison rollout stopped early. Compare completion/failure before RMS errors.")
        print("\n=== Pipeline finished ===")
        for w in warnings:
            print("WARNING:", w)
        if viewer is not None:
            viewer.finish()
        if warnings:
            raise SystemExit(1)
    elif a.command == "check":
        exp.check_models(settings)
    elif a.command == "smoke":
        exp.smoke(settings, a.verbose)
    elif a.command == "collect":
        exp.collect(settings, a.samples or settings["training"]["samples"], a.verbose)
    elif a.command == "train":
        exp.train(settings, a.epochs or settings["training"]["epochs"], a.verbose)
    elif a.command == "baseline":
        # if not run_baseline(exp, settings, a.horizon, a):
        ok = run_baseline(exp, settings, a.horizon, a, viewer)
        if viewer is not None:
            viewer.finish()
        if not ok:
            raise SystemExit("Baseline stopped early; see saved metrics.")
    else:
        # if not run_compare(exp, settings, a, parser):
        ok = run_compare(exp, settings, a, parser, viewer)
        if viewer is not None:
            viewer.finish()
        if not ok:
            raise SystemExit("A comparison rollout stopped early. Compare completion/failure before RMS errors.")


if __name__ == "__main__":
    main()
