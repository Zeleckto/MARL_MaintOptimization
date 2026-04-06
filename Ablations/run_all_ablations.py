"""
run_all_ablations.py  —  Run All Ablation Studies
==================================================
Orchestrates all ablation studies in priority order.

Priority groups:
  GROUP 1 (MUST DO, <2h total):
    A1 - ΔRUL signal on/off      (answers RQ2)
    A9 - PM timing distribution  (strongest visual result)
    A10 - Health dispatch        (shows cooperative behaviour)

  GROUP 2 (SHOULD DO, <4h total):
    A2 - w_fail_idle sensitivity
    A12 - λ coupling sensitivity

  GROUP 3 (NEED NEW EVAL ENV, ~30min):
    A8 - Zero-shot scaling M=5→M=10

Usage:
  # Run all groups
  python ablations/run_all_ablations.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --outdir results/ablations/

  # Run only group 1 (fast, paper essentials)
  python ablations/run_all_ablations.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --group 1 \\
      --outdir results/ablations/

  # Run specific ablation
  python ablations/run_all_ablations.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --only A1 A9 \\
      --outdir results/ablations/
"""

import argparse, os, sys, subprocess, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


ABLATIONS = {
    # (script, default_episodes, group, description)
    "A1":  ("ablations/A1_delta_rul.py",          100, 1, "ΔRUL signal ablation (RQ2)"),
    "A9":  ("ablations/A9_pm_timing.py",            50, 1, "PM timing vs ABR t*"),
    "A10": ("ablations/A10_health_dispatch.py",     50, 1, "Health-conditioned dispatch"),
    "A2":  ("ablations/A2_fail_idle_sensitivity.py", 50, 2, "w_fail_idle sensitivity"),
    "A12": ("ablations/A12_lambda_sensitivity.py",   50, 2, "λ coupling sensitivity"),
    "A4":  ("ablations/A4_independent_ppo.py",       50, 2, "IndepPPO vs MAPPO (retrains)"),
    "A8":  ("ablations/A8_zeroshot_scaling.py",      20, 3, "Zero-shot M=5→M=10"),
}


def run_ablation(script: str, checkpoint: str, episodes: int, outdir: str,
                 stoch: int, extra_args: list = None) -> dict:
    """Run a single ablation script and return timing + status."""
    cmd = [
        sys.executable, str(ROOT / script),
        "--checkpoint", checkpoint,
        "--episodes",   str(episodes),
        "--stoch",      str(stoch),
        "--outdir",     outdir,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Running: {script}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {script} — {elapsed:.0f}s")
    return {"script": script, "status": status, "elapsed_s": round(elapsed)}


def main():
    parser = argparse.ArgumentParser(description="Run all ablation studies")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained Phase 3 checkpoint")
    parser.add_argument("--outdir",     default="results/ablations/",
                        help="Root output directory")
    parser.add_argument("--stoch",      type=int, default=3,
                        help="Stochasticity level (default: 3)")
    parser.add_argument("--group",      type=int, default=None,
                        help="Run only this group (1, 2, or 3)")
    parser.add_argument("--only",       nargs="+", default=None,
                        help="Run only these ablations (e.g. --only A1 A9)")
    parser.add_argument("--episodes",   type=int, default=None,
                        help="Override episode count for all ablations")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine which ablations to run
    to_run = list(ABLATIONS.keys())
    if args.group is not None:
        to_run = [k for k, v in ABLATIONS.items() if v[2] == args.group]
    if args.only is not None:
        to_run = [k for k in args.only if k in ABLATIONS]

    print("\n" + "="*60)
    print("  ABLATION STUDY SUITE")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Running: {to_run}")
    print("="*60)

    # Print estimated time
    total_eps = sum(
        (args.episodes or ABLATIONS[k][1]) for k in to_run
    )
    print(f"\n  Estimated total episodes: ~{total_eps}")
    print(f"  Estimated time: ~{total_eps * 0.5 / 60:.0f} min")
    print()

    # Run
    log = []
    for ablation_id in to_run:
        script, default_eps, group, desc = ABLATIONS[ablation_id]
        episodes = args.episodes or default_eps
        outdir_i = os.path.join(args.outdir, ablation_id.lower())

        print(f"\n[{ablation_id}] {desc}")
        result = run_ablation(
            script=script,
            checkpoint=args.checkpoint,
            episodes=episodes,
            outdir=outdir_i,
            stoch=args.stoch,
        )
        result["ablation_id"] = ablation_id
        result["description"] = desc
        log.append(result)

    # Summary
    print("\n" + "="*60)
    print("  ABLATION SUITE COMPLETE")
    print("="*60)
    total_time = sum(r["elapsed_s"] for r in log)
    print(f"\n  Total time: {total_time//60:.0f}m {total_time%60:.0f}s")
    print()
    for r in log:
        status_mark = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"  {status_mark} [{r['ablation_id']}] {r['description']:<40} "
              f"{r['elapsed_s']}s  →  results/ablations/{r['ablation_id'].lower()}/")

    # Save run log
    log_path = os.path.join(args.outdir, "ablation_run_log.json")
    with open(log_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "ablations": log}, f, indent=2)
    print(f"\n  Run log: {log_path}")

    # Quick paper table (load all JSON reports and summarise)
    print("\n" + "="*60)
    print("  RESULTS SUMMARY FOR PAPER")
    print("="*60)

    for r in log:
        if r["status"] != "SUCCESS":
            continue
        abl_id = r["ablation_id"]
        report_path = os.path.join(args.outdir, abl_id.lower(),
                                   f"{abl_id.lower()}_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            print(f"\n  [{abl_id}] {r['description']}")
            if "conclusion" in report:
                print(f"    Conclusion: {report['conclusion']}")


if __name__ == "__main__":
    main()
