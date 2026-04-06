"""
check_convergence.py  —  Training Convergence Monitor
======================================================
Reads TensorBoard events and reports whether each training phase
has converged, with a GO / NO-GO recommendation.

Convergence criteria (all must pass for GO):
  1. n_CM > 0 regularly  (CM learning confirmed)
  2. n_PM > 0 regularly  (PM learning confirmed)
  3. failures/ep trending down  (late mean < 90% of early mean)
  4. entropy not collapsed  (late entropy > 1.0)
  5. availability > 0.90  (fleet staying operational)
  6. r1 CV < 20%  (reward not wildly oscillating over last 100 eps)

Usage:
  python ablations/check_convergence.py              # auto-detects latest run
  python ablations/check_convergence.py --run runs/phase1_12345
  python ablations/check_convergence.py --window 100  # use last 100 episodes
"""

import argparse, os, sys, glob
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np


# ── Colour codes ──────────────────────────────────────────────────────────────
G  = "\033[92m";  R  = "\033[91m";  Y  = "\033[93m";  X  = "\033[0m"
B  = "\033[94m";  W  = "\033[97m"


def load_tb_data(run_dir: str) -> dict:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("tensorboard not installed: pip install tensorboard")
        sys.exit(1)

    ea = event_accumulator.EventAccumulator(run_dir,
             size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = np.array([e.value for e in events])
    return data


def find_latest_run(runs_dir: str = "runs") -> str:
    candidates = [c for c in glob.glob(os.path.join(runs_dir, "*"))
                  if os.path.isdir(c)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def check_metric(
    data: dict, tag: str, window: int,
    criterion: str, threshold: float,
    label: str,
) -> tuple:
    """
    Returns (passed: bool, message: str, value: float|None)

    criterion: "gt"  = late_mean > threshold
               "lt"  = late_mean < threshold
               "pos" = late_mean > 0
               "trend_down" = late_mean < early_mean * threshold
               "cv"  = CV of late window < threshold
    """
    if tag not in data or len(data[tag]) < window * 2:
        return None, f"{label}: {Y}NOT ENOUGH DATA{X} (need {window*2} points)", None

    vals = data[tag]
    n    = len(vals)
    late  = vals[max(0, n - window):]
    early = vals[:min(window, n//2)]

    late_mean  = float(np.mean(late))
    early_mean = float(np.mean(early))
    late_std   = float(np.std(late))
    cv         = late_std / (abs(late_mean) + 1e-8)

    if criterion == "gt":
        passed = late_mean > threshold
        msg = f"{label}: {late_mean:.3f} (threshold > {threshold})"
    elif criterion == "lt":
        passed = late_mean < threshold
        msg = f"{label}: {late_mean:.3f} (threshold < {threshold})"
    elif criterion == "pos":
        passed = late_mean > 0.01
        msg = f"{label}: mean={late_mean:.3f} {'> 0 ✓' if passed else '= 0 ✗'}"
    elif criterion == "trend_down":
        passed = late_mean <= early_mean * threshold
        pct = (late_mean - early_mean) / (abs(early_mean) + 1e-8) * 100
        msg = f"{label}: early={early_mean:.2f} → late={late_mean:.2f} ({pct:+.0f}%)"
    elif criterion == "cv":
        passed = cv < threshold
        msg = f"{label}: CV={cv:.2f} (threshold < {threshold})"
    else:
        passed = False
        msg = f"{label}: unknown criterion"

    colour = G if passed else R
    return passed, f"  {colour}{'PASS' if passed else 'FAIL'}{X}  {msg}", late_mean


def main():
    parser = argparse.ArgumentParser(description="Check training convergence")
    parser.add_argument("--run",    default=None)
    parser.add_argument("--window", type=int, default=100,
                        help="Episodes in late/early window (default: 100)")
    parser.add_argument("--outdir", default=None,
                        help="Save report to this directory")
    args = parser.parse_args()

    run_dir = args.run or find_latest_run()
    if run_dir is None:
        print("No runs/ directory found. Run training first.")
        sys.exit(1)
    print(f"{B}Auto-detected run: {run_dir}{X}" if args.run is None
          else f"{B}Run: {run_dir}{X}")

    data = load_tb_data(run_dir)
    n_tags = len(data)
    n_pts  = sum(len(v) for v in data.values())
    print(f"Loaded {n_tags} tags, {n_pts:,} data points\n")

    w = args.window
    print(f"{W}{'='*60}{X}")
    print(f"{W}  CONVERGENCE CHECK  (window = last/first {w} episodes){X}")
    print(f"{W}{'='*60}{X}\n")

    # ── Define checks ─────────────────────────────────────────────────────────
    checks = [
        # (tag, criterion, threshold, label, mandatory)
        ("episode/n_CM",           "pos",        0.01, "CM events > 0",              True),
        ("episode/n_PM",           "pos",        0.01, "PM events > 0",              True),
        ("episode/failures",       "trend_down", 0.90, "Failures trending down",     True),
        ("train/entropy1",         "gt",         1.0,  "Entropy not collapsed",      True),
        ("episode/availability",   "gt",         0.90, "Availability > 90%",         True),
        ("episode/return_agent1",  "cv",         0.25, "r1 CV < 25% (stable)",       False),
        ("episode/pm_cm_ratio",    "gt",         1.5,  "PM/CM ratio > 1.5",          False),
        ("episode/service_level",  "gt",         0.40, "Service level > 40%",        False),
        ("episode/jobs_completed", "trend_down", 1.05, "Jobs not declining",         False),
        ("train/entropy2",         "gt",         1.0,  "Entropy2 not collapsed",     False),
    ]

    results = []
    for tag, criterion, threshold, label, mandatory in checks:
        passed, msg, val = check_metric(data, tag, w, criterion, threshold, label)
        results.append((passed, msg, mandatory))
        print(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    mandatory_results = [(p, m) for p, m, mand in results if mand and p is not None]
    optional_results  = [(p, m) for p, m, mand in results if not mand and p is not None]

    n_mand_pass  = sum(1 for p, _ in mandatory_results if p)
    n_mand_total = len(mandatory_results)
    n_opt_pass   = sum(1 for p, _ in optional_results if p)
    n_opt_total  = len(optional_results)

    print(f"\n{W}{'='*60}{X}")
    if n_mand_pass == n_mand_total:
        print(f"\n  {G}✓ GO FOR NEXT PHASE{X}")
        print(f"  All {n_mand_total}/{n_mand_total} mandatory checks passed.")
        print(f"  Optional: {n_opt_pass}/{n_opt_total} passed.")
        decision = "GO"
    else:
        failed = [m for p, m, mand in results if mand and p is False]
        print(f"\n  {R}✗ NO-GO — CONTINUE TRAINING{X}")
        print(f"  {n_mand_pass}/{n_mand_total} mandatory checks passed.")
        print(f"  Failed mandatory checks:")
        for m in failed:
            clean = m.replace(G,'').replace(R,'').replace(Y,'').replace(X,'').replace(B,'')
            print(f"    {clean}")
        decision = "NO-GO"

    # ── Training summary ──────────────────────────────────────────────────────
    print(f"\n{W}  TRAINING SUMMARY:{X}")

    for tag, label in [
        ("episode/failures",       "  Failures/ep"),
        ("episode/n_PM",           "  PM events/ep"),
        ("episode/n_CM",           "  CM events/ep"),
        ("episode/jobs_completed", "  Jobs done/ep"),
        ("episode/availability",   "  Availability"),
        ("train/entropy1",         "  Entropy (A1)"),
    ]:
        if tag in data and len(data[tag]) > 0:
            v = data[tag]
            n = len(v)
            early_w = min(w, n//4)
            late_w  = min(w, n//4)
            e_mean = np.mean(v[:early_w])
            l_mean = np.mean(v[max(0, n-late_w):])
            trend  = "↓" if l_mean < e_mean * 0.95 else "↑" if l_mean > e_mean * 1.05 else "→"
            print(f"    {label:<22}: early={e_mean:.3f}  late={l_mean:.3f}  {trend}")

    total_eps = max((len(v) for v in data.values()), default=0)
    print(f"\n    Total episodes logged: {total_eps}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{W}  RECOMMENDED NEXT STEPS:{X}")
    if decision == "GO":
        print(f"  {G}→ Save checkpoint and move to next phase:{X}")
        print(f"    cp checkpoints/latest.pt checkpoints/phase1_best.pt")
        print(f"    python scripts/train.py --config configs/phase2.yaml \\")
        print(f"        --timesteps 150000 --resume checkpoints/phase1_best.pt")
    else:
        remaining = n_mand_total - n_mand_pass
        print(f"  {Y}→ Continue Phase 1 training:{X}")
        print(f"    {remaining} mandatory check(s) still failing.")
        print(f"    python scripts/train.py --config configs/phase1.yaml \\")
        print(f"        --timesteps 50000 --resume checkpoints/latest.pt")
        print(f"    Then re-run: python ablations/check_convergence.py")

    # ── Save report ───────────────────────────────────────────────────────────
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        import json
        report = {
            "run_dir": run_dir,
            "window": w,
            "decision": decision,
            "mandatory_pass": n_mand_pass,
            "mandatory_total": n_mand_total,
            "optional_pass": n_opt_pass,
            "optional_total": n_opt_total,
            "total_episodes": total_eps,
        }
        path = os.path.join(args.outdir, "convergence_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {path}")

    print(f"\n{W}{'='*60}{X}")


if __name__ == "__main__":
    main()
