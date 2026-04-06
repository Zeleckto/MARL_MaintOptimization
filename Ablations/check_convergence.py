"""
check_convergence.py  —  Training Convergence Monitor
======================================================
Checks whether training has converged using ALL available TensorBoard
data — including runs_archive/ (old sessions) and current runs/.

Usage:
  python ablations/check_convergence.py              # auto-merges all runs
  python ablations/check_convergence.py --window 50  # smaller window
  python ablations/check_convergence.py --runs-dirs runs/ runs_archive/
"""

import argparse, os, sys, glob
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; X = "\033[0m"; B = "\033[94m"; W = "\033[97m"


def merge_all_runs(*dirs) -> dict:
    """
    Load and merge ALL TensorBoard event files from every directory given.
    Returns {tag: {"steps": np.array, "values": np.array}} sorted by step.
    De-duplicates by keeping last value at each step.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("pip install tensorboard")
        sys.exit(1)

    raw = {}  # {tag: {step: value}}
    n_runs = 0

    for base_dir in dirs:
        if not os.path.isdir(base_dir):
            continue
        run_dirs = [d for d in sorted(glob.glob(os.path.join(base_dir, "*")))
                    if os.path.isdir(d)]
        # Also check if base_dir itself has events (single-run case)
        if not run_dirs:
            run_dirs = [base_dir]

        for run_dir in run_dirs:
            try:
                ea = event_accumulator.EventAccumulator(
                    run_dir, size_guidance={event_accumulator.SCALARS: 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                if not tags:
                    continue
                n_runs += 1
                for tag in tags:
                    if tag not in raw:
                        raw[tag] = {}
                    for e in ea.Scalars(tag):
                        raw[tag][e.step] = e.value
            except Exception:
                continue

    # Convert to sorted arrays
    data = {}
    for tag, step_val in raw.items():
        steps = sorted(step_val.keys())
        data[tag] = {
            "steps":  np.array(steps),
            "values": np.array([step_val[s] for s in steps]),
        }

    total_pts = sum(len(v["steps"]) for v in data.values())
    print(f"  Merged {n_runs} run(s): {len(data)} tags, {total_pts:,} data points")
    return data


def check_metric(data, tag, window, criterion, threshold, label):
    """Returns (passed, message, late_mean)."""
    if tag not in data:
        return None, f"  {Y}----{X}  {label}: tag not logged", None

    vals = data[tag]["values"]
    n    = len(vals)

    if n < window:
        return None, (f"  {Y}SKIP{X}  {label}: only {n} points "
                      f"(need {window} — use --window {max(5, n//2)})"), None

    late   = vals[max(0, n - window):]
    early  = vals[:min(window, n // 2)]
    l_mean = float(np.mean(late))
    e_mean = float(np.mean(early))
    cv     = float(np.std(late)) / (abs(l_mean) + 1e-8)

    if criterion == "gt":
        passed = l_mean > threshold
        msg = f"{label}: {l_mean:.3f} (need > {threshold})"
    elif criterion == "lt":
        passed = l_mean < threshold
        msg = f"{label}: {l_mean:.3f} (need < {threshold})"
    elif criterion == "pos":
        passed = l_mean > 0.05
        msg = f"{label}: late_mean={l_mean:.3f} {'> 0 ✓' if passed else '= 0 ✗ — not learned yet'}"
    elif criterion == "trend_down":
        passed = l_mean <= e_mean * threshold
        pct = (l_mean - e_mean) / (abs(e_mean) + 1e-8) * 100
        msg = f"{label}: early={e_mean:.2f} → late={l_mean:.2f} ({pct:+.0f}%)"
    elif criterion == "trend_up":
        passed = l_mean >= e_mean * threshold
        pct = (l_mean - e_mean) / (abs(e_mean) + 1e-8) * 100
        msg = f"{label}: early={e_mean:.2f} → late={l_mean:.2f} ({pct:+.0f}%)"
    elif criterion == "cv":
        passed = cv < threshold
        msg = f"{label}: CV={cv:.2f} (need < {threshold})"
    else:
        passed, msg = False, f"{label}: unknown criterion"

    colour = G if passed else R
    return passed, f"  {colour}{'PASS' if passed else 'FAIL'}{X}  {msg}", l_mean


def main():
    parser = argparse.ArgumentParser(description="Training convergence checker")
    parser.add_argument("--runs-dirs", nargs="+",
                        default=["runs", "runs_archive"],
                        help="Directories to merge (default: runs/ runs_archive/)")
    parser.add_argument("--window",    type=int, default=100,
                        help="Episodes in early/late window (default: 100)")
    parser.add_argument("--outdir",    default=None)
    args = parser.parse_args()

    print(f"\n{B}{'='*60}{X}")
    print(f"{B}  CONVERGENCE CHECK — loading from: {args.runs_dirs}{X}")
    print(f"{B}{'='*60}{X}\n")

    data = merge_all_runs(*args.runs_dirs)
    w    = args.window

    # ── Auto-reduce window if not enough data ────────────────────────────────
    # Find max episode data available
    ep_tags = [t for t in data if t.startswith("episode/")]
    if ep_tags:
        max_ep_pts = max(len(data[t]["values"]) for t in ep_tags)
    else:
        max_ep_pts = 0

    # Fall back to per-step reward data if episode data is sparse
    has_ep = max_ep_pts >= w
    has_rewards = "rewards/agent1_r1" in data and len(data["rewards/agent1_r1"]["values"]) >= w

    if not has_ep and not has_rewards:
        w = max(5, max_ep_pts // 2 if max_ep_pts > 0 else 5)
        print(f"{Y}  ⚠ Low data: auto-reducing window to {w}{X}\n")
    elif not has_ep and has_rewards:
        print(f"{Y}  ⚠ Episode-level data sparse ({max_ep_pts} pts). "
              f"Using per-step reward data for convergence.{X}\n")

    print(f"  Window = {w} | Episode data pts = {max_ep_pts} | "
          f"Reward data pts = {len(data.get('rewards/agent1_r1', {}).get('values', []))}\n")

    # ── Define checks ─────────────────────────────────────────────────────────
    # (tag, criterion, threshold, label, mandatory)
    checks = []

    # Episode-level checks (if we have enough data)
    if has_ep:
        checks += [
            ("episode/n_CM",           "pos",        0.05, "CM events > 0",              True),
            ("episode/n_PM",           "pos",        0.05, "PM events > 0",              True),
            ("episode/failures",       "trend_down", 0.90, "Failures trending down",     True),
            ("train/entropy1",         "gt",         1.0,  "Entropy not collapsed",      True),
            ("episode/availability",   "gt",         0.85, "Availability > 85%",         True),
            ("episode/pm_cm_ratio",    "gt",         1.0,  "PM/CM ratio > 1",            False),
            ("episode/service_level",  "gt",         0.35, "Service level > 35%",        False),
            ("episode/jobs_completed", "trend_up",   0.95, "Jobs not declining",         False),
        ]
    else:
        # Fall back to per-step reward convergence
        checks += [
            ("rewards/agent1_r1",  "trend_up",   0.80, "r1 improving (less negative)", True),
            ("rewards/total",      "trend_up",   0.80, "Total reward improving",       True),
            ("train/entropy1",     "gt",         1.0,  "Entropy not collapsed",        True),
        ]
        # Add any episode tags we DO have
        for tag, label in [
            ("episode/n_PM",    "PM events > 0"),
            ("episode/n_CM",    "CM events > 0"),
            ("episode/failures","Failures trending down"),
        ]:
            if tag in data and len(data[tag]["values"]) > 0:
                checks.append((tag, "pos", 0.0, label, False))

    # Run checks
    results = []
    for check in checks:
        tag, criterion, threshold, label, mandatory = check
        passed, msg, val = check_metric(data, tag, w, criterion, threshold, label)
        results.append((passed, msg, mandatory))
        print(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    mand    = [(p, m) for p, m, mand in results if mand and p is not None]
    opt     = [(p, m) for p, m, mand in results if not mand and p is not None]
    skipped = sum(1 for p, m, mand in results if p is None)

    n_mand_pass = sum(1 for p, _ in mand if p)
    n_opt_pass  = sum(1 for p, _ in opt if p)

    print(f"\n{W}{'='*60}{X}")

    if not mand and skipped > 0:
        print(f"\n  {Y}⚠ INSUFFICIENT DATA — cannot make GO/NO-GO decision{X}")
        print(f"  All {len(checks)} checks skipped due to low data.")
        print(f"\n  Data available:")
        for t in sorted(data.keys()):
            n = len(data[t]["values"])
            if n > 5:
                print(f"    {t}: {n} pts")
        decision = "UNKNOWN"
    elif n_mand_pass == len(mand):
        print(f"\n  {G}✓ GO FOR NEXT PHASE{X}")
        print(f"  {n_mand_pass}/{len(mand)} mandatory + {n_opt_pass}/{len(opt)} optional passed.")
        decision = "GO"
    else:
        failed = [m for p, m, mand in results if mand and p is False]
        print(f"\n  {R}✗ NO-GO — CONTINUE TRAINING{X}")
        print(f"  {n_mand_pass}/{len(mand)} mandatory passed. Still failing:")
        for m in failed:
            clean = m.replace(G,'').replace(R,'').replace(Y,'').replace(X,'')
            print(f"    {clean.strip()}")
        decision = "NO-GO"

    # ── Training summary ──────────────────────────────────────────────────────
    print(f"\n{W}  TRAINING SUMMARY:{X}")
    n = len(data.get("rewards/agent1_r1", {}).get("values", []))
    if n > 0:
        r1 = data["rewards/agent1_r1"]["values"]
        w10 = max(1, n // 10)
        print(f"    r1/step: early={np.mean(r1[:w10]):+.3f}  "
              f"late={np.mean(r1[-w10:]):+.3f}  "
              f"({'improving ✓' if np.mean(r1[-w10:]) > np.mean(r1[:w10]) else 'not improving'})")

    for tag, label in [
        ("episode/failures",       "  Failures/ep     "),
        ("episode/n_PM",           "  PM events/ep    "),
        ("episode/n_CM",           "  CM events/ep    "),
        ("episode/jobs_completed", "  Jobs done/ep    "),
        ("episode/service_level",  "  Service level   "),
    ]:
        if tag in data and len(data[tag]["values"]) > 2:
            v = data[tag]["values"]
            n = len(v)
            w5 = max(1, n // 5)
            e_m = np.mean(v[:w5])
            l_m = np.mean(v[-w5:])
            trend = "↓" if l_m < e_m * 0.95 else "↑" if l_m > e_m * 1.05 else "→"
            print(f"    {label}: {e_m:.2f} → {l_m:.2f}  {trend}  (n={n})")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{W}  NEXT STEPS:{X}")
    if decision == "GO":
        print(f"  {G}→ Ready for Phase 2:{X}")
        print(f"    cp checkpoints/latest.pt checkpoints/phase1_end.pt")
        print(f"    python scripts/train.py --config configs/phase2.yaml \\")
        print(f"        --timesteps 50000 --resume checkpoints/phase1_end.pt")
    elif decision == "UNKNOWN":
        print(f"  {Y}→ Not enough data to decide. Try:{X}")
        print(f"    python ablations/check_convergence.py --window 10")
        print(f"    tensorboard --logdir runs_archive/ --port 6007")
    else:
        print(f"  {Y}→ Continue training:{X}")
        print(f"    python scripts/train.py --config configs/phase1.yaml \\")
        print(f"        --timesteps 50000 --resume checkpoints/latest.pt")
        print(f"    Then re-run: python ablations/check_convergence.py")

    print(f"\n{W}{'='*60}{X}\n")

    if args.outdir:
        import json
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "convergence.json"), "w") as f:
            json.dump({"decision": decision, "window": w,
                       "mandatory_pass": n_mand_pass,
                       "mandatory_total": len(mand)}, f, indent=2)


if __name__ == "__main__":
    main()