"""
Ablations/check_convergence.py — GO/NO-GO for Phase Transition
===============================================================
Reads TensorBoard data from outputs/runs/ and determines whether
Phase 1 has converged enough to proceed to Phase 2.

Criteria (all three must pass):
  1. Mean availability (last --window episodes) > 0.80
  2. Entropy agent1 (last 10 updates) < 3.0  (was declining from 3.47)
  3. Critic loss (last 10 updates) > 0  (critic is actually training)

Usage:
    python Ablations/check_convergence.py
    python Ablations/check_convergence.py --window 50 --avail-target 0.85
"""
import argparse, os, sys, glob, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_OK = True
except ImportError:
    print("pip install tensorboard"); sys.exit(1)


def load_tag(logdirs, tag):
    """Load all values for one tag across runs, sorted by step."""
    raw = {}
    for base in logdirs:
        for run in (sorted(os.listdir(base)) if os.path.isdir(base) else [base]):
            path = os.path.join(base, run) if os.path.isdir(base) else base
            if not os.path.isdir(path): continue
            try:
                ea = EventAccumulator(path, size_guidance={EventAccumulator.SCALARS: 0})
                ea.Reload()
                if tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        raw[e.step] = e.value
            except: continue
    if not raw: return np.array([]), np.array([])
    steps = sorted(raw); vals = np.array([raw[s] for s in steps])
    return np.array(steps), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdirs",      nargs="+", default=["outputs/runs/", "runs/", "runs_archive/"])
    ap.add_argument("--window",       type=int, default=50)
    ap.add_argument("--avail-target", type=float, default=0.80)
    ap.add_argument("--entropy-max",  type=float, default=3.0)
    args = ap.parse_args()

    dirs = [d for d in args.logdirs if os.path.exists(d)]
    if not dirs:
        print("No TB log directories found."); return

    print(f"\n{'='*60}")
    print(f"  CONVERGENCE CHECK (window={args.window} episodes)")
    print(f"{'='*60}")

    results = {}
    max_ent = 5 * math.log(2)

    # ── Availability ──────────────────────────────────────────────
    _, av = load_tag(dirs, "episode/availability")
    if len(av) >= args.window:
        last_av = np.mean(av[-args.window:])
        std_av  = np.std(av[-args.window:])
        ok_av   = last_av >= args.avail_target
        results["Availability"] = (ok_av, last_av,
            f"mean={last_av:.4f} ± {std_av:.4f}  target>={args.avail_target}")
    else:
        results["Availability"] = (False, 0.0,
            f"Only {len(av)} eps logged (need {args.window})")

    # ── Entropy ───────────────────────────────────────────────────
    _, ent = load_tag(dirs, "train/entropy1")
    if len(ent) >= 5:
        last_ent = np.mean(ent[-min(10, len(ent)):])
        ok_ent   = last_ent < args.entropy_max
        pct_max  = last_ent / max_ent * 100
        results["Entropy A1"] = (ok_ent, last_ent,
            f"last_mean={last_ent:.4f}  max_entropy={max_ent:.4f} ({pct_max:.0f}% of max)"
            f"  target<{args.entropy_max}")
    else:
        results["Entropy A1"] = (False, max_ent,
            f"Only {len(ent)} training updates logged (need ≥5)")

    # ── Critic loss ───────────────────────────────────────────────
    _, cl = load_tag(dirs, "train/critic_loss")
    if len(cl) >= 5:
        last_cl = np.mean(cl[-min(10, len(cl)):])
        ok_cl   = last_cl > 0
        results["Critic loss"] = (ok_cl, last_cl,
            f"last_mean={last_cl:.6f}  {'TRAINING ✓' if ok_cl else 'ZERO! Bug 1 still present'}")
    else:
        results["Critic loss"] = (None, 0.0, "Not enough data yet — check after first PPO update")

    # ── Failures ──────────────────────────────────────────────────
    _, fail = load_tag(dirs, "episode/failures")
    if len(fail) >= args.window:
        last_fail = np.mean(fail[-args.window:])
        results["Failures/ep"] = (last_fail < 3.0, last_fail,
            f"mean={last_fail:.2f}  (informational, not a GO criterion)")

    # ── CM events ─────────────────────────────────────────────────
    _, n_cm = load_tag(dirs, "episode/n_CM")
    if len(n_cm) >= 5:
        cm_any = np.any(n_cm > 0)
        results["CM events"] = (cm_any, float(np.mean(n_cm[-20:])),
            f"CM appeared: {cm_any}  recent_mean={np.mean(n_cm[-20:]):.2f}")

    # ── r_shared check ────────────────────────────────────────────
    _, rs = load_tag(dirs, "rewards/shared")
    if len(rs) > 0:
        rs_nz = (rs != 0).mean() * 100
        results["r_shared"] = (rs_nz > 0, rs_nz,
            f"{rs_nz:.1f}% of steps non-zero  {'OK' if rs_nz>0 else 'Bug 3 still present!'}")

    # ── Print results ─────────────────────────────────────────────
    GO_criteria = ["Availability", "Entropy A1", "Critic loss"]
    all_go = True
    print()
    for name, (ok, val, detail) in results.items():
        is_go = name in GO_criteria
        icon  = ("✓ GO " if ok else "✗ WAIT") if is_go else ("  OK " if ok else "  --- ")
        mark  = "  [GO CRITERION]" if is_go else ""
        print(f"  {icon}  {name:<18} {detail}{mark}")
        if is_go and not ok: all_go = False

    print()
    print(f"{'='*60}")
    if all_go:
        print(f"  ✓ GO FOR NEXT PHASE")
        print(f"    Save: copy outputs\\checkpoints\\latest.pt outputs\\checkpoints\\phase1_end.pt")
    else:
        wait_items = [n for n in GO_criteria if not results.get(n, (False,))[0]]
        print(f"  ✗ NOT YET — waiting on: {', '.join(wait_items)}")
        print(f"    Run 30k more steps then check again:")
        print(f"    python scripts/train.py --config configs/phase1.yaml --timesteps 30000 --resume outputs/checkpoints/latest.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()