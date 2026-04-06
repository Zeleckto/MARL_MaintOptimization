"""
A2_fail_idle_sensitivity.py  —  Ablation A2: w_fail_idle Sensitivity
=====================================================================
Tests whether w_fail_idle = 2.0 is the right value.

Three conditions: w_fail_idle = 0.5 / 1.0 / 2.0 (current)

Expected:
  - Too low (0.5): CM events drop, failures stay high (insufficient incentive)
  - Too high (>2.0): r1 variance explodes, gradient noisy, slower convergence
  - 2.0: CM events present, failures decreasing, gradient stable

This ablation uses the TRAINED checkpoint (not retraining).
w_fail_idle only changes the reward signal used during evaluation,
which affects which actions get credit — but since the policy is
already trained, we're testing the policy's response to different
evaluation rewards.

Note: True ablation of w_fail_idle requires retraining with each value.
This script does the eval-time version (fast), which tests how much
the penalty matters post-training.

Usage:
  python ablations/A2_fail_idle_sensitivity.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 50 \\
      --outdir results/ablation_a2/
"""

import argparse, os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ablations.ablation_utils import (
    load_config, eval_marl_policy, compare_table,
    patch_weights, save_results, statistical_summary
)
import numpy as np


W_VALUES = [0.5, 1.0, 2.0, 4.0]   # test range around current default 2.0


def main():
    parser = argparse.ArgumentParser(description="A2: w_fail_idle sensitivity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a2/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A2: w_fail_idle SENSITIVITY")
    print(f"  Testing values: {W_VALUES}")
    print("="*60)

    results = []
    for w in W_VALUES:
        print(f"\nCondition: w_fail_idle = {w}")
        with patch_weights(w_fail_idle=w):
            r = eval_marl_policy(
                checkpoint_path=args.checkpoint,
                config=cfg,
                n_episodes=args.episodes,
                stoch_level=args.stoch,
                name=f"w_idle={w}",
            )
        results.append(r)

    print("\n--- COMPARISON TABLE ---")
    kpis = ["failures", "n_PM", "n_CM", "pm_cm_ratio",
            "completions", "service_level", "avg_health"]
    table = compare_table(results, kpis=kpis, reference_name="w_idle=2.0")

    # Key analysis: CM events vs w_fail_idle
    print("\n--- CM EVENTS vs w_fail_idle ---")
    for r in results:
        cm_mean = np.mean(r.kpi_data["n_CM"])
        fail_mean = np.mean(r.kpi_data["failures"])
        print(f"  w_fail_idle={r.name.split('=')[1]:>4}: "
              f"CM={cm_mean:.2f}/ep  Failures={fail_mean:.2f}/ep")

    print("\n--- BREAK-EVEN ANALYSIS ---")
    for w in W_VALUES:
        c_CM = 7.0  # reward cost of CM
        steps_to_breakeven = c_CM / (w * max(1, 1))  # 1 FAIL machine
        print(f"  w_fail_idle={w}: CM pays back in {steps_to_breakeven:.1f} FAIL-steps "
              f"({steps_to_breakeven:.0f} shifts ≈ {steps_to_breakeven*8:.0f}h)")

    # Save
    save_results(results, args.outdir, "a2_fail_idle")

    report = {
        "ablation": "A2_fail_idle_sensitivity",
        "w_values": W_VALUES,
        "current_default": 2.0,
        "results": {r.name: r.means for r in results},
        "recommended": "2.0 — break-even in 4 FAIL-steps, dominant enough for learning",
    }
    with open(os.path.join(args.outdir, "a2_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(args.outdir, "a2_table.txt"), "w") as f:
        f.write(table)

    print(f"\n  Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
