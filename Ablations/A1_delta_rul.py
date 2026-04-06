"""
A1_delta_rul.py  —  Ablation A1: ΔRUL Signal On vs Off
=======================================================
Directly answers RQ2: "Does the ΔRUL-based dense reward signal provide
sufficient credit assignment for the PDM agent?"

Method:
  Evaluate the trained checkpoint under two conditions:
    (1) Full model:       w_RUL = 0.05  (normal)
    (2) No ΔRUL signal:  w_RUL = 0.0   (ablated)

  Metrics of interest:
    - PM Events/ep:    should drop significantly without ΔRUL
    - Failures/ep:     should rise without ΔRUL (less proactive maintenance)
    - Service Level:   downstream effect of more failures
    - Total Cost:      composite metric

Expected finding:
  Without ΔRUL, the agent has no immediate reward for PM.
  It must wait for failure avoidance (delayed signal, discounted by γ^40 ≈ 0.67).
  PM frequency should drop and failures should rise.

Usage:
  python ablations/A1_delta_rul.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 100 \\
      --outdir results/ablation_a1/
"""

import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ablations.ablation_utils import (
    load_config, eval_marl_policy, compare_table,
    patch_weights, save_results, statistical_summary
)
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description="A1: ΔRUL ablation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained Phase 3 checkpoint")
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a1/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A1: ΔRUL SIGNAL")
    print("  RQ2: Does ΔRUL provide sufficient credit assignment?")
    print("="*60)

    # ── Condition 1: Full model (w_RUL = 0.05) ──────────────────────────────
    print("\nCondition 1: Full model (w_RUL = 0.05)")
    result_full = eval_marl_policy(
        checkpoint_path=args.checkpoint,
        config=cfg,
        n_episodes=args.episodes,
        stoch_level=args.stoch,
        name="MARL_w_RUL=0.05",
    )

    # ── Condition 2: No ΔRUL (w_RUL = 0.0) ──────────────────────────────────
    print("\nCondition 2: No ΔRUL signal (w_RUL = 0.0)")
    with patch_weights(w_RUL=0.0):
        result_no_rul = eval_marl_policy(
            checkpoint_path=args.checkpoint,
            config=cfg,
            n_episodes=args.episodes,
            stoch_level=args.stoch,
            name="MARL_w_RUL=0.0",
        )

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n--- COMPARISON TABLE ---")
    kpis_of_interest = ["failures", "n_PM", "n_CM", "pm_cm_ratio",
                        "completions", "tardiness", "service_level"]
    table = compare_table(
        [result_full, result_no_rul],
        kpis=kpis_of_interest,
        reference_name="MARL_w_RUL=0.05",
    )

    # ── Statistical summary ──────────────────────────────────────────────────
    print("\n--- STATISTICAL TESTS (Full vs No ΔRUL) ---")
    stats_out = statistical_summary(result_full, result_no_rul, kpis=kpis_of_interest)
    for kpi, s in stats_out.items():
        sig = "* SIGNIFICANT *" if s["significant"] else ""
        direction = ""
        if kpi in ["failures", "tardiness"] and s["mean_a"] < s["mean_b"]:
            direction = "↓ ΔRUL reduces failures"
        elif kpi in ["n_PM"] and s["mean_a"] > s["mean_b"]:
            direction = "↑ ΔRUL increases PM"
        print(f"  {kpi:<20}: full={s['mean_a']:.3f}  no_rul={s['mean_b']:.3f}"
              f"  p={s['p_welch']:.4f}  d={s['cohen_d']:.2f}  {sig} {direction}")

    # ── PM timing analysis ────────────────────────────────────────────────────
    pm_full   = np.mean(result_full.kpi_data["n_PM"])
    pm_no_rul = np.mean(result_no_rul.kpi_data["n_PM"])
    print(f"\n  PM reduction without ΔRUL: {pm_full:.2f} → {pm_no_rul:.2f}"
          f" ({(pm_no_rul - pm_full)/max(pm_full,0.01)*100:+.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_results([result_full, result_no_rul], args.outdir, "a1_delta_rul")

    report = {
        "ablation": "A1_delta_rul",
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "w_RUL_full": 0.05,
        "w_RUL_ablated": 0.0,
        "results_full": result_full.means,
        "results_no_rul": result_no_rul.means,
        "statistical_tests": stats_out,
        "conclusion": (
            "ΔRUL signal is necessary" if stats_out.get("n_PM", {}).get("significant")
            else "No significant difference (ΔRUL may not be critical)"
        )
    }
    with open(os.path.join(args.outdir, "a1_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    table_path = os.path.join(args.outdir, "a1_comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(f"Ablation A1: ΔRUL Signal (w_RUL = 0.05 vs 0.0)\n")
        f.write(f"Episodes: {args.episodes}, Stoch level: {args.stoch}\n\n")
        f.write(table)

    print(f"\n  Outputs saved to: {args.outdir}")
    print(f"  → {args.outdir}/a1_delta_rul.json       (raw data)")
    print(f"  → {args.outdir}/a1_report.json           (summary + stats)")
    print(f"  → {args.outdir}/a1_comparison_table.txt  (paper table)")


if __name__ == "__main__":
    main()
