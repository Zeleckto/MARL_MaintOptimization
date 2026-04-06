"""
A12_lambda_sensitivity.py  —  Ablation A12: Shared Reward Coupling λ
=====================================================================
Tests the effect of the shared failure penalty coupling strength λ.

λ = 0:   Agents are fully decoupled. Agent 1 ignores scheduling consequences
          of failures; Agent 2 ignores machine health.
λ = 0.3: Current setting. Moderate coupling.
λ = 0.6: Tighter coupling. Both agents more strongly incentivised to avoid failures.

Expected:
  λ = 0:   Less cooperative behaviour (health-aware routing may degrade)
  λ = 0.3: Balanced; best joint performance
  λ = 0.6: May over-weight failure avoidance at expense of throughput

Usage:
  python ablations/A12_lambda_sensitivity.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 50 \\
      --outdir results/ablation_a12/
"""

import argparse, os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from ablations.ablation_utils import (
    load_config, eval_marl_policy, compare_table,
    patch_weights, save_results, statistical_summary
)


LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.6, 1.0]


def main():
    parser = argparse.ArgumentParser(description="A12: lambda sensitivity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a12/")
    parser.add_argument("--lambdas",    nargs="+", type=float, default=LAMBDA_VALUES)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A12: SHARED REWARD COUPLING λ")
    print(f"  Testing: λ ∈ {args.lambdas}")
    print("="*60)

    results = []
    for lam in args.lambdas:
        print(f"\nCondition: λ = {lam}")
        with patch_weights(lambda_shared=lam):
            r = eval_marl_policy(
                checkpoint_path=args.checkpoint,
                config=cfg,
                n_episodes=args.episodes,
                stoch_level=args.stoch,
                name=f"λ={lam}",
            )
        results.append(r)

    print("\n--- COMPARISON TABLE ---")
    kpis = ["failures", "n_PM", "n_CM", "completions",
            "service_level", "avg_health", "tardiness"]
    table = compare_table(results, kpis=kpis, reference_name="λ=0.3")

    # Key trends
    print("\n--- FAILURES AND THROUGHPUT vs λ ---")
    print(f"{'λ':>6} {'Failures/ep':>14} {'Jobs Done/ep':>14} {'Service Level':>15}")
    for r in results:
        f  = np.mean(r.kpi_data["failures"])
        c  = np.mean(r.kpi_data["completions"])
        sl = np.mean(r.kpi_data["service_level"])
        print(f"  {r.name.split('=')[1]:>4}:  {f:>12.2f}   {c:>12.2f}   {sl:>13.3f}")

    # Statistical test: decoupled (λ=0) vs default (λ=0.3)
    r_zero    = next((r for r in results if "λ=0.0" in r.name or "λ=0" in r.name), None)
    r_default = next((r for r in results if "λ=0.3" in r.name), None)
    if r_zero and r_default:
        print("\n--- STATISTICAL TEST: λ=0 vs λ=0.3 ---")
        stats = statistical_summary(r_default, r_zero, kpis=kpis)
        for kpi, s in stats.items():
            sig = "* p<0.05 *" if s["significant"] else ""
            print(f"  {kpi:<20}: 0.3={s['mean_a']:.3f}  0.0={s['mean_b']:.3f}"
                  f"  p={s['p_welch']:.4f}  d={s['cohen_d']:.2f}  {sig}")

    save_results(results, args.outdir, "a12_lambda")

    report = {
        "ablation": "A12_lambda_sensitivity",
        "lambda_values": args.lambdas,
        "default": 0.3,
        "results": {r.name: r.means for r in results},
    }
    with open(os.path.join(args.outdir, "a12_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.outdir, "a12_table.txt"), "w") as f:
        f.write(table)

    print(f"\n  Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
