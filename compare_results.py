"""
compare_results.py — Statistical comparison of MARL vs baselines
=================================================================
Run after you have both baseline Excel and MARL results.

Usage:
    python compare_results.py \\
        --baselines results/final/baseline_comparison.xlsx \\
        --marl      results/final/marl_results.xlsx

Outputs:
    - Significance table (t-test + Mann-Whitney U)
    - Effect sizes (Cohen's d)
    - Win/loss/tie summary
    - results/final/statistical_comparison.xlsx
"""
import argparse, sys, os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))

# Metrics where LOWER is better
LOWER_BETTER = {"failures", "weighted_tardiness", "n_CM", "jobs_late", "avg_hazard_rate"}
# Metrics where HIGHER is better  
HIGHER_BETTER = {"service_level", "jobs_completed", "mtbf", "avg_health", "n_PM"}

def cohens_d(a, b):
    """Effect size: |mean_a - mean_b| / pooled_std"""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return abs(np.mean(a) - np.mean(b)) / max(pooled, 1e-10)

def interpret_d(d):
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def marl_wins(metric, marl_mean, baseline_mean):
    """True if MARL is better on this metric"""
    if any(k in metric.lower() for k in LOWER_BETTER):
        return marl_mean < baseline_mean
    return marl_mean > baseline_mean


def compare(baselines_path, marl_path, outdir):
    """Run full statistical comparison."""
    os.makedirs(outdir, exist_ok=True)

    try:
        bl_df   = pd.read_excel(baselines_path, sheet_name=None)
        marl_df = pd.read_excel(marl_path, sheet_name=None)
    except Exception as e:
        print(f"Could not load Excel files: {e}")
        print("Make sure you've run analyze_baselines.py and have MARL results.")
        return

    # Expect episode-level data in sheet named 'episodes' or first sheet
    def get_episodes(df_dict):
        for name in ['episodes', 'Episodes', 'raw', 'data']:
            if name in df_dict: return df_dict[name]
        return list(df_dict.values())[0]

    marl_eps = get_episodes(marl_df)
    metrics  = [c for c in marl_eps.columns
                if c not in ('episode', 'seed', 'baseline') and
                marl_eps[c].dtype in [float, int, 'float64', 'int64']]

    print(f"\n{'='*70}")
    print(f"  MARL vs Baselines — Statistical Comparison")
    print(f"  Metrics: {metrics}")
    print(f"{'='*70}\n")

    rows = []
    win_counts = {}

    for bl_name, bl_sheet in bl_df.items():
        if bl_name in ('summary', 'Summary'): continue
        if 'episode' not in str(bl_sheet.columns).lower(): continue

        wins = 0
        print(f"\n--- MARL vs {bl_name} ---")
        print(f"{'Metric':25s} {'MARL':>8s} {'Baseline':>10s} "
              f"{'t-p':>7s} {'MW-p':>7s} {'d':>6s} {'effect':>10s} {'winner':>8s}")
        print("-" * 90)

        for metric in metrics:
            if metric not in marl_eps.columns or metric not in bl_sheet.columns:
                continue

            marl_vals = marl_eps[metric].dropna().values
            bl_vals   = bl_sheet[metric].dropna().values

            if len(marl_vals) < 3 or len(bl_vals) < 3:
                continue

            # Two-sided t-test and Mann-Whitney
            _, p_t  = stats.ttest_ind(marl_vals, bl_vals)
            _, p_mw = stats.mannwhitneyu(marl_vals, bl_vals, alternative='two-sided')
            d = cohens_d(marl_vals, bl_vals)

            m_mean = marl_vals.mean()
            b_mean = bl_vals.mean()
            winner = "MARL" if marl_wins(metric, m_mean, b_mean) else bl_name[:8]
            if marl_wins(metric, m_mean, b_mean):
                wins += 1

            sig = "*" if p_t < 0.05 else ("~" if p_t < 0.1 else " ")
            print(f"{metric:25s} {m_mean:8.3f} {b_mean:10.3f} "
                  f"{p_t:6.3f}{sig} {p_mw:7.3f} {d:6.3f} {interpret_d(d):>10s} {winner:>8s}")

            rows.append({
                "baseline":    bl_name,
                "metric":      metric,
                "marl_mean":   m_mean,
                "marl_std":    marl_vals.std(),
                "bl_mean":     b_mean,
                "bl_std":      bl_vals.std(),
                "p_ttest":     p_t,
                "p_mannwhit":  p_mw,
                "cohens_d":    d,
                "effect_size": interpret_d(d),
                "significant": p_t < 0.05,
                "winner":      winner,
                "n_marl":      len(marl_vals),
                "n_baseline":  len(bl_vals),
            })

        win_counts[bl_name] = (wins, len(metrics))
        print(f"\n  MARL wins {wins}/{len(metrics)} metrics vs {bl_name}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  OVERALL WIN SUMMARY")
    print(f"{'='*70}")
    for bl, (w, total) in win_counts.items():
        bar = "█" * w + "░" * (total - w)
        print(f"  vs {bl:20s}: {w:2d}/{total} [{bar}]")

    # Export
    if rows:
        result_df = pd.DataFrame(rows)
        out_path  = os.path.join(outdir, "statistical_comparison.xlsx")
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="full_results", index=False)

            # Summary pivot
            pivot = result_df.pivot_table(
                index="metric", columns="baseline",
                values=["marl_mean", "bl_mean", "p_ttest", "significant", "cohens_d"],
                aggfunc="first"
            )
            pivot.to_excel(writer, sheet_name="comparison_table")

            # Win counts
            win_df = pd.DataFrame([
                {"baseline": k, "marl_wins": v[0], "total_metrics": v[1],
                 "win_pct": v[0]/v[1]*100}
                for k, v in win_counts.items()
            ])
            win_df.to_excel(writer, sheet_name="win_summary", index=False)

        print(f"\n  Saved: {out_path}")

    print(f"\n  Note: * p<0.05 significant  ~ p<0.1 marginal  (blank) not significant")
    print(f"  Tests: Welch's t-test (parametric) + Mann-Whitney U (non-parametric)")
    print(f"  Effect size d: <0.2 negligible  0.2-0.5 small  0.5-0.8 medium  >0.8 large")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical comparison: MARL vs baselines")
    parser.add_argument("--baselines", required=True,
                        help="Path to baseline_comparison.xlsx from analyze_baselines.py")
    parser.add_argument("--marl",      required=True,
                        help="Path to MARL results Excel (from analyze_baselines.py --checkpoint)")
    parser.add_argument("--outdir",    default="results/final/",
                        help="Output directory for statistical_comparison.xlsx")
    args = parser.parse_args()
    compare(args.baselines, args.marl, args.outdir)
