"""
analyze_training.py
====================
Reads TensorBoard event files from runs/ and generates a comprehensive
training analysis report with matplotlib figures.

Usage:
    python analyze_training.py                    # auto-picks latest run
    python analyze_training.py --run runs/phase1_1774584774
    python analyze_training.py --run runs/phase1_1774584774 --smooth 20

Outputs:
    results/training_analysis.png   -- main figure (9 panels)
    results/training_report.txt     -- plain-text summary
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")


# ── PALETTE ──────────────────────────────────────────────────────────────────
TEAL   = "#0A9396"
AMBER  = "#E9C46A"
CORAL  = "#E76F51"
GREEN  = "#2A9D8F"
NAVY   = "#0D1B2A"
SLATE  = "#264653"
MINT   = "#94D2BD"
LAVEND = "#7B9EBC"
BGCOL  = "#F8FAFB"
GRIDCOL= "#E0ECEE"


# ── LOAD TENSORBOARD EVENTS ───────────────────────────────────────────────────

def load_run(run_dir: str) -> dict:
    """
    Reads all scalar events from a TensorBoard run directory.
    Returns dict: {tag: (steps[], values[])}
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("pip install tensorboard")
        sys.exit(1)

    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)

    print(f"Loaded {len(data)} metrics from {run_dir}")
    for tag in sorted(data.keys()):
        print(f"  {tag}  ({len(data[tag][0])} points)")
    return data


def smooth(values, window=10):
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    out   = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out


def get(data, tag, default_steps=None):
    """Safe getter — returns (steps, raw, smoothed) or zeros."""
    if tag in data:
        steps, vals = data[tag]
        return steps, vals, smooth(vals)
    if default_steps is not None:
        z = np.zeros(len(default_steps))
        return default_steps, z, z
    return np.array([0,1]), np.array([0,0]), np.array([0,0])


# ── EARLY vs LATE WINDOWS ─────────────────────────────────────────────────────

def split_early_late(steps, values, frac=0.2):
    """
    Splits into early (first frac) and late (last frac) windows.
    Returns (early_mean, late_mean, early_std, late_std).
    """
    n = len(steps)
    if n < 4:
        return float(np.mean(values)), float(np.mean(values)), 0.0, 0.0
    cut = max(1, int(n * frac))
    early = values[:cut]
    late  = values[max(0, n-cut):]
    return float(np.mean(early)), float(np.mean(late)), float(np.std(early)), float(np.std(late))


# ── MAIN PLOT ─────────────────────────────────────────────────────────────────

def make_figure(data: dict, run_dir: str, smooth_win: int, out_path: str):

    fig = plt.figure(figsize=(20, 24), facecolor=BGCOL)
    fig.suptitle(
        "Manufacturing MARL — Training Analysis\nThree-Tier Optimization Framework (Phase 1)",
        fontsize=18, fontweight="bold", color=NAVY, y=0.98,
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.94, bottom=0.04, left=0.07, right=0.97,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]

    # Reference step array
    ref_steps = None
    for tag in data:
        s, _ = data[tag]
        if ref_steps is None or len(s) > len(ref_steps):
            ref_steps = s
    if ref_steps is None:
        ref_steps = np.arange(100)

    # ── Helper ────────────────────────────────────────────────────────────────
    def plot_metric(ax, tag, title, ylabel, color, invert=False, baseline=None,
                    fill=True, ymin=None):
        steps, raw, smo = get(data, tag, ref_steps)

        ax.set_facecolor(BGCOL)
        ax.grid(True, color=GRIDCOL, linewidth=0.8, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(GRIDCOL)

        if fill:
            ax.fill_between(steps, raw, alpha=0.12, color=color)
        ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, smo, color=color, linewidth=2.2, label="EMA")

        if baseline is not None:
            ax.axhline(baseline, color=CORAL, linewidth=1.2,
                       linestyle="--", alpha=0.7, label=f"Baseline={baseline:.1f}")

        # Early / late annotation
        e_mean, l_mean, _, _ = split_early_late(steps, raw)
        change = l_mean - e_mean
        pct    = (change / abs(e_mean) * 100) if abs(e_mean) > 1e-6 else 0.0
        sign   = "(+)" if change > 0 else "(-)"
        good   = (change > 0) != invert
        ann_col = GREEN if good else CORAL

        ax.set_title(title, fontsize=12, fontweight="bold", color=SLATE, pad=6)
        ax.set_xlabel("Training Step", fontsize=9, color=SLATE)
        ax.set_ylabel(ylabel, fontsize=9, color=SLATE)
        ax.tick_params(colors=SLATE, labelsize=8)

        # Annotation box
        ax.text(0.97, 0.05,
                f"Early: {e_mean:.2f}\nLate:  {l_mean:.2f}\n{sign} {abs(pct):.0f}%",
                transform=ax.transAxes, fontsize=8, color=ann_col,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=ann_col, alpha=0.9))

        if ymin is not None:
            ax.set_ylim(bottom=ymin)

    # ── PANEL 1: Agent 1 reward ───────────────────────────────────────────────
    plot_metric(axes[0], "rewards/agent1_r1",
                "Agent 1 Reward (PDM)", "r₁ per step", TEAL)

    # ── PANEL 2: Agent 2 reward ───────────────────────────────────────────────
    plot_metric(axes[1], "rewards/agent2_r2",
                "Agent 2 Reward (Scheduling)", "r₂ per step", LAVEND)

    # ── PANEL 3: Shared failure penalty ──────────────────────────────────────
    plot_metric(axes[2], "rewards/shared",
                "Shared Failure Penalty", "R_shared per step", CORAL,
                invert=True, baseline=0.0, ymin=None)

    # ── PANEL 4: Episode failures ─────────────────────────────────────────────
    plot_metric(axes[3], "episode/failures",
                "Machine Failures per Episode", "# failures", CORAL,
                invert=True, ymin=0)

    # ── PANEL 5: Jobs completed ───────────────────────────────────────────────
    plot_metric(axes[4], "episode/jobs_completed",
                "Jobs Completed per Episode", "# jobs", GREEN, ymin=0)

    # ── PANEL 6: Weighted tardiness ───────────────────────────────────────────
    plot_metric(axes[5], "episode/weighted_tardiness",
                "Weighted Tardiness per Episode", "Σ wⱼ·tardⱼ", AMBER,
                invert=True, ymin=0)

    # ── PANEL 7: Machine health ───────────────────────────────────────────────
    plot_metric(axes[6], "episode/avg_machine_health",
                "Avg Machine Health", "health %", GREEN, ymin=0)

    # ── PANEL 8: Actor 1 loss ─────────────────────────────────────────────────
    plot_metric(axes[7], "train/actor1_loss",
                "Agent 1 PPO Loss", "actor1 loss", TEAL, invert=True)

    # ── PANEL 9: Entropy ─────────────────────────────────────────────────────
    plot_metric(axes[8], "train/entropy1",
                "Agent 1 Policy Entropy", "entropy", MINT)

    # ── PANEL 10: Episode return Agent 1 ─────────────────────────────────────
    plot_metric(axes[9], "episode/return_agent1",
                "Episode Return — Agent 1", "total r1", TEAL)

    # ── PANEL 11: Episode return Agent 2 ─────────────────────────────────────
    plot_metric(axes[10], "episode/return_agent2",
                "Episode Return — Agent 2", "total r2", LAVEND)

    # ── PANEL 12: Episode length ──────────────────────────────────────────────
    plot_metric(axes[11], "episode/length",
                "Episode Length", "steps", AMBER)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BGCOL)
    print(f"Figure saved: {out_path}")
    return fig


# ── EARLY vs LATE BAR CHART ───────────────────────────────────────────────────

def make_early_late_chart(data: dict, out_path: str):
    """
    Bar chart comparing early training vs late training on key KPIs.
    This is the main 'before vs after' result figure for the paper.
    """
    metrics = [
        ("episode/failures",          "Machine Failures",    True,  CORAL),
        ("episode/jobs_completed",     "Jobs Completed",      False, GREEN),
        ("episode/weighted_tardiness", "Weighted Tardiness",  True,  AMBER),
        ("episode/avg_machine_health", "Avg Health (%)",      False, TEAL),
        ("episode/return_agent1",      "Agent 1 Return",      False, LAVEND),
        ("episode/return_agent2",      "Agent 2 Return",      False, MINT),
    ]

    labels   = [m[0] for m in metrics]
    titles   = [m[1] for m in metrics]
    inverted = [m[2] for m in metrics]
    colors   = [m[3] for m in metrics]

    early_means, late_means = [], []
    early_stds,  late_stds  = [], []

    ref = None
    for tag in data:
        s, _ = data[tag]
        if ref is None or len(s) > len(ref):
            ref = s

    for tag, *_ in metrics:
        steps, vals = data.get(tag, (ref, np.zeros(len(ref))))
        e_m, l_m, e_s, l_s = split_early_late(steps, vals)
        early_means.append(e_m); early_stds.append(e_s)
        late_means.append(l_m);  late_stds.append(l_s)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BGCOL)
    fig.suptitle(
        "Training Outcome: Early vs Late Training Window\n"
        "(First 20% vs Last 20% of training steps)",
        fontsize=15, fontweight="bold", color=NAVY, y=1.01,
    )

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(BGCOL)
        ax.grid(True, axis="y", color=GRIDCOL, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color(GRIDCOL)

        em = early_means[i]; lm = late_means[i]
        es = early_stds[i];  ls = late_stds[i]
        col = colors[i]

        bars = ax.bar(
            ["Early\nTraining", "Late\nTraining"],
            [em, lm],
            yerr=[es, ls],
            color=[col + "66", col],   # light / dark
            edgecolor=[col, col],
            linewidth=1.5,
            error_kw=dict(ecolor=NAVY, elinewidth=1.2, capsize=5),
            width=0.5,
        )

        # Value labels on bars
        for bar, val in zip(bars, [em, lm]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(em), abs(lm)) * 0.03,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=NAVY,
            )

        # Change arrow
        change = lm - em
        pct    = (change / abs(em) * 100) if abs(em) > 1e-6 else 0.0
        good   = (change > 0) != inverted[i]
        sign   = "(+)" if change > 0 else "(-)"
        c      = GREEN if good else CORAL

        ax.set_title(titles[i], fontsize=12, fontweight="bold", color=SLATE)
        ax.tick_params(colors=SLATE, labelsize=9)

        ax.text(0.5, 0.97, f"({'up' if change>0 else 'down'}) {abs(pct):.1f}%",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=c, ha="center", va="top")

        ypad = max(abs(em), abs(lm)) * 0.15
        ax.set_ylim(
            min(0, em - es - ypad) if em < 0 else -(ypad),
            max(abs(em), abs(lm)) * 1.3
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BGCOL)
    print(f"Early/Late chart saved: {out_path}")


# ── MAINTENANCE ACTION BREAKDOWN ─────────────────────────────────────────────

def make_maintenance_chart(data: dict, out_path: str):
    """
    Shows Agent 1's maintenance policy evolution over training.
    Proxied from reward components if direct PM/CM counts not logged.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BGCOL)
    fig.suptitle(
        "Agent 1 — Predictive Maintenance Policy Analysis",
        fontsize=14, fontweight="bold", color=NAVY,
    )

    # Panel 1: Availability + health over training
    ax = axes[0]
    ax.set_facecolor(BGCOL)
    ax.grid(True, color=GRIDCOL, linewidth=0.8)

    if "episode/avg_machine_health" in data:
        steps, vals = data["episode/avg_machine_health"]
        smo = smooth(vals, 15)
        ax.fill_between(steps, vals, alpha=0.12, color=GREEN)
        ax.plot(steps, vals, color=GREEN, alpha=0.2, linewidth=0.8)
        ax.plot(steps, smo, color=GREEN, linewidth=2.5, label="Avg Health %")

    if "episode/failures" in data:
        ax2 = ax.twinx()
        steps, vals = data["episode/failures"]
        smo = smooth(vals, 15)
        ax2.fill_between(steps, vals, alpha=0.08, color=CORAL)
        ax2.plot(steps, smo, color=CORAL, linewidth=2.0,
                 linestyle="--", label="Failures (right)")
        ax2.set_ylabel("Failures per Episode", color=CORAL, fontsize=10)
        ax2.tick_params(colors=CORAL)
        ax2.set_ylim(bottom=0)

    ax.set_title("Machine Health vs Failures Over Training",
                 fontsize=11, fontweight="bold", color=SLATE)
    ax.set_xlabel("Training Step", fontsize=9, color=SLATE)
    ax.set_ylabel("Avg Health (%)", color=GREEN, fontsize=10)
    ax.tick_params(colors=SLATE, labelsize=8)
    ax.set_ylim(0, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc="lower left", fontsize=8)

    # Panel 2: Agent 1 reward breakdown by component
    ax = axes[1]
    ax.set_facecolor(BGCOL)
    ax.grid(True, color=GRIDCOL, linewidth=0.8)

    plotted = False
    for tag, label, col in [
        ("rewards/agent1_r1", "Total r1",         TEAL),
        ("rewards/shared",    "Failure Penalty",   CORAL),
        ("rewards/total",     "Combined Reward",   NAVY),
    ]:
        if tag in data:
            steps, vals = data[tag]
            smo = smooth(vals, 15)
            ax.plot(steps, smo, color=col, linewidth=2.0, label=label)
            ax.fill_between(steps, smo, alpha=0.08, color=col)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "Reward component logs\nnot found in this run",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=SLATE)

    ax.axhline(0, color=NAVY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Reward Components Over Training",
                 fontsize=11, fontweight="bold", color=SLATE)
    ax.set_xlabel("Training Step", fontsize=9, color=SLATE)
    ax.set_ylabel("Reward Value", fontsize=9, color=SLATE)
    ax.tick_params(colors=SLATE, labelsize=8)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BGCOL)
    print(f"Maintenance chart saved: {out_path}")


# ── TEXT REPORT ───────────────────────────────────────────────────────────────

def make_report(data: dict, run_dir: str, out_path: str):
    """Generates plain-text summary report."""

    ref = None
    for tag in data:
        s, _ = data[tag]
        if ref is None or len(s) > len(ref):
            ref = s

    total_steps = int(ref[-1]) if len(ref) > 0 else 0

    lines = []
    lines.append("=" * 60)
    lines.append("MANUFACTURING MARL — TRAINING ANALYSIS REPORT")
    lines.append("Three-Tier Optimization Framework (Phase 1)")
    lines.append("=" * 60)
    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Total training steps: {total_steps:,}")
    lines.append(f"Metrics logged: {len(data)}")
    lines.append("")

    report_metrics = [
        ("episode/failures",           "Machine Failures/Episode",  True),
        ("episode/jobs_completed",      "Jobs Completed/Episode",    False),
        ("episode/weighted_tardiness",  "Weighted Tardiness",        True),
        ("episode/avg_machine_health",  "Avg Machine Health (%)",    False),
        ("episode/return_agent1",       "Agent 1 Episode Return",    False),
        ("episode/return_agent2",       "Agent 2 Episode Return",    False),
        ("episode/length",              "Episode Length (steps)",    None),
        ("train/actor1_loss",           "Agent 1 PPO Loss",          True),
        ("train/entropy1",              "Policy Entropy (Agent 1)",  None),
    ]

    lines.append("KEY METRICS: EARLY vs LATE TRAINING")
    lines.append("-" * 60)
    lines.append(f"{'Metric':<35} {'Early':>8} {'Late':>8} {'Change':>10}")
    lines.append("-" * 60)

    for tag, name, lower_better in report_metrics:
        if tag not in data:
            continue
        steps, vals = data[tag]
        e_m, l_m, e_s, l_s = split_early_late(steps, vals)
        change = l_m - e_m
        pct    = (change / abs(e_m) * 100) if abs(e_m) > 1e-6 else 0.0
        sign   = "(+)" if change > 0 else "(-)"

        if lower_better is not None:
            good = (change < 0) if lower_better else (change > 0)
            flag = " ✓" if good else " ✗"
        else:
            flag = ""

        lines.append(
            f"{name:<35} {e_m:>8.2f} {l_m:>8.2f} "
            f"{sign}{abs(pct):>7.1f}%{flag}"
        )

    lines.append("")
    lines.append("TRAINING CONVERGENCE ASSESSMENT")
    lines.append("-" * 60)

    # Check key convergence signals
    checks = []

    if "episode/failures" in data:
        _, vals = data["episode/failures"]
        e_m, l_m, _, _ = split_early_late(np.arange(len(vals)), vals)
        if l_m < e_m:
            checks.append(("✓", "Failure rate DECREASING — Agent 1 learning PM"))
        elif l_m == 0 and e_m == 0:
            checks.append(("✓", "Zero failures throughout — stable environment"))
        else:
            checks.append(("✗", "Failure rate not decreasing — check reward weights"))

    if "train/entropy1" in data:
        _, vals = data["train/entropy1"]
        e_m, l_m, _, _ = split_early_late(np.arange(len(vals)), vals)
        if l_m < e_m * 0.5:
            checks.append(("✗", "Entropy collapsed too fast — reduce lr or increase entropy_coef"))
        elif l_m < e_m:
            checks.append(("✓", "Entropy decreasing gradually — healthy exploration"))
        else:
            checks.append(("~", "Entropy stable — policy still exploring"))

    if "episode/jobs_completed" in data:
        _, vals = data["episode/jobs_completed"]
        e_m, l_m, _, _ = split_early_late(np.arange(len(vals)), vals)
        if l_m >= e_m:
            checks.append(("✓", f"Job completion maintained ({l_m:.1f}/episode late)"))
        else:
            checks.append(("✗", "Job completion declining — check Agent 2"))

    if "episode/avg_machine_health" in data:
        _, vals = data["episode/avg_machine_health"]
        late_health = np.mean(vals[-max(1, len(vals)//5):])
        if late_health > 75:
            checks.append(("✓", f"Machine health maintained above 75% ({late_health:.1f}%)"))
        else:
            checks.append(("✗", f"Machine health low ({late_health:.1f}%) — PM not proactive enough"))

    for sym, msg in checks:
        lines.append(f"  {sym}  {msg}")

    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 60)
    lines.append("  • If r1 is trending up: proceed to 500k step full Phase 1 run")
    lines.append("  • If failures > 0:     increase c_fail in reward_weights.yaml")
    lines.append("  • If entropy < 0.1:    increase entropy_coef in base.yaml mappo section")
    lines.append("  • If jobs_completed=0: increase w_comp in reward_weights.yaml")
    lines.append("  • After Phase 1 converges: run with configs/phase2.yaml --resume")
    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 60)
    lines.append("  results/training_analysis.png   — 12-panel training curves")
    lines.append("  results/early_late_comparison.png — KPI before vs after")
    lines.append("  results/maintenance_analysis.png  — PDM policy analysis")
    lines.append("  results/training_report.txt      — this report")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved: {out_path}")
    return report_text


# ── MAIN ─────────────────────────────────────────────────────────────────────

def find_latest_run(runs_dir="runs"):
    """Auto-finds the most recently modified run directory."""
    if not os.path.exists(runs_dir):
        return None
    candidates = glob.glob(os.path.join(runs_dir, "*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",    default=None,
                        help="Path to TensorBoard run dir (auto-detects latest if omitted)")
    parser.add_argument("--smooth", type=int, default=15,
                        help="EMA smoothing window (default: 15)")
    parser.add_argument("--outdir", default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    # Find run directory
    run_dir = args.run
    if run_dir is None:
        run_dir = find_latest_run()
        if run_dir is None:
            print("No runs/ directory found. Run training first:")
            print("  python scripts/train.py --config configs/phase1.yaml --timesteps 500000")
            sys.exit(1)
        print(f"Auto-detected run: {run_dir}")

    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    data = load_run(run_dir)

    if not data:
        print("No scalar data found in run directory.")
        print("Make sure training has run for at least a few episodes.")
        sys.exit(1)

    # Generate outputs
    make_figure(
        data, run_dir, args.smooth,
        os.path.join(args.outdir, "training_analysis.png"),
    )
    make_early_late_chart(
        data,
        os.path.join(args.outdir, "early_late_comparison.png"),
    )
    make_maintenance_chart(
        data,
        os.path.join(args.outdir, "maintenance_analysis.png"),
    )
    make_report(
        data, run_dir,
        os.path.join(args.outdir, "training_report.txt"),
    )

    print(f"\nAll outputs saved to: {args.outdir}/")


if __name__ == "__main__":
    main()