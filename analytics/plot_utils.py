"""
analytics/plot_utils.py
========================
Conference-quality figures: dark background, consistent style.
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

BG   = "#0A0C12"; AXES = "#13151E"; TEXT = "#FAFAFA"; MUTED = "#556677"
PALETTE = ["#FF3246","#FFB900","#B43CFF","#00DC6E","#00C8FF"]
MARL_C  = "#00C8FF"; GOLD_C = "#FFB900"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AXES)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2A2D35")
    ax.grid(True, color="#1E2230", linewidth=0.6, linestyle="--", alpha=0.7)
    if title:   ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    if xlabel:  ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.title.set_text(title)

def bar_comparison(data: dict, metrics: list, outpath: str,
                   lower_better: set = None, title="Baseline Comparison"):
    """
    Grouped bar chart: policies × metrics.
    data = {policy_name: {metric: (mean, std), ...}}
    """
    lower_better = lower_better or set()
    n_metrics = len(metrics)
    n_policies = len(data)
    policies = list(data.keys())
    colors = PALETTE[:n_policies]

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 5))
    fig.patch.set_facecolor(BG)
    if n_metrics == 1: axes = [axes]

    for ax, (mkey, mlabel) in zip(axes, metrics):
        vals = [data[p].get(mkey, (0,0))[0] for p in policies]
        errs = [data[p].get(mkey, (0,0))[1] for p in policies]
        x    = np.arange(n_policies)
        bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.88,
                      edgecolor="#2A2D35", linewidth=0.8, capsize=4,
                      error_kw={"ecolor": MUTED, "linewidth": 1.5})
        # Colour best bar gold
        best_idx = np.argmin(vals) if mkey in lower_better else np.argmax(vals)
        bars[best_idx].set_edgecolor(GOLD_C)
        bars[best_idx].set_linewidth(2.5)
        # MARL bar cyan edge
        marl_idx = next((i for i, p in enumerate(policies) if "MARL" in p), None)
        if marl_idx is not None:
            bars[marl_idx].set_edgecolor(MARL_C)
            bars[marl_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split()[0] for p in policies], rotation=30, ha="right",
                           color=MUTED, fontsize=7)
        _style_ax(ax, title=mlabel)

    fig.suptitle(title, color=TEXT, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def radar_chart(data: dict, metrics: list, outpath: str, title="Radar"):
    """
    Spider/radar chart.
    data = {policy_name: {metric: normalised_score in [0,1]}}
    metrics = [(key, label), ...]
    """
    policies = list(data.keys())
    colors   = PALETTE[:len(policies)]
    n = len(metrics)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG); ax.set_facecolor(AXES)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], color=TEXT, size=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], color=MUTED, size=7)
    ax.yaxis.grid(True, color="#2A2D35"); ax.xaxis.grid(True, color="#2A2D35")

    for p, c in zip(policies, colors):
        vals = [data[p].get(k, 0) for k, _ in metrics]
        vals += [vals[0]]
        lw = 2.5 if "MARL" in p else 1.5
        ls = "-"  if "MARL" in p else "--"
        ax.plot(angles, vals, color=c, linewidth=lw, linestyle=ls, label=p, alpha=0.9)
        ax.fill(angles, vals, color=c, alpha=0.07)

    ax.set_title(title, color=TEXT, size=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=AXES, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def training_curves(tb_data: dict, outpath: str, title="Training Progress"):
    """
    Multi-panel training curves from TensorBoard data.
    tb_data = {tag: {"steps": arr, "values": arr}}
    """
    panels = [
        ("episode/availability",    "Availability",      True),
        ("episode/failures",        "Failures / ep",     False),
        ("train/critic_loss",       "Critic Loss",       False),
        ("train/entropy1",          "Policy Entropy A1", False),
        ("rewards/agent1_r1",       "r1 / step",         True),
        ("rewards/agent2_r2",       "r2 / step",         True),
        ("rewards/shared",          "r_shared / step",   False),
        ("episode/n_PM",            "PM Events / ep",    True),
    ]
    present = [(t, l, g) for t, l, g in panels if t in tb_data]
    if not present: print("  No TB tags found for training curves."); return

    n = len(present)
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.patch.set_facecolor(BG)
    axes = np.array(axes).flatten()

    for i, (tag, label, good_up) in enumerate(present):
        ax = axes[i]
        d = tb_data[tag]
        x, y = d["steps"], d["values"]
        # Smooth
        window = max(len(y) // 20, 1)
        y_sm = np.convolve(y, np.ones(window)/window, mode="valid")
        x_sm = x[:len(y_sm)]
        ax.plot(x, y, alpha=0.2, color=MARL_C, linewidth=0.7)
        ax.plot(x_sm, y_sm, color=MARL_C, linewidth=1.8, label="smoothed")
        _style_ax(ax, title=label, xlabel="Step", ylabel="")
        col = MARL_C if good_up else "#FF3246"
        ax.yaxis.label.set_color(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, color=TEXT, size=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def lambda_sensitivity_plot(lambda_vals, metric_vals, metric_name, outpath):
    fig, ax = plt.subplots(figsize=(6, 4)); fig.patch.set_facecolor(BG)
    ax.plot(lambda_vals, metric_vals, color=MARL_C, marker="o",
            linewidth=2, markersize=7)
    ax.axvline(lambda_vals[np.argmax(metric_vals) if "avail" in metric_name.lower()
                           else np.argmin(metric_vals)],
               color=GOLD_C, linestyle="--", linewidth=1.5, label="Optimal λ")
    _style_ax(ax, title=f"λ Sensitivity — {metric_name}",
              xlabel="λ (R_shared coupling)", ylabel=metric_name)
    ax.legend(facecolor=AXES, edgecolor=MUTED, labelcolor=TEXT)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {outpath}")
