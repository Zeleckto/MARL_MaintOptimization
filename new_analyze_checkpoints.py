"""
analyze_checkpoints.py — Learning Curve Across Checkpoints
============================================================
Evaluates each saved checkpoint on N episodes and shows how KPIs
evolved during training — the "learning curve" figure for the paper.

Usage:
    python analyze_checkpoints.py
    python analyze_checkpoints.py --ckpt-dir outputs/checkpoints/ --episodes 30

Output → outputs/results/checkpoints/
    learning_curve.png    (KPI vs checkpoint step)
    checkpoint_kpis.xlsx
"""
import argparse, os, sys, glob, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analytics"))

import yaml
from environments.mfg_env import ManufacturingEnv, AGENT_PDM
from benchmarks.baselines import get_all_baselines
from analytics.episode_kpis import compute_episode_kpis

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analytics.plot_utils import BG, AXES, TEXT, MUTED, MARL_C, GOLD_C, _style_ax


def eval_checkpoint(ckpt_path, cfg, episodes, seed_offset=500):
    """Evaluate a checkpoint on N episodes, return mean KPIs."""
    try:
        import torch
        from agents.pdm_agent import PDMAgent
        from agents.jobshop_agent import JobShopAgent
        from models.critic import CentralizedCritic
        from utils.checkpoint import load_checkpoint

        env = ManufacturingEnv(cfg); env.reset(seed=0)
        obs_dim = len(env._build_agent1_obs())
        a1 = PDMAgent(cfg, device="cpu", obs_dim=obs_dim)
        a2 = JobShopAgent(cfg, device="cpu")
        cr = CentralizedCritic(cfg)
        load_checkpoint(ckpt_path, a1.policy, a2.tgin, cr, device="cpu")
        a1.policy.eval(); a2.tgin.eval()

        kpis = []
        for ep in range(episodes):
            env.reset(seed=seed_offset + ep * 11)
            done = False; steps = 0
            while not done and steps < 200:
                obs1 = env._build_agent1_obs()
                act1, _, _ = a1.act(obs1, env.machine_states, env.machine_busy,
                                     env.resource_state, env.rho_PM, env.rho_CM)
                env._step_agent1(act1)
                env._step_agent2(0 if not env._valid_pairs else
                                  a2.act(env._build_agent2_obs(), env._valid_pairs)[0])
                env._resolve_physics(); env._compute_rewards()
                done  = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
                steps += 1
            kpis.append(compute_episode_kpis(env, steps))

        return {k: float(np.mean([e[k] for e in kpis])) for k in kpis[0]}
    except Exception as ex:
        print(f"  Checkpoint {os.path.basename(ckpt_path)}: {ex}")
        return None


def extract_step(ckpt_name):
    """Extract step number from filename like phase1_step_050k.pt."""
    m = re.search(r"step_(\d+)k", ckpt_name)
    if m: return int(m.group(1)) * 1000
    m = re.search(r"(\d+)k", ckpt_name)
    if m: return int(m.group(1)) * 1000
    return 0


def make_learning_curve(checkpoint_data, outpath):
    """Multi-panel learning curve: KPI vs training steps."""
    if not checkpoint_data: return
    steps = sorted(checkpoint_data.keys())
    if not steps: return

    metrics = [
        ("availability",       "Availability",         MARL_C),
        ("failures",           "Failures / ep",        "#FF3246"),
        ("service_level",      "Service Level",        "#00DC6E"),
        ("weighted_tardiness", "Wt. Tardiness",        "#FFB900"),
        ("n_PM",               "PM Events / ep",       "#B43CFF"),
        ("mtbf",               "MTBF (shifts)",        "#00C8FF"),
        ("mean_rul_norm",      "Mean RUL (norm)",      "#FF6B35"),
        ("total_cost",         "Total Cost",           "#FF3246"),
    ]

    first = checkpoint_data[steps[0]]
    present = [(m, l, c) for m, l, c in metrics if m in first]

    n = len(present); cols = 4; rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.2))
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten()

    for i, (metric, label, colour) in enumerate(present):
        ax = axes_flat[i]
        x  = [s / 1000 for s in steps]  # convert to k-steps
        y  = [checkpoint_data[s].get(metric, 0) for s in steps]
        ax.plot(x, y, color=colour, marker="o", markersize=5,
                linewidth=1.8, markeredgecolor="#2A2D35", markeredgewidth=0.8)
        # Add target line for availability
        if metric == "availability":
            ax.axhline(0.85, color=GOLD_C, linestyle=":", linewidth=1, alpha=0.8)
        _style_ax(ax, title=label, xlabel="Training (k steps)", ylabel="")

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("KPI Learning Curve (per checkpoint)", color=TEXT,
                 size=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def write_checkpoint_excel(checkpoint_data, baseline_means, outpath):
    """Write checkpoint KPIs vs baseline reference to Excel."""
    try:
        import openpyxl
        from analytics.excel_writer import _cell, HEADER, BG, TEXT_FG, GOLD, SUB_FG, MARL_BG, MARL_FG, BEST_FG, WORST_FG
        wb = openpyxl.Workbook()
        ws = wb.active; ws.title = "Learning Curve"
        ws.sheet_view.showGridLines = False

        steps   = sorted(checkpoint_data.keys())
        if not steps: return
        kpi_keys = list(checkpoint_data[steps[0]].keys())
        lower_b  = {"failures","weighted_tardiness","total_cost","failure_cost","n_CM"}

        # Headers
        headers = ["KPI"] + [f"step_{s//1000}k" for s in steps]
        if baseline_means:
            headers += list(baseline_means.keys())
        for ci, h in enumerate(headers, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = 14
            _cell(ws, 1, ci, h, HEADER, GOLD, bold=True)

        for ri, kpi in enumerate(kpi_keys, 2):
            _cell(ws, ri, 1, kpi, BG, TEXT_FG)
            vals = [checkpoint_data[s].get(kpi, 0) for s in steps]
            best = min(vals) if kpi in lower_b else max(vals)
            for ci, (s, v) in enumerate(zip(steps, vals), 2):
                bg = MARL_BG if v == best else BG
                fg = BEST_FG if v == best else TEXT_FG
                _cell(ws, ri, ci, round(v, 4), bg, fg)
            # Baseline reference columns
            if baseline_means:
                for ci, (bl_name, bl_kpis) in enumerate(baseline_means.items(),
                                                          2 + len(steps)):
                    bl_v = bl_kpis.get(kpi, 0)
                    _cell(ws, ri, ci, round(bl_v, 4), BG, SUB_FG)

        os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
        wb.save(outpath)
        print(f"  Saved: {outpath}")
    except Exception as ex:
        print(f"  Excel skipped: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints/")
    ap.add_argument("--episodes",  type=int, default=20)
    ap.add_argument("--outdir",    default="outputs/results/checkpoints/")
    ap.add_argument("--config",    default="configs/base.yaml")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Find milestone checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*step_*k*.pt")))
    if not ckpts:
        print(f"  No milestone checkpoints found in {args.ckpt_dir}")
        print("  Looking for *step_*k*.pt pattern")
        return

    print(f"\n{'='*55}")
    print(f"  CHECKPOINT LEARNING CURVE ({len(ckpts)} checkpoints × {args.episodes} eps)")
    print(f"{'='*55}")

    checkpoint_data = {}
    for ckpt in ckpts:
        step = extract_step(os.path.basename(ckpt))
        print(f"  Evaluating: {os.path.basename(ckpt)} (step={step:,})")
        kpis = eval_checkpoint(ckpt, cfg, args.episodes)
        if kpis:
            checkpoint_data[step] = kpis
            print(f"    avail={kpis.get('availability',0):.3f}  "
                  f"fail={kpis.get('failures',0):.1f}  "
                  f"svc={kpis.get('service_level',0):.3f}")

    if not checkpoint_data:
        print("  No checkpoint data collected."); return

    # Run one baseline for comparison reference
    print("\n  Running ABR baseline for reference...")
    baselines = get_all_baselines()
    abr = next((b for b in baselines if "ABR" in b.name), baselines[-1])
    env = ManufacturingEnv(cfg)
    bl_kpis = []
    for ep in range(min(args.episodes, 20)):
        abr.reset(); env.reset(seed=ep*17)
        done = False; steps = 0
        while not done and steps < 200:
            env._step_agent1(abr.agent1_action(env))
            env._step_agent2(abr.agent2_action(env))
            env._resolve_physics(); env._compute_rewards()
            done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
            steps += 1
        bl_kpis.append(compute_episode_kpis(env, steps))
    bl_means = {abr.name: {k: float(np.mean([e[k] for e in bl_kpis])) for k in bl_kpis[0]}}

    make_learning_curve(checkpoint_data,
                        os.path.join(args.outdir, "learning_curve.png"))
    write_checkpoint_excel(checkpoint_data, bl_means,
                           os.path.join(args.outdir, "checkpoint_kpis.xlsx"))
    print(f"\n  All outputs: {args.outdir}\n")


if __name__ == "__main__":
    main()