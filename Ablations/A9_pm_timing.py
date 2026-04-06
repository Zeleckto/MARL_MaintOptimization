"""
A9_pm_timing.py  —  Ablation A9: PM Timing Distribution
=========================================================
Tests whether the trained MARL agent converges to the analytically
optimal ABR threshold (t* ≈ 49–65 shifts for our machines).

This is the single most striking analysis: if the histogram of machine
age at PM initiation clusters near ABR t*, the agent independently
rediscovered the analytical optimum without being told.

ABR optimal thresholds (computed numerically):
  M0 CNC Mill:   t* ≈ 49 shifts  (H* ≈ 72%)
  M1 Lathe:      t* ≈ 61 shifts  (H* ≈ 74%)
  M2 Grinder:    t* ≈ 38 shifts  (H* ≈ 71%)
  M3 Press:      t* ≈ 55 shifts  (H* ≈ 73%)
  M4 CMM:        t* ≈ 75 shifts  (H* ≈ 75%)

Outputs:
  - Histogram: effective_age at PM initiation (hours)
  - Histogram: machine health at PM initiation (%)
  - Per-machine breakdown
  - Comparison vs ABR t* overlay

Usage:
  python ablations/A9_pm_timing.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 50 \\
      --outdir results/ablation_a9/
"""

import argparse, os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from ablations.ablation_utils import load_config


# ABR t* (shifts) and H(t*) (%) for each machine — computed from Weibull params
ABR_T_STAR = {
    "M0 CNC Mill":   {"t_star_sh": 49, "h_star_pct": 72, "beta": 2.8, "eta_h": 900},
    "M1 Lathe":      {"t_star_sh": 61, "h_star_pct": 74, "beta": 2.2, "eta_h": 1100},
    "M2 Grinder":    {"t_star_sh": 38, "h_star_pct": 71, "beta": 3.0, "eta_h": 700},
    "M3 Press":      {"t_star_sh": 55, "h_star_pct": 73, "beta": 3.2, "eta_h": 1000},
    "M4 CMM":        {"t_star_sh": 75, "h_star_pct": 75, "beta": 2.5, "eta_h": 1350},
}


def collect_pm_events(
    checkpoint_path: str,
    cfg: dict,
    n_episodes: int,
    stoch_level: int = 3,
    seed_offset: int = 42,
) -> dict:
    """
    Run MARL policy and log machine state at every PM initiation.
    Returns dict: {machine_id: [{age_h, health, step, episode}]}
    """
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from environments.transitions.degradation import MachineStatus
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent
    from models.critic import CentralizedCritic
    from utils.checkpoint import load_checkpoint
    import torch

    cfg_eval = {**cfg, "stochasticity_level": stoch_level}
    env    = ManufacturingEnv(cfg_eval)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env.reset(seed=0)
    obs1_sample = env._build_agent1_obs()
    obs_dim = len(obs1_sample)

    agent1 = PDMAgent(cfg_eval, device=device, obs_dim=obs_dim)
    agent2 = JobShopAgent(cfg_eval, device=device)
    critic = CentralizedCritic(cfg_eval)
    load_checkpoint(checkpoint_path, agent1, agent2, critic, device=device)
    agent1.eval(); agent2.eval()

    pm_events = {m: [] for m in range(5)}  # {machine_id: [event_dicts]}

    print(f"  Collecting PM events ({n_episodes} episodes)...")
    for ep in range(n_episodes):
        env.reset(seed=seed_offset + ep * 7)
        done, steps = False, 0

        while not done and steps < 300:
            obs1 = env._build_agent1_obs()
            action1, _, _ = agent1.act(
                obs_np=obs1,
                machine_states=env.machine_states,
                machine_busy=env.machine_busy,
                resource_state=env.resource_state,
                rho_PM=env.rho_PM,
                rho_CM=env.rho_CM,
            )

            # Log PM actions BEFORE they're applied (record current state)
            for m_id, maint_act in enumerate(action1["maintenance"]):
                if maint_act == 1:  # PM
                    s = env.machine_states[m_id]
                    if s.status == MachineStatus.OP:  # valid PM
                        eff_age = s.virtual_age + s.time_since_maint
                        pm_events[m_id].append({
                            "episode":     ep,
                            "step":        steps,
                            "eff_age_h":   float(eff_age),
                            "eff_age_sh":  float(eff_age / 8),
                            "health":      float(s.health),
                            "virtual_age": float(s.virtual_age),
                            "t_since_h":   float(s.time_since_maint),
                        })

            env._step_agent1(action1)
            obs2, valid_pairs = env._build_agent2_obs()
            if valid_pairs:
                _, idx, _, _ = agent2.act(obs2, valid_pairs)
                env._step_agent2(idx)
            else:
                env._step_agent2(len(valid_pairs))

            env._resolve_physics()
            env._compute_rewards()
            done = (env.terminations.get(AGENT_PDM, False) or
                    env.truncations.get(AGENT_PDM, False))
            steps += 1

        sys.stdout.write(f"\r    Episode {ep+1}/{n_episodes}")
        sys.stdout.flush()
    print()

    return pm_events


def analyse_and_plot(pm_events: dict, outdir: str):
    """Analyse PM timing and create plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots, saving stats only")
        return

    machine_names = list(ABR_T_STAR.keys())
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("A9: PM Timing Distribution vs ABR Optimal Threshold",
                 fontsize=14, fontweight="bold")

    all_ages_h = []
    all_healths = []

    for m_id, events in pm_events.items():
        if not events:
            continue

        ages_h   = [e["eff_age_h"] for e in events]
        ages_sh  = [e["eff_age_sh"] for e in events]
        healths  = [e["health"] for e in events]
        all_ages_h.extend(ages_h)
        all_healths.extend(healths)

        name = machine_names[m_id] if m_id < len(machine_names) else f"M{m_id}"
        abr  = ABR_T_STAR.get(name, {})
        t_star_h  = abr.get("t_star_sh", 55) * 8
        h_star    = abr.get("h_star_pct", 72)

        # Age histogram
        ax_age = axes[0, m_id]
        ax_age.hist(ages_h, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
        ax_age.axvline(t_star_h, color="red", linestyle="--", linewidth=2,
                       label=f"ABR t*={t_star_h:.0f}h")
        ax_age.axvspan(t_star_h * 0.9, t_star_h * 1.2, alpha=0.1, color="red")
        ax_age.set_xlabel("Age at PM (h)")
        ax_age.set_ylabel("Count")
        ax_age.set_title(f"{name}\nn={len(events)}", fontsize=9)
        ax_age.legend(fontsize=7)

        # Health histogram
        ax_h = axes[1, m_id]
        ax_h.hist(healths, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax_h.axvline(h_star, color="red", linestyle="--", linewidth=2,
                     label=f"H(t*)={h_star}%")
        ax_h.set_xlabel("Health at PM (%)")
        ax_h.set_ylabel("Count")
        ax_h.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(outdir, "a9_pm_timing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")

    # Fleet-wide summary plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("A9: Fleet PM Timing — All Machines Combined", fontsize=12)

    ax1.hist(all_ages_h, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    # ABR range: min to max t* across machines
    t_min = min(v["t_star_sh"] for v in ABR_T_STAR.values()) * 8
    t_max = max(v["t_star_sh"] for v in ABR_T_STAR.values()) * 8
    ax1.axvspan(t_min, t_max, alpha=0.15, color="red",
                label=f"ABR range [{t_min:.0f}–{t_max:.0f}h]")
    ax1.axvline((t_min + t_max) / 2, color="red", linestyle="--", linewidth=2,
                label="ABR midpoint")
    ax1.set_xlabel("Effective Age at PM (hours)")
    ax1.set_ylabel("Count")
    ax1.set_title("Age at PM — All Machines")
    ax1.legend()

    ax2.hist(all_healths, bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
    h_avg = np.mean([v["h_star_pct"] for v in ABR_T_STAR.values()])
    ax2.axvline(h_avg, color="red", linestyle="--", linewidth=2,
                label=f"Avg H(t*) = {h_avg:.0f}%")
    ax2.set_xlabel("Machine Health at PM (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Health at PM — All Machines")
    ax2.legend()

    plt.tight_layout()
    path2 = os.path.join(outdir, "a9_pm_timing_fleet.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fleet plot saved: {path2}")

    return all_ages_h, all_healths


def main():
    parser = argparse.ArgumentParser(description="A9: PM timing distribution")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a9/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A9: PM TIMING DISTRIBUTION")
    print("  Tests: does agent converge to ABR t*?")
    print("="*60)

    pm_events = collect_pm_events(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        n_episodes=args.episodes,
        stoch_level=args.stoch,
    )

    # Summary statistics
    print("\n--- PM TIMING STATISTICS ---")
    print(f"{'Machine':<20} {'n_PM':>6} {'Mean age (h)':>14} {'Mean H%':>10} "
          f"{'ABR t* (h)':>12} {'% in ABR range':>16}")
    print("-" * 82)

    machine_names = list(ABR_T_STAR.keys())
    all_ages_h, all_healths = [], []
    report_stats = {}

    for m_id, events in pm_events.items():
        name = machine_names[m_id] if m_id < len(machine_names) else f"M{m_id}"
        abr  = ABR_T_STAR.get(name, {})
        t_star_h = abr.get("t_star_sh", 55) * 8

        if not events:
            print(f"  {name:<18}: NO PM events in {args.episodes} episodes!")
            continue

        ages_h  = [e["eff_age_h"] for e in events]
        healths = [e["health"] for e in events]
        all_ages_h.extend(ages_h)
        all_healths.extend(healths)

        mean_age = np.mean(ages_h)
        mean_h   = np.mean(healths)
        # % within ±20% of t*
        in_range = sum(1 for a in ages_h if t_star_h * 0.8 <= a <= t_star_h * 1.2)
        pct_in   = in_range / len(ages_h) * 100

        print(f"  {name:<18}: {len(events):>6}  {mean_age:>12.0f}h  {mean_h:>8.1f}%  "
              f"{t_star_h:>10.0f}h  {pct_in:>14.0f}%")

        report_stats[name] = {
            "n_PM": len(events),
            "mean_age_h": round(mean_age, 1),
            "std_age_h": round(float(np.std(ages_h)), 1),
            "mean_health": round(mean_h, 1),
            "abr_t_star_h": t_star_h,
            "pct_within_20pct_of_tstar": round(pct_in, 1),
        }

    # Fleet summary
    if all_ages_h:
        t_min = min(v["t_star_sh"] for v in ABR_T_STAR.values()) * 8
        t_max = max(v["t_star_sh"] for v in ABR_T_STAR.values()) * 8
        fleet_in = sum(1 for a in all_ages_h if t_min <= a <= t_max)
        fleet_pct = fleet_in / len(all_ages_h) * 100

        print(f"\n  Fleet: n={len(all_ages_h)} PM events")
        print(f"  Mean age at PM: {np.mean(all_ages_h):.0f}h = {np.mean(all_ages_h)/8:.1f} shifts")
        print(f"  Mean health at PM: {np.mean(all_healths):.1f}%")
        print(f"  ABR range: [{t_min:.0f}–{t_max:.0f}h]")
        print(f"  % PM within ABR range: {fleet_pct:.0f}%")

        if fleet_pct > 50:
            print("  ✓ Agent converged near ABR optimal threshold!")
        else:
            print("  ✗ Agent PM timing does not align with ABR threshold")

    # Plots
    analyse_and_plot(pm_events, args.outdir)

    # Save
    with open(os.path.join(args.outdir, "a9_pm_events.json"), "w") as f:
        json.dump({k: v for k, v in pm_events.items()}, f, indent=2)

    with open(os.path.join(args.outdir, "a9_report.json"), "w") as f:
        json.dump({
            "ablation": "A9_pm_timing",
            "episodes": args.episodes,
            "per_machine": report_stats,
            "fleet": {
                "n_total_PM": len(all_ages_h),
                "mean_age_h": round(float(np.mean(all_ages_h)), 1) if all_ages_h else 0,
                "mean_health": round(float(np.mean(all_healths)), 1) if all_healths else 0,
            }
        }, f, indent=2)

    print(f"\n  Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
