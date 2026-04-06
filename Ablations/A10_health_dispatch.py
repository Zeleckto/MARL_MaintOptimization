"""
A10_health_dispatch.py  —  Ablation A10: Health-Conditioned Assignment
======================================================================
Tests whether Agent 2 has learned health-aware routing:
"Does Agent 2 preferentially assign operations to healthier machines?"

Method:
  At every assignment step, log:
  - Health of the machine chosen
  - Health of the best available alternative (healthiest eligible machine)
  - Health gap: chosen_health - best_alternative_health

  If Agent 2 is health-aware: mean health gap ≥ 0 (never picks unhealthier
  when a healthier alternative exists).

  Compare vs Reactive+FCFS baseline (no health awareness).

Expected:
  MARL Agent 2 mean health gap > 0 (prefers healthier machines)
  Reactive baseline mean health gap ≈ 0 (random w.r.t. health)

Usage:
  python ablations/A10_health_dispatch.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 50 \\
      --outdir results/ablation_a10/
"""

import argparse, os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from ablations.ablation_utils import load_config


def collect_assignment_stats(
    env_class, cfg: dict, agent1, agent2,
    n_episodes: int, stoch_level: int,
    label: str, seed_offset: int = 42,
) -> dict:
    """Collect health stats at each assignment decision."""
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from environments.transitions.degradation import MachineStatus

    cfg_eval = {**cfg, "stochasticity_level": stoch_level}
    env = ManufacturingEnv(cfg_eval)

    chosen_healths = []
    best_alt_healths = []
    health_gaps = []
    assigned_to_unhealthiest = 0
    total_assignments = 0

    for ep in range(n_episodes):
        env.reset(seed=seed_offset + ep * 7)
        done, steps = False, 0

        while not done and steps < 300:
            # Agent 1 action
            if agent1 is not None:
                obs1 = env._build_agent1_obs()
                action1, _, _ = agent1.act(
                    obs_np=obs1, machine_states=env.machine_states,
                    machine_busy=env.machine_busy,
                    resource_state=env.resource_state,
                    rho_PM=env.rho_PM, rho_CM=env.rho_CM,
                )
            else:
                from benchmarks.baselines import ReactiveBaseline
                b = ReactiveBaseline(cfg_eval)
                action1 = b.agent1_action(env)
            env._step_agent1(action1)

            # Before Agent 2 acts: record available machines
            obs2, valid_pairs = env._build_agent2_obs()
            if valid_pairs:
                # Available machine healths for this decision
                avail_machine_ids = list(set(m_id for (_, _, m_id) in valid_pairs))
                avail_healths = {
                    m_id: env.machine_states[m_id].health
                    for m_id in avail_machine_ids
                    if env.machine_states[m_id].status == MachineStatus.OP
                }

                if len(avail_healths) > 1:
                    # Agent 2 selects
                    if agent2 is not None:
                        _, idx, _, _ = agent2.act(obs2, valid_pairs)
                    else:
                        idx = 0  # FCFS: pick first valid pair
                    chosen_pair = valid_pairs[idx] if idx < len(valid_pairs) else valid_pairs[0]
                    chosen_m_id = chosen_pair[2]

                    if chosen_m_id in avail_healths:
                        chosen_h = avail_healths[chosen_m_id]
                        best_h   = max(avail_healths.values())
                        gap      = chosen_h - best_h

                        chosen_healths.append(chosen_h)
                        best_alt_healths.append(best_h)
                        health_gaps.append(gap)
                        total_assignments += 1
                        if chosen_h == min(avail_healths.values()) and len(avail_healths) > 1:
                            assigned_to_unhealthiest += 1

                    env._step_agent2(idx)
                else:
                    env._step_agent2(0)
            else:
                env._step_agent2(len(valid_pairs))

            env._resolve_physics()
            env._compute_rewards()
            done = (env.terminations.get(AGENT_PDM, False) or
                    env.truncations.get(AGENT_PDM, False))
            steps += 1

        sys.stdout.write(f"\r    {label}: Episode {ep+1}/{n_episodes}")
        sys.stdout.flush()
    print()

    return {
        "label":                    label,
        "n_assignments":            total_assignments,
        "mean_chosen_health":       float(np.mean(chosen_healths)) if chosen_healths else 0,
        "mean_best_alt_health":     float(np.mean(best_alt_healths)) if best_alt_healths else 0,
        "mean_health_gap":          float(np.mean(health_gaps)) if health_gaps else 0,
        "std_health_gap":           float(np.std(health_gaps)) if health_gaps else 0,
        "pct_chose_unhealthiest":   assigned_to_unhealthiest / max(total_assignments, 1) * 100,
        "chosen_healths":           chosen_healths[:200],  # sample for histogram
        "health_gaps":              health_gaps[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="A10: Health-conditioned dispatch")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a10/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A10: HEALTH-CONDITIONED ASSIGNMENT")
    print("  Tests: does Agent 2 prefer healthier machines?")
    print("="*60)

    from environments.mfg_env import ManufacturingEnv
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent
    from models.critic import CentralizedCritic
    from utils.checkpoint import load_checkpoint
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_eval = {**cfg, "stochasticity_level": args.stoch}
    env_temp = ManufacturingEnv(cfg_eval)
    env_temp.reset(seed=0)
    obs_dim = len(env_temp._build_agent1_obs())

    agent1 = PDMAgent(cfg_eval, device=device, obs_dim=obs_dim)
    agent2 = JobShopAgent(cfg_eval, device=device)
    critic = CentralizedCritic(cfg_eval)
    load_checkpoint(args.checkpoint, agent1, agent2, critic, device=device)
    agent1.eval(); agent2.eval()

    print("\nCollecting MARL assignment statistics...")
    marl_stats = collect_assignment_stats(
        ManufacturingEnv, cfg, agent1, agent2,
        n_episodes=args.episodes, stoch_level=args.stoch,
        label="MARL", seed_offset=42,
    )

    print("\nCollecting Reactive+FCFS assignment statistics (baseline)...")
    reactive_stats = collect_assignment_stats(
        ManufacturingEnv, cfg, None, None,  # None = use baseline logic
        n_episodes=args.episodes, stoch_level=args.stoch,
        label="Reactive+FCFS", seed_offset=42,
    )

    # ── Print results ────────────────────────────────────────────────────────
    print("\n--- HEALTH-CONDITIONED DISPATCH ANALYSIS ---")
    print(f"\n{'Metric':<35} {'MARL':>12} {'Reactive+FCFS':>15}")
    print("-" * 64)
    metrics = [
        ("Assignments with choice", "n_assignments"),
        ("Mean chosen health (%)", "mean_chosen_health"),
        ("Mean best available health (%)", "mean_best_alt_health"),
        ("Mean health gap (chosen - best)", "mean_health_gap"),
        ("Std health gap", "std_health_gap"),
        ("% chose unhealthiest machine", "pct_chose_unhealthiest"),
    ]
    for label, key in metrics:
        m_val = marl_stats.get(key, 0)
        r_val = reactive_stats.get(key, 0)
        print(f"  {label:<33}: {m_val:>12.2f}  {r_val:>14.2f}")

    # Key finding
    gap_m = marl_stats["mean_health_gap"]
    gap_r = reactive_stats["mean_health_gap"]
    print(f"\n  Health gap interpretation:")
    print(f"  MARL gap = {gap_m:.2f}%  (>0 = chose healthier than best alt → expected, impossible by definition)")
    print(f"  Better metric: % assigned to ABOVE-median health machine:")

    # Compute % chose above median
    gaps_marl = np.array(marl_stats.get("health_gaps", []))
    print(f"  MARL pct gap >= 0: {(gaps_marl >= 0).mean()*100:.1f}%  "
          f"(100% means always picked the healthiest eligible machine)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("A10: Health-Conditioned Assignment Analysis", fontsize=12)

        # Health gap distribution
        ax1 = axes[0]
        if marl_stats.get("health_gaps"):
            ax1.hist(marl_stats["health_gaps"], bins=30, alpha=0.7,
                     color="#2196F3", label="MARL", edgecolor="white")
        ax1.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax1.set_xlabel("Health gap: chosen - best available (%)")
        ax1.set_ylabel("Count")
        ax1.set_title("Health Gap at Assignment\n(0 = always picks best)")
        ax1.legend()

        # Chosen health distribution
        ax2 = axes[1]
        if marl_stats.get("chosen_healths"):
            ax2.hist(marl_stats["chosen_healths"], bins=30, alpha=0.7,
                     color="#2196F3", label="MARL", edgecolor="white")
        ax2.set_xlabel("Health of assigned machine (%)")
        ax2.set_ylabel("Count")
        ax2.set_title("Machine Health at Assignment")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "a10_health_dispatch.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {args.outdir}/a10_health_dispatch.png")
    except ImportError:
        print("  matplotlib not available — skipping plot")

    # Save
    report = {
        "ablation": "A10_health_dispatch",
        "marl": {k: v for k, v in marl_stats.items()
                 if not isinstance(v, list)},
        "reactive": {k: v for k, v in reactive_stats.items()
                     if not isinstance(v, list)},
        "conclusion": (
            "Agent 2 shows health-aware routing"
            if marl_stats["mean_chosen_health"] > reactive_stats["mean_chosen_health"] + 2
            else "No significant health-aware routing detected"
        )
    }
    with open(os.path.join(args.outdir, "a10_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
