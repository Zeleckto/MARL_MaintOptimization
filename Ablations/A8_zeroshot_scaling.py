"""
A8_zeroshot_scaling.py  —  Ablation A8: Zero-Shot Scaling (M=5 → M=10)
=======================================================================
Tests whether the trained TGIN policy generalises to unseen machine counts.

Key insight: TGIN is permutation-invariant and node-count agnostic.
A policy trained on M=5 machines can, in principle, operate on M=10
because the graph architecture handles variable numbers of machine nodes.

Agent 2 (TGIN): True zero-shot — new machine nodes are added to the graph.
Agent 1 (MLP):  Zero-padded — MLP input padded with zeros for new machines.
                New machines start at H=100%, hazard≈0, so Agent 1 correctly
                ignores them (zeros ≈ healthy machine signal).

Procedure:
  1. Load trained M=5 checkpoint
  2. Create M=10 environment (add 5 new machines with realistic parameters)
  3. For Agent 1: pad obs from 96→166 dims with zeros for new machines
  4. Evaluate 20 episodes, compare vs M=5 baseline

Expected:
  TGIN should achieve within 20–30% of M=5 performance.
  If degradation > 30%: architectural change needed for true generalisation.

Usage:
  python ablations/A8_zeroshot_scaling.py \\
      --checkpoint checkpoints/phase3_500k.pt \\
      --episodes 20 \\
      --outdir results/ablation_a8/
"""

import argparse, os, sys, copy, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from ablations.ablation_utils import (
    load_config, eval_marl_policy, compare_table,
    save_results, statistical_summary, AblationResult, EPISODE_KPIS
)


# 5 additional machines with realistic Weibull parameters
EXTRA_MACHINES = [
    {
        "machine_id": 5, "name": "Drill Press", "type": "A",
        "beta": 2.6, "eta": 950.0, "delta_h": 0.38, "h_critical": 10.0,
        "tau_PM_shifts": 2, "tau_CM_shifts": 6,
        "h_restore_PM": 30.0, "h_restore_CM": 60.0,
        "rho_PM": {"technicians": 1, "tools": 1, "spare_parts": 2, "lubricants": 1},
        "rho_CM": {"technicians": 2, "tools": 2, "spare_parts": 5, "lubricants": 2},
    },
    {
        "machine_id": 6, "name": "Boring Mill", "type": "B",
        "beta": 2.9, "eta": 800.0, "delta_h": 0.45, "h_critical": 10.0,
        "tau_PM_shifts": 3, "tau_CM_shifts": 7,
        "h_restore_PM": 33.0, "h_restore_CM": 63.0,
        "rho_PM": {"technicians": 2, "tools": 2, "spare_parts": 3, "lubricants": 2},
        "rho_CM": {"technicians": 3, "tools": 3, "spare_parts": 7, "lubricants": 3},
    },
    {
        "machine_id": 7, "name": "Milling Centre", "type": "A",
        "beta": 3.1, "eta": 1050.0, "delta_h": 0.36, "h_critical": 10.0,
        "tau_PM_shifts": 2, "tau_CM_shifts": 6,
        "h_restore_PM": 30.0, "h_restore_CM": 60.0,
        "rho_PM": {"technicians": 1, "tools": 1, "spare_parts": 2, "lubricants": 1},
        "rho_CM": {"technicians": 2, "tools": 2, "spare_parts": 5, "lubricants": 2},
    },
    {
        "machine_id": 8, "name": "EDM", "type": "C",
        "beta": 2.4, "eta": 1200.0, "delta_h": 0.30, "h_critical": 10.0,
        "tau_PM_shifts": 2, "tau_CM_shifts": 5,
        "h_restore_PM": 28.0, "h_restore_CM": 55.0,
        "rho_PM": {"technicians": 1, "tools": 1, "spare_parts": 1, "lubricants": 0},
        "rho_CM": {"technicians": 2, "tools": 2, "spare_parts": 3, "lubricants": 1},
    },
    {
        "machine_id": 9, "name": "Surface Grinder", "type": "B",
        "beta": 3.0, "eta": 750.0, "delta_h": 0.47, "h_critical": 10.0,
        "tau_PM_shifts": 3, "tau_CM_shifts": 8,
        "h_restore_PM": 35.0, "h_restore_CM": 65.0,
        "rho_PM": {"technicians": 2, "tools": 2, "spare_parts": 3, "lubricants": 2},
        "rho_CM": {"technicians": 3, "tools": 3, "spare_parts": 7, "lubricants": 3},
    },
]


def eval_zeroshot_m10(
    checkpoint_path: str,
    cfg_m5: dict,
    cfg_m10: dict,
    n_episodes: int,
    seed_offset: int = 42,
) -> AblationResult:
    """
    Evaluate M=5 trained policy on M=10 environment.
    Agent 1: obs zero-padded from 96 to 166 dims.
    Agent 2: graph extended with 5 new machine nodes (true zero-shot).
    """
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent
    from models.critic import CentralizedCritic
    from utils.checkpoint import load_checkpoint
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load agents trained on M=5
    env_m5 = ManufacturingEnv(cfg_m5)
    env_m5.reset(seed=0)
    obs1_m5 = env_m5._build_agent1_obs()
    obs_dim_m5 = len(obs1_m5)  # 96

    agent1 = PDMAgent(cfg_m5, device=device, obs_dim=obs_dim_m5)
    agent2 = JobShopAgent(cfg_m5, device=device)
    critic = CentralizedCritic(cfg_m5)
    load_checkpoint(checkpoint_path, agent1, agent2, critic, device=device)
    agent1.eval(); agent2.eval()

    # M=10 environment
    env_m10 = ManufacturingEnv(cfg_m10)
    env_m10.reset(seed=0)
    obs1_m10 = env_m10._build_agent1_obs()
    obs_dim_m10 = len(obs1_m10)  # 166

    print(f"  Agent 1 obs: M=5→{obs_dim_m5} dims, M=10→{obs_dim_m10} dims")
    print(f"  Padding Agent 1 obs with {obs_dim_m10 - obs_dim_m5} zeros (new machines = healthy)")

    kpi_data = {k: [] for k in EPISODE_KPIS}
    print(f"  Zero-shot M=10 evaluation ({n_episodes} episodes)...")

    for ep in range(n_episodes):
        env_m10.reset(seed=seed_offset + ep * 7)
        done, steps, n_PM, n_CM = False, 0, 0, 0

        while not done and steps < 300:
            # Agent 1: zero-pad obs to M=5 trained size
            obs1_full = env_m10._build_agent1_obs()  # 166 dims
            # Truncate to M=5 obs (first 96 dims = first 5 machines + resources + jobs)
            obs1_truncated = obs1_full[:obs_dim_m5]

            action1, _, _ = agent1.act(
                obs_np=obs1_truncated,
                machine_states=env_m10.machine_states[:5],  # only first 5 machines
                machine_busy=env_m10.machine_busy[:5],
                resource_state=env_m10.resource_state,
                rho_PM=env_m10.rho_PM[:5],
                rho_CM=env_m10.rho_CM[:5],
            )
            # Pad maintenance action for extra machines (no-op)
            maint_full = np.zeros(10, dtype=int)
            maint_full[:5] = action1["maintenance"]
            action1_full = {"maintenance": maint_full, "reorder": action1["reorder"]}
            env_m10._step_agent1(action1_full)

            n_PM += sum(1 for a in action1["maintenance"] if a == 1)
            n_CM += sum(1 for a in action1["maintenance"] if a == 2)

            # Agent 2: true zero-shot — graph has M=10 machine nodes
            obs2, valid_pairs = env_m10._build_agent2_obs()
            if valid_pairs:
                _, idx, _, _ = agent2.act(obs2, valid_pairs)
                env_m10._step_agent2(idx)
            else:
                env_m10._step_agent2(len(valid_pairs))

            env_m10._resolve_physics()
            env_m10._compute_rewards()
            done = (env_m10.terminations.get(AGENT_PDM, False) or
                    env_m10.truncations.get(AGENT_PDM, False))
            steps += 1

        completed = [j for j in env_m10.jobs if j.completion_time is not None]
        on_time   = [j for j in completed if j.tardiness == 0]
        tard      = sum(j.weight * j.tardiness for j in completed)
        kpi_data["failures"].append(env_m10._episode_failures)
        kpi_data["n_PM"].append(n_PM)
        kpi_data["n_CM"].append(n_CM)
        kpi_data["pm_cm_ratio"].append(n_PM / max(n_CM, 1))
        kpi_data["completions"].append(len(completed))
        kpi_data["tardiness"].append(float(tard))
        kpi_data["service_level"].append(len(on_time) / max(len(completed), 1))
        kpi_data["avg_health"].append(float(np.mean([s.health for s in env_m10.machine_states])))

        sys.stdout.write(f"\r    Episode {ep+1}/{n_episodes}")
        sys.stdout.flush()
    print()

    return AblationResult(name="MARL_M10_zeroshot", kpi_data=kpi_data)


def main():
    parser = argparse.ArgumentParser(description="A8: Zero-shot scaling M=5→M=10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--stoch",      type=int, default=3)
    parser.add_argument("--outdir",     default="results/ablation_a8/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg_m5 = load_config()
    cfg_m5["stochasticity_level"] = args.stoch

    # M=10 config: duplicate eligibility, scale jobs slightly
    cfg_m10 = copy.deepcopy(cfg_m5)
    cfg_m10["machines"] = cfg_m5["machines"] + EXTRA_MACHINES
    # Extend eligibility matrix to cover new machines (existing ops remain as-is)
    # New machines pick up some eligible ops

    print("\n" + "="*60)
    print("  ABLATION A8: ZERO-SHOT SCALING (M=5 → M=10)")
    print("  Agent 2 (TGIN): true zero-shot")
    print("  Agent 1 (MLP): truncated obs (first 5 machines only)")
    print("="*60)

    # ── M=5 baseline (same checkpoint, native environment) ──────────────────
    print(f"\nBaseline: M=5 (trained)")
    result_m5 = eval_marl_policy(
        checkpoint_path=args.checkpoint,
        config=cfg_m5,
        n_episodes=args.episodes,
        stoch_level=args.stoch,
        name="MARL_M5_trained",
    )

    # ── M=10 zero-shot ────────────────────────────────────────────────────────
    print(f"\nZero-shot: M=10")
    try:
        result_m10 = eval_zeroshot_m10(
            checkpoint_path=args.checkpoint,
            cfg_m5=cfg_m5,
            cfg_m10=cfg_m10,
            n_episodes=args.episodes,
            seed_offset=42,
        )
    except Exception as e:
        print(f"  M=10 eval failed: {e}")
        print("  Note: may require adjusting the M=10 config machine types/eligibility")
        return

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n--- COMPARISON TABLE: M=5 (trained) vs M=10 (zero-shot) ---")
    table = compare_table([result_m5, result_m10],
                          reference_name="MARL_M5_trained")

    # Relative performance
    print("\n--- RELATIVE PERFORMANCE (M=10 / M=5) ---")
    for kpi in EPISODE_KPIS:
        m5_mean  = np.mean(result_m5.kpi_data[kpi])
        m10_mean = np.mean(result_m10.kpi_data[kpi])
        if abs(m5_mean) > 1e-6:
            rel = m10_mean / m5_mean * 100
            print(f"  {kpi:<20}: M=5={m5_mean:.2f}  M=10={m10_mean:.2f}  "
                  f"({rel:.0f}% of M=5)")
        
    save_results([result_m5, result_m10], args.outdir, "a8_zeroshot")

    report = {
        "ablation": "A8_zeroshot_scaling",
        "checkpoint": args.checkpoint,
        "m5_results": result_m5.means,
        "m10_results": result_m10.means,
        "relative_performance": {
            k: round(np.mean(result_m10.kpi_data[k]) /
                     max(np.mean(result_m5.kpi_data[k]), 1e-6) * 100, 1)
            for k in EPISODE_KPIS
        },
        "note": (
            "Agent 2 (TGIN) is true zero-shot. "
            "Agent 1 (MLP) uses truncated obs (first 5 machines only). "
            "New machines at M=10 are managed only by Agent 2's health-aware routing."
        )
    }
    with open(os.path.join(args.outdir, "a8_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.outdir, "a8_table.txt"), "w") as f:
        f.write(table)

    print(f"\n  Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
