"""
benchmarks/evaluate.py
=======================
Runs a trained Tier 3 policy on benchmark instances and reports metrics.
Tier 1 comparison will be added once CP-SAT formulation is complete.

Usage:
    python scripts/evaluate.py --config configs/benchmark_instances/small_3m_5j.yaml
                                --checkpoint checkpoints/latest.pt
"""

import numpy as np
from typing import Optional
import yaml

from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
from agents.pdm_agent import PDMAgent
from agents.jobshop_agent import JobShopAgent
from benchmarks.metrics import summarise_episode


def evaluate_policy(
    config:    dict,
    ckpt_path: Optional[str] = None,
    n_episodes: int = 10,
    render:    bool = False,
    seed:      int  = 999,
) -> dict:
    """
    Evaluates a trained policy on the given config.

    Args:
        config:     Full config dict (use benchmark instance yaml)
        ckpt_path:  Path to checkpoint .pt file (None = random policy)
        n_episodes: Number of evaluation episodes
        render:     True to show pygame window
        seed:       Evaluation seed (different from training seed)

    Returns:
        Dict of mean metrics across n_episodes
    """
    device = "cpu"   # evaluation on CPU for reproducibility

    # Use eval episode length
    eval_config = config.copy()
    eval_config["episode"] = dict(config.get("episode", {}))
    eval_config["episode"]["t_max_train"] = config.get("episode", {}).get(
        "t_max_eval", 500
    )

    env    = ManufacturingEnv(eval_config, render_mode="human" if render else None)
    agent1 = PDMAgent(eval_config, device=device)
    agent2 = JobShopAgent(eval_config, device=device)

    # Load checkpoint if provided
    if ckpt_path is not None:
        try:
            from utils.checkpoint import load_checkpoint
            from models.critic import CentralizedCritic
            critic = CentralizedCritic(eval_config)
            load_checkpoint(ckpt_path, agent1.policy, agent2.tgin, critic,
                            device=device)
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Using random policy.")

    all_metrics = []

    for ep in range(n_episodes):
        obs1, _ = env.reset(seed=seed + ep)
        obs2    = env.observe(AGENT_JOBSHOP)

        ep_n_failures = 0
        ep_n_PM = 0
        ep_n_CM = 0
        ep_ordering_cost = 0.0

        done = False
        while not done:
            # Agent 1
            action1, _, _ = agent1.act(
                obs_np=obs1,
                machine_states=env.machine_states,
                machine_busy=env.machine_busy,
                resource_state=env.resource_state,
                rho_PM=env.rho_PM,
                rho_CM=env.rho_CM,
            )
            ep_n_PM += sum(1 for a in action1["maintenance"] if a == 1)
            ep_n_CM += sum(1 for a in action1["maintenance"] if a == 2)

            env.step(action1)
            obs2 = env.observe(AGENT_JOBSHOP)

            # Agent 2
            _, action2_idx, _, _ = agent2.act(obs2, env._valid_pairs)
            env.step(action2_idx)

            obs1  = env.observe(AGENT_PDM)
            obs2  = env.observe(AGENT_JOBSHOP)
            term  = env.terminations[AGENT_PDM]
            trunc = env.truncations[AGENT_PDM]
            done  = term or trunc

            ep_n_failures    += len(env._newly_failed)
            ep_ordering_cost += env._last_ordering_cost

            if render:
                env.render()

        metrics = summarise_episode(
            jobs=env.jobs,
            machine_states=env.machine_states,
            n_failures=ep_n_failures,
            n_PM=ep_n_PM,
            n_CM=ep_n_CM,
            ordering_cost=ep_ordering_cost,
            episode_length=env.current_step,
            weights=config.get("reward", {}),
        )
        all_metrics.append(metrics)

        print(f"Episode {ep+1}/{n_episodes}: "
              f"cost={metrics['total_cost']:.1f}, "
              f"tard={metrics['weighted_tardiness']:.1f}, "
              f"failures={metrics['n_failures']}, "
              f"avail={metrics['system_availability']:.3f}")

    env.close()

    # Average across episodes
    keys = all_metrics[0].keys()
    mean_metrics = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in keys
    }

    print("\n── Mean metrics across episodes ──")
    for k, v in mean_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    return mean_metrics