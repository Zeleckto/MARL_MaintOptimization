"""
scripts/render_episode.py
==========================
Runs one episode with pygame visualisation.
Use this to visually verify environment logic is correct.

Usage:
    python scripts/render_episode.py --config configs/phase1.yaml
    python scripts/render_episode.py --config configs/phase1.yaml --checkpoint checkpoints/latest.pt
"""

import argparse
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    base_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "base.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(args.config) as f:
        override = yaml.safe_load(f)
    if override:
        config.update(override)

    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent

    env    = ManufacturingEnv(config, render_mode="human")
    agent1 = PDMAgent(config, device="cpu")
    agent2 = JobShopAgent(config, device="cpu")

    obs1, _ = env.reset(seed=args.seed)

    done = False
    step = 0
    while not done:
        obs2 = env.observe(AGENT_JOBSHOP)

        action1, _, _ = agent1.act(
            obs_np=obs1,
            machine_states=env.machine_states,
            machine_busy=env.machine_busy,
            resource_state=env.resource_state,
            rho_PM=env.rho_PM,
            rho_CM=env.rho_CM,
        )
        env.step(action1)
        obs2 = env.observe(AGENT_JOBSHOP)

        _, action2_idx, _, _ = agent2.act(obs2, env._valid_pairs)
        env.step(action2_idx)

        obs1  = env.observe(AGENT_PDM)
        term  = env.terminations[AGENT_PDM]
        trunc = env.truncations[AGENT_PDM]
        done  = term or trunc
        step += 1

        env.render()
        print(f"Step {step:>4} | "
              f"r1={env.rewards[AGENT_PDM]:>7.3f} "
              f"r2={env.rewards[AGENT_JOBSHOP]:>7.3f} | "
              f"failures={env._episode_failures}")

    print(f"\nEpisode done in {step} steps.")
    env.close()


if __name__ == "__main__":
    main()
