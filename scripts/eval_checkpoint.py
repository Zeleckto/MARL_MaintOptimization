"""
scripts/eval_checkpoint.py
===========================
Evaluate a trained MARL checkpoint against baselines.

Usage:
    python scripts/eval_checkpoint.py --checkpoint outputs/checkpoints_archive/phase1_step_0225k.pt
    python scripts/eval_checkpoint.py --checkpoint outputs/checkpoints/latest.pt --episodes 50
"""
import argparse, os, sys, yaml
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from environments.mfg_env import ManufacturingEnv, AGENT_PDM
from environments.transitions.degradation import MachineStatus
from environments.transitions.job_dynamics import OpStatus
from benchmarks.baselines import get_all_baselines


def eval_marl(config, checkpoint_path, n_episodes=30, device="cpu"):
    from training.mappo_trainer import MAPPOTrainer
    from utils.checkpoint import load_checkpoint

    trainer = MAPPOTrainer(config)
    meta = load_checkpoint(
        path=checkpoint_path,
        actor1=trainer.agent1.policy,
        actor2=trainer.agent2.tgin,
        critic=trainer.critic,
        device=device,
        action_scorer=trainer.agent2.action_scorer,
    )
    print(f"  Loaded: ep={meta['episode']}, step={meta['global_step']}")

    agent1 = trainer.agent1
    agent2 = trainer.agent2
    env = ManufacturingEnv(config)

    results = []
    for seed in range(42, 42 + n_episodes):
        env.reset(seed=seed)
        for step in range(env.t_max):
            # Agent 1
            obs1 = env._build_agent1_obs() if hasattr(env, '_build_agent1_obs') else env.observations[AGENT_PDM]
            a1, _, _ = agent1.act(
                obs_np=obs1,
                machine_states=env.machine_states,
                machine_busy=env.machine_busy,
                resource_state=env.resource_state,
                rho_PM=env.rho_PM, rho_CM=env.rho_CM,
            )
            env._step_agent1(a1)

            # Agent 2
            obs2 = env._build_agent2_obs()
            valid_pairs = env._valid_pairs
            if valid_pairs:
                with torch.no_grad():
                    sem, idx, lp, ent = agent2.act(obs2, valid_pairs)
                if idx < len(valid_pairs):
                    env._step_agent2(idx)
                else:
                    env._step_agent2(None)
            else:
                env._step_agent2(None)

            env._resolve_physics()
            env._compute_rewards()

        cj = [j for j in env.jobs if j.completion_time is not None]
        ot = sum(1 for j in cj if j.completion_time <= j.due_date)
        tard = sum(j.weight * max(0, j.completion_time - j.due_date) for j in cj)
        results.append({
            'jobs': env._episode_completions, 'fail': env._episode_failures,
            'pm': env._episode_pm, 'cm': env._episode_cm,
            'on_time': ot, 'tard': tard,
            'health': np.mean([s.health for s in env.machine_states]),
            'svc_lvl': ot / max(len(cj), 1),
            'r1': env._cumulative_rewards.get(AGENT_PDM, 0),
            'r2': env._cumulative_rewards.get('jobshop_agent', 0),
        })
    return results


def eval_baseline(config, policy, n_episodes=30):
    env = ManufacturingEnv(config)
    env.bypass_health_gate = True
    results = []
    for seed in range(42, 42 + n_episodes):
        env.reset(seed=seed)
        policy.reset()
        for step in range(env.t_max):
            a1 = policy.agent1_action(env)
            env._step_agent1(a1)
            a2 = policy.agent2_action(env)
            if env._valid_pairs and isinstance(a2, (int, np.integer)) and a2 < len(env._valid_pairs):
                env._step_agent2(a2)
            else:
                env._step_agent2(None)
            env._resolve_physics()
            env._compute_rewards()
        cj = [j for j in env.jobs if j.completion_time is not None]
        ot = sum(1 for j in cj if j.completion_time <= j.due_date)
        tard = sum(j.weight * max(0, j.completion_time - j.due_date) for j in cj)
        results.append({
            'jobs': env._episode_completions, 'fail': env._episode_failures,
            'pm': env._episode_pm, 'cm': env._episode_cm,
            'on_time': ot, 'tard': tard,
            'health': np.mean([s.health for s in env.machine_states]),
            'svc_lvl': ot / max(len(cj), 1),
        })
    return results


def print_results(name, results):
    a = lambda k: np.mean([r[k] for r in results])
    s = lambda k: np.std([r[k] for r in results])
    print(f"{name:<25} jobs={a('jobs'):>5.1f}±{s('jobs'):>3.1f}  fail={a('fail'):>4.1f}±{s('fail'):>3.1f}  "
          f"PM={a('pm'):>5.1f}  on_time={a('on_time'):>5.1f}  svc={a('svc_lvl'):>5.1%}  "
          f"tard={a('tard'):>6.0f}  health={a('health'):>5.1f}%")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--config", default="configs/base.yaml")
    pa.add_argument("--episodes", type=int, default=30)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = pa.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["stochasticity_level"] = 1

    print(f"\n{'='*90}")
    print(f"  MARL vs BASELINES COMPARISON ({args.episodes} episodes each)")
    print(f"{'='*90}\n")

    # Baselines
    baselines = get_all_baselines()
    for pol in baselines:
        res = eval_baseline(config, pol, args.episodes)
        print_results(pol.name, res)

    print(f"{'-'*90}")

    # MARL
    print(f"Loading MARL from: {args.checkpoint}")
    marl_res = eval_marl(config, args.checkpoint, args.episodes, args.device)
    print_results("MARL (trained)", marl_res)

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
