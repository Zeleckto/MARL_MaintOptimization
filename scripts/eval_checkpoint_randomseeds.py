"""
scripts/eval_checkpoint.py
===========================
Evaluate MARL checkpoint against baselines with configurable seeds and horizon.

Usage:
    python scripts/eval_checkpoint.py --checkpoint outputs/checkpoints/latest.pt
    python scripts/eval_checkpoint.py --checkpoint latest.pt --episodes 50 --seed-start 100
    python scripts/eval_checkpoint.py --checkpoint latest.pt --seed-sets 3   # 3 independent sets
    python scripts/eval_checkpoint.py --checkpoint latest.pt --outdir outputs/eval_results
"""
import argparse, os, sys, yaml, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from environments.mfg_env import ManufacturingEnv, AGENT_PDM
from environments.transitions.degradation import MachineStatus
from environments.transitions.job_dynamics import OpStatus
from benchmarks.baselines import get_all_baselines


def run_eval(env, agent1_fn, agent2_fn, seeds, use_torch_agent2=False):
    """Generic eval: takes action functions, returns results per seed."""
    results = []
    for seed in seeds:
        env.reset(seed=seed)
        for step in range(env.t_max):
            a1 = agent1_fn(env)
            env._step_agent1(a1)
            a2 = agent2_fn(env)
            if env._valid_pairs and a2 is not None and isinstance(a2, (int, np.integer)) and a2 < len(env._valid_pairs):
                env._step_agent2(a2)
            else:
                env._step_agent2(None)
            env._resolve_physics()
            env._compute_rewards()

        cj = [j for j in env.jobs if j.completion_time is not None]
        ot = sum(1 for j in cj if j.completion_time <= j.due_date)
        tard = sum(j.weight * max(0, j.completion_time - j.due_date) for j in cj)
        ms = max((j.completion_time for j in cj), default=env.t_max)
        ops_done = sum(1 for j in env.jobs for op in j.operations if op.status == OpStatus.DONE)

        results.append({
            'seed': seed,
            'jobs': env._episode_completions, 'fail': env._episode_failures,
            'pm': env._episode_pm, 'cm': env._episode_cm,
            'on_time': ot, 'tard': tard, 'makespan': ms,
            'health': np.mean([s.health for s in env.machine_states]),
            'svc_lvl': ot / max(len(cj), 1),
            'ops_done': ops_done,
            'r1': env._cumulative_rewards.get(AGENT_PDM, 0),
            'r2': env._cumulative_rewards.get('jobshop_agent', 0),
            'avail': sum(1 for s in env.machine_states if s.status == MachineStatus.OP) / 5,
        })
    return results


def eval_marl(config, checkpoint_path, seeds, device="cpu"):
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

    def a1_fn(env):
        obs1 = env._build_agent1_obs() if hasattr(env, '_build_agent1_obs') else env.observations.get(AGENT_PDM, np.zeros(85))
        a1, _, _ = agent1.act(obs_np=obs1, machine_states=env.machine_states,
                              machine_busy=env.machine_busy, resource_state=env.resource_state,
                              rho_PM=env.rho_PM, rho_CM=env.rho_CM)
        return a1

    def a2_fn(env):
        if not env._valid_pairs:
            return None
        obs2 = env._build_agent2_obs()
        with torch.no_grad():
            sem, idx, lp, ent = agent2.act(obs2, env._valid_pairs)
        return idx

    return run_eval(env, a1_fn, a2_fn, seeds)


def eval_baseline(config, policy, seeds):
    env = ManufacturingEnv(config)
    env.bypass_health_gate = True

    def a1_fn(env):
        return policy.agent1_action(env)

    def a2_fn(env):
        a2 = policy.agent2_action(env)
        return a2

    policy.reset()
    results = []
    for seed in seeds:
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
        ms = max((j.completion_time for j in cj), default=env.t_max)
        ops_done = sum(1 for j in env.jobs for op in j.operations if op.status == OpStatus.DONE)

        results.append({
            'seed': seed, 'jobs': env._episode_completions, 'fail': env._episode_failures,
            'pm': env._episode_pm, 'cm': env._episode_cm,
            'on_time': ot, 'tard': tard, 'makespan': ms,
            'health': np.mean([s.health for s in env.machine_states]),
            'svc_lvl': ot / max(len(cj), 1), 'ops_done': ops_done,
            'avail': sum(1 for s in env.machine_states if s.status == MachineStatus.OP) / 5,
        })
    return results


def print_results(name, results):
    a = lambda k: np.mean([r[k] for r in results])
    s = lambda k: np.std([r[k] for r in results])
    print(f"{name:<28} jobs={a('jobs'):>5.1f}±{s('jobs'):>3.1f}  fail={a('fail'):>4.1f}±{s('fail'):>3.1f}  "
          f"PM={a('pm'):>5.1f}  ontime={a('on_time'):>5.1f}  svc={a('svc_lvl'):>5.1%}  "
          f"tard={a('tard'):>6.0f}  health={a('health'):>5.1f}%  ms={a('makespan'):>5.1f}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--config", default="configs/base.yaml")
    pa.add_argument("--episodes", type=int, default=30)
    pa.add_argument("--seed-start", type=int, default=42)
    pa.add_argument("--seed-sets", type=int, default=1, help="Run N independent seed sets for robustness")
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--outdir", default="outputs/eval_results")
    args = pa.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["stochasticity_level"] = 1

    os.makedirs(args.outdir, exist_ok=True)

    for seed_set in range(args.seed_sets):
        start = args.seed_start + seed_set * args.episodes
        seeds = list(range(start, start + args.episodes))

        print(f"\n{'='*100}")
        print(f"  SEED SET {seed_set+1}/{args.seed_sets}: seeds {seeds[0]}-{seeds[-1]} ({args.episodes} episodes)")
        print(f"{'='*100}\n")

        all_results = {}

        # Baselines
        for pol in get_all_baselines():
            res = eval_baseline(config, pol, seeds)
            print_results(pol.name, res)
            all_results[pol.name] = res

        print(f"{'-'*100}")

        # MARL
        print(f"Loading MARL: {args.checkpoint}")
        marl_res = eval_marl(config, args.checkpoint, seeds, args.device)
        print_results("MARL (trained)", marl_res)
        all_results["MARL (trained)"] = marl_res

        # Save per-seed CSV
        csv_path = os.path.join(args.outdir, f"eval_seedset_{seed_set+1}.csv")
        with open(csv_path, "w", newline="") as f:
            fields = ["policy", "seed", "jobs", "fail", "pm", "cm", "on_time",
                       "tard", "makespan", "health", "svc_lvl", "ops_done", "avail"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for pname, results in all_results.items():
                for r in results:
                    r["policy"] = pname
                    writer.writerow(r)
        print(f"\nSaved: {csv_path}")

    # Summary across seed sets
    if args.seed_sets > 1:
        print(f"\n{'='*100}")
        print(f"  ROBUSTNESS: {args.seed_sets} seed sets × {args.episodes} episodes = {args.seed_sets * args.episodes} total")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
