"""
scripts/bc_warmstart.py
========================
Behavioral Cloning warm-start for Agent 2 (TGIN + ActionScorer).

Collects expert demonstrations using fewest-ops-left heuristic,
then trains Agent 2 via cross-entropy loss to imitate the expert.

Usage:
    python scripts/bc_warmstart.py                              # fresh BC
    python scripts/bc_warmstart.py --checkpoint outputs/checkpoints/latest.pt  # keep Agent1
    python scripts/bc_warmstart.py --expert spt                 # use SPT expert
    python scripts/bc_warmstart.py --episodes 100 --epochs 30   # more data/training

Output:
    outputs/checkpoints/bc_warmstart.pt   — checkpoint with BC-trained Agent 2
"""
import argparse, os, sys, yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments.mfg_env import ManufacturingEnv
from environments.transitions.degradation import MachineStatus
from environments.transitions.job_dynamics import OpStatus

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# EXPERT HEURISTICS
# ═══════════════════════════════════════════════════════════════════

def fewest_ops_left(env):
    """Prioritize jobs closest to completion."""
    if not env._valid_pairs: return None
    best = 0; br = 999
    for i, (j, o, m) in enumerate(env._valid_pairs):
        job = next((jj for jj in env.jobs if jj.job_id == j), None)
        if job:
            remaining = sum(1 for op in job.operations if op.status != OpStatus.DONE)
            if remaining < br:
                br = remaining; best = i
    return best


def spt(env):
    """Shortest Processing Time."""
    if not env._valid_pairs: return None
    best = 0; bt = 999
    for i, (j, o, m) in enumerate(env._valid_pairs):
        job = next((jj for jj in env.jobs if jj.job_id == j), None)
        if job:
            pt = job.operations[o].nominal_proc_times.get(m, 999) / 8
            if pt < bt: bt = pt; best = i
    return best


def fcfs(env):
    """First Come First Served — pick index 0."""
    return 0 if env._valid_pairs else None


EXPERTS = {"fewest_ops": fewest_ops_left, "spt": spt, "fcfs": fcfs}


# ═══════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_expert_data(config, expert_fn, n_episodes=50, seeds=None):
    """Run expert policy, collect (obs2, expert_action_idx) pairs."""
    env = ManufacturingEnv(config)
    if seeds is None:
        seeds = list(range(100, 100 + n_episodes))

    dataset = []  # list of (obs_dict, action_idx)
    total_jobs = 0

    for seed in seeds:
        env.reset(seed=seed)
        for step in range(env.t_max):
            # Agent 1: h<75 PM policy
            maint = np.array([
                1 if env.machine_states[i].status == MachineStatus.OP
                     and not env.machine_busy[i]
                     and env.machine_states[i].health < 75
                else 0 for i in range(5)
            ], dtype=int)
            inv = env.resource_state.consumable_inventory
            reorder = np.array([8.0 if inv[i] < 10 else 0.0 for i in range(len(inv))])
            env._step_agent1({"maintenance": maint, "reorder": reorder})

            # Get Agent 2 observation AFTER Agent 1 acted
            obs2 = env._build_agent2_obs()
            valid_pairs = env._valid_pairs

            # Expert action
            expert_idx = expert_fn(env)

            if valid_pairs and expert_idx is not None:
                # Store (obs, expert_action) for training
                dataset.append((obs2, expert_idx))
                env._step_agent2(expert_idx)
            else:
                env._step_agent2(None)

            env._resolve_physics()
            env._compute_rewards()

        total_jobs += env._episode_completions

    avg_jobs = total_jobs / n_episodes
    print(f"  Collected {len(dataset)} expert samples from {n_episodes} episodes")
    print(f"  Expert avg jobs: {avg_jobs:.1f}")
    return dataset


# ═══════════════════════════════════════════════════════════════════
# BC TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_bc(agent2, dataset, n_epochs=20, batch_size=64, lr=1e-3, device="cpu"):
    """Train Agent 2 via cross-entropy on expert demonstrations."""
    optimizer = torch.optim.Adam(
    list(agent2.action_scorer.fast_mlp.parameters()) + 
    [agent2.action_scorer.gate],
    lr=lr
)

    # Shuffle dataset
    rng = np.random.default_rng(42)
    indices = np.arange(len(dataset))

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_idx:
                obs, expert_action = dataset[idx]

                # Forward pass through TGIN + ActionScorer
                dist, vp, _ = agent2._forward(obs)
                n_actions = len(vp) + 1  # valid pairs + WAIT

                # Clamp expert action to valid range
                target = min(expert_action, n_actions - 1)
                target_t = torch.tensor(target, device=device)

                # Cross-entropy loss
                log_prob = dist.log_prob(target_t)
                batch_loss = batch_loss - log_prob

                # Track accuracy
                predicted = dist.probs.argmax().item()
                correct += (predicted == target)
                total += 1

            batch_loss = batch_loss / len(batch_idx)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent2.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        print(f"  Epoch {epoch+1:>3}/{n_epochs}: loss={avg_loss:.4f}  accuracy={accuracy:.1%}")

    return agent2


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(agent2, config, n_episodes=20, device="cpu"):
    """Run BC-trained Agent 2 with h<75 PM and measure performance."""
    env = ManufacturingEnv(config)
    results = []

    for seed in range(200, 200 + n_episodes):
        env.reset(seed=seed)
        for step in range(env.t_max):
            maint = np.array([
                1 if env.machine_states[i].status == MachineStatus.OP
                     and not env.machine_busy[i]
                     and env.machine_states[i].health < 75
                else 0 for i in range(5)
            ], dtype=int)
            inv = env.resource_state.consumable_inventory
            reorder = np.array([8.0 if inv[i] < 10 else 0.0 for i in range(len(inv))])
            env._step_agent1({"maintenance": maint, "reorder": reorder})

            obs2 = env._build_agent2_obs()
            valid_pairs = env._valid_pairs

            if valid_pairs:
                with torch.no_grad():
                    dist, vp, _ = agent2._forward(obs2)
                    action_idx = dist.probs.argmax().item()  # greedy
                    action_idx = min(action_idx, len(valid_pairs) - 1)
                env._step_agent2(action_idx)
            else:
                env._step_agent2(None)

            env._resolve_physics()
            env._compute_rewards()

        results.append({
            "jobs": env._episode_completions,
            "fail": env._episode_failures,
            "r2": env._cumulative_rewards["jobshop_agent"],
        })

    avg_j = np.mean([r["jobs"] for r in results])
    avg_f = np.mean([r["fail"] for r in results])
    avg_r2 = np.mean([r["r2"] for r in results])
    print(f"\n  BC Agent 2 evaluation ({n_episodes} episodes):")
    print(f"    Jobs: {avg_j:.1f}  Failures: {avg_f:.1f}  r2: {avg_r2:+.0f}")
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="configs/base.yaml")
    pa.add_argument("--checkpoint", default=None, help="Load Agent1+Critic from this checkpoint")
    pa.add_argument("--expert", default="fewest_ops", choices=list(EXPERTS.keys()))
    pa.add_argument("--episodes", type=int, default=50, help="Expert episodes to collect")
    pa.add_argument("--epochs", type=int, default=20, help="BC training epochs")
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--outdir", default="outputs/checkpoints")
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = pa.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["stochasticity_level"] = 1

    print(f"\n{'='*60}")
    print(f"  BC WARM-START FOR AGENT 2")
    print(f"  Expert: {args.expert}")
    print(f"  Episodes: {args.episodes}, Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # ── Build models ──────────────────────────────────────────
    from training.mappo_trainer import MAPPOTrainer
    trainer = MAPPOTrainer(config)
    agent2 = trainer.agent2

    # ── Load existing checkpoint (keeps Agent 1 + Critic) ─────
    if args.checkpoint:
        print(f"Loading Agent1 + Critic from: {args.checkpoint}")
        from utils.checkpoint import load_checkpoint
        meta = load_checkpoint(
            path=args.checkpoint,
            actor1=trainer.agent1.policy,
            actor2=trainer.agent2.tgin,
            critic=trainer.critic,
            device=args.device,
            action_scorer=trainer.agent2.action_scorer,
        )
        print(f"  Loaded ep={meta['episode']}, step={meta['global_step']}")
        print(f"  Agent 1 weights: KEPT from checkpoint")
        print(f"  Agent 2 weights: will be OVERWRITTEN by BC")
        print()

    # ── Collect expert data ───────────────────────────────────
    print(f"Collecting {args.expert} expert data...")
    expert_fn = EXPERTS[args.expert]
    dataset = collect_expert_data(config, expert_fn, n_episodes=args.episodes)

    # ── Train Agent 2 via BC ──────────────────────────────────
    print(f"\nTraining Agent 2 (TGIN + ActionScorer) via BC...")
    agent2 = train_bc(agent2, dataset, n_epochs=args.epochs, lr=args.lr, device=args.device)

    # ── Evaluate ──────────────────────────────────────────────
    evaluate(agent2, config, n_episodes=20, device=args.device)

    # ── Save checkpoint ───────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    from utils.checkpoint import save_checkpoint
    save_checkpoint(
        checkpoint_dir=args.outdir,
        episode=0,
        global_step=0,
        actor1=trainer.agent1.policy,
        actor2=trainer.agent2.tgin,
        critic=trainer.critic,
        optim_actor1=trainer.optim1,
        optim_actor2=trainer.optim2,
        optim_critic=trainer.optim_critic,
        config=config,
        tag="bc_warmstart",
        action_scorer=trainer.agent2.action_scorer,
    )

    print(f"\n{'='*60}")
    print(f"  DONE. Resume RL training with:")
    print(f"  python scripts/train.py --config configs/phase1.yaml \\")
    print(f"    --timesteps 500000 --resume {args.outdir}/bc_warmstart.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()