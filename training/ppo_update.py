from __future__ import annotations
"""
training/ppo_update.py
=======================
PPO update — Phase 1 version.

Only Agent 1 actor is updated via proper PPO (has gradient path: recomputed
log probs through MLP network).

Agent 2 and critic updates require storing global state in the rollout buffer
to reconstruct V(s) and recompute log probs. This is the Phase 2 TODO.
For Phase 1: Agent 1 learns, environment physics are validated, training
pipeline confirmed working end-to-end.
"""

from typing import Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def ppo_update(
    agent1, agent2, critic,
    buffer1, buffer2,
    optim_actor1, optim_actor2, optim_critic,
    config: dict,
) -> Dict[str, float]:

    if not TORCH_AVAILABLE:
        return {}

    mappo      = config.get("mappo", {})
    clip_eps   = mappo.get("clip_eps", 0.2)
    entropy_c  = mappo.get("entropy_coef", 0.01)
    max_grad   = mappo.get("max_grad_norm", 0.5)
    ppo_epochs = mappo.get("ppo_epochs", 10)
    mb_size    = mappo.get("minibatch_size", 64)

    dev1 = next(agent1.policy.parameters()).device if agent1.policy else torch.device("cpu")

    metrics = {
        "actor1_loss": 0.0, "actor2_loss": 0.0,
        "critic_loss": 0.0, "entropy1":    0.0,
        "entropy2":    0.0, "n_updates":   0,
    }

    for epoch in range(ppo_epochs):
        for mb1, mb2 in zip(
            buffer1.get_minibatches(mb_size),
            buffer2.get_minibatches(mb_size),
        ):
            batch = len(mb1["obs"])

            # ── Agent 1 actor update (proper PPO, gradient path exists) ────
            if agent1.policy is not None:
                obs1_t  = torch.tensor(
                    np.stack(mb1["obs"]), dtype=torch.float32
                ).to(dev1)
                adv1    = torch.tensor(
                    mb1["advantages"], dtype=torch.float32
                ).to(dev1)
                old_lp1 = torch.tensor(
                    mb1["log_probs"], dtype=torch.float32
                ).to(dev1)

                # Normalise advantages
                adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)

                # Recompute log probs through network (creates gradient path)
                maint_dist, reorder_dist = agent1.policy.forward(
                    obs1_t, maint_mask=None, reorder_mask=None
                )

                maint_acts = torch.tensor(
                    np.array([a["maintenance"] for a in mb1["actions"]]),
                    dtype=torch.long
                ).to(dev1)

                reorder_acts = torch.tensor(
                    np.array([a["reorder"] for a in mb1["actions"]]),
                    dtype=torch.long
                ).clamp(0, agent1.policy.q_max).to(dev1)

                n_m = agent1.n_machines
                n_c = agent1.n_consumable

                lp_maint = maint_dist.log_prob(
                    maint_acts.view(-1)
                ).view(batch, n_m).sum(-1)

                lp_reorder = reorder_dist.log_prob(
                    reorder_acts.view(-1)
                ).view(batch, n_c).sum(-1)

                new_lp1 = lp_maint + lp_reorder  # has gradient ✓

                # PPO clipped objective
                ratio   = torch.exp(new_lp1 - old_lp1)
                surr1   = ratio * adv1
                surr2   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv1
                a1_loss = -torch.min(surr1, surr2).mean()

                ent1    = maint_dist.entropy().mean() + reorder_dist.entropy().mean()
                a1_loss = a1_loss - entropy_c * ent1  # has gradient ✓

                optim_actor1.zero_grad()
                a1_loss.backward()  # safe — gradient path through MLP
                nn.utils.clip_grad_norm_(agent1.policy.parameters(), max_grad)
                optim_actor1.step()

                metrics["actor1_loss"] += a1_loss.item()
                metrics["entropy1"]    += ent1.item()

            # ── Agent 2 and Critic: skipped in Phase 1 ─────────────────────
            # Reason: no gradient path without storing/reconstructing global state.
            # Phase 2 TODO: store obs2 dicts and graph observations in buffer,
            # rebuild graphs in minibatch to get differentiable new_logp2 and V(s).
            # For Phase 1: Agent 1 learning validates the full training pipeline.

            metrics["n_updates"] += 1

    n = max(metrics["n_updates"], 1)
    for k in ["actor1_loss", "actor2_loss", "critic_loss", "entropy1", "entropy2"]:
        metrics[k] /= n

    return metrics


def build_optimizers(agent1, agent2, critic, config: dict):
    if not TORCH_AVAILABLE:
        return None, None, None

    import torch.optim as optim
    mappo = config.get("mappo", {})

    o1 = optim.Adam(agent1.parameters(), lr=mappo.get("lr_actor1", 1e-4))
    o2 = optim.Adam(agent2.parameters(), lr=mappo.get("lr_actor2", 3e-4))
    oc = optim.Adam(critic.parameters(), lr=mappo.get("lr_critic",  1e-3))

    return o1, o2, oc