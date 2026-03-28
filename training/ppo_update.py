from __future__ import annotations
"""
training/ppo_update.py
=======================
PPO updates for both agents + critic.

Agent 1 (MLP): full PPO with proper ratio. Always worked.

Agent 2 (TGIN): NOW FIXED with proper ratio.
    For each minibatch sample, call agent2.get_log_prob(obs2, action_idx)
    to get new_lp2 through the current TGIN weights (differentiable).
    Then compute ratio = exp(new_lp2 - old_lp2) and clip normally.
    
    We loop over samples individually (not batched) because graph sizes
    differ per step. This is slower but correct. With mb_size=32 and
    ppo_epochs=5 this adds ~2s per update on RTX 3060 — acceptable.

Critic: skipped (Phase 2 TODO — needs global state reconstruction).
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
    ppo_epochs = mappo.get("ppo_epochs", 5)     # reduced: agent2 loop is slow
    mb_size    = mappo.get("minibatch_size", 32) # reduced: each sample = 1 TGIN forward

    dev1 = next(agent1.policy.parameters()).device if agent1.policy else torch.device("cpu")
    dev2 = next(agent2.tgin.parameters()).device   if agent2.tgin   else torch.device("cpu")

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

            # ── Agent 1: full PPO (unchanged) ─────────────────────────────
            if agent1.policy is not None:
                obs1_t  = torch.tensor(
                    np.stack(mb1["obs"]), dtype=torch.float32
                ).to(dev1)
                adv1    = torch.tensor(mb1["advantages"], dtype=torch.float32).to(dev1)
                old_lp1 = torch.tensor(mb1["log_probs"],  dtype=torch.float32).to(dev1)
                adv1    = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)

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

                lp_maint   = maint_dist.log_prob(maint_acts.view(-1)).view(batch, n_m).sum(-1)
                lp_reorder = reorder_dist.log_prob(reorder_acts.view(-1)).view(batch, n_c).sum(-1)
                new_lp1    = lp_maint + lp_reorder

                ratio1  = torch.exp(new_lp1 - old_lp1)
                surr1   = ratio1 * adv1
                surr2   = torch.clamp(ratio1, 1-clip_eps, 1+clip_eps) * adv1
                a1_loss = -torch.min(surr1, surr2).mean()
                ent1    = maint_dist.entropy().mean() + reorder_dist.entropy().mean()
                a1_loss = a1_loss - entropy_c * ent1

                optim_actor1.zero_grad()
                a1_loss.backward()
                nn.utils.clip_grad_norm_(agent1.policy.parameters(), max_grad)
                optim_actor1.step()

                metrics["actor1_loss"] += a1_loss.item()
                metrics["entropy1"]    += ent1.item()

            # ── Agent 2: proper PPO via per-sample TGIN forward ───────────
            if agent2.tgin is not None:
                adv2    = torch.tensor(mb2["advantages"], dtype=torch.float32).to(dev2)
                old_lp2 = torch.tensor(mb2["log_probs"],  dtype=torch.float32).to(dev2)
                adv2    = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)

                # Loop over samples — each needs its own TGIN forward pass
                # because graph sizes differ (variable ops/jobs per step)
                new_lps  = []
                entropies = []

                for i in range(batch):
                    obs2_i     = mb2["obs"][i]         # stored graph obs dict
                    action_i   = int(mb2["actions"][i]) # stored action index

                    # get_log_prob runs TGIN forward — differentiable
                    lp_i, ent_i = agent2.get_log_prob(obs2_i, action_i)
                    new_lps.append(lp_i)
                    entropies.append(ent_i)

                new_lp2 = torch.stack(new_lps)        # [batch] — has gradient ✓
                ent2    = torch.stack(entropies).mean()

                # PPO clipped objective — same as Agent 1
                ratio2  = torch.exp(new_lp2 - old_lp2)
                surr1   = ratio2 * adv2
                surr2   = torch.clamp(ratio2, 1-clip_eps, 1+clip_eps) * adv2
                a2_loss = -torch.min(surr1, surr2).mean()
                a2_loss = a2_loss - entropy_c * ent2

                optim_actor2.zero_grad()
                a2_loss.backward()  # safe — gradient through TGIN ✓
                nn.utils.clip_grad_norm_(
                    list(agent2.tgin.parameters()) +
                    list(agent2.action_scorer.parameters()),
                    max_grad,
                )
                optim_actor2.step()

                metrics["actor2_loss"] += a2_loss.item()
                metrics["entropy2"]    += ent2.item()

            metrics["n_updates"] += 1

    n = max(metrics["n_updates"], 1)
    for k in ["actor1_loss","actor2_loss","critic_loss","entropy1","entropy2"]:
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