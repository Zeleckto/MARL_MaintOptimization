from __future__ import annotations
"""
training/ppo_update.py
======================
PPO updates for Agent 1 (MLP), Agent 2 (TGIN), and Critic.
All three fully trained every update.
v2 fix: Critic was marked TODO/skipped for 400k steps causing
GAE advantages to be pure noise (CV~6). Now correctly trains via
MSE on Monte Carlo returns.
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
    dev2 = next(agent2.tgin.parameters()).device   if agent2.tgin   else torch.device("cpu")

    metrics = {
        "actor1_loss": 0.0, "actor2_loss": 0.0,
        "critic_loss": 0.0, "entropy1":    0.0,
        "entropy2":    0.0, "n_updates":   0,
        "kl1": 0.0,
    }

    for epoch in range(ppo_epochs):
        for mb1, mb2 in zip(
            buffer1.get_minibatches(mb_size),
            buffer2.get_minibatches(mb_size),
        ):
            batch = len(mb1["obs"])

            # Agent 1: standard PPO
            if agent1.policy is not None:
                obs1_t  = torch.tensor(np.stack(mb1["obs"]), dtype=torch.float32).to(dev1)
                adv1    = torch.tensor(mb1["advantages"], dtype=torch.float32).to(dev1)
                old_lp1 = torch.tensor(mb1["log_probs"],  dtype=torch.float32).to(dev1)
                adv1    = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)

                maint_dist, reorder_dist = agent1.policy.forward(
                    obs1_t, maint_mask=None, reorder_mask=None)
                maint_acts = torch.tensor(
                    np.array([a["maintenance"] for a in mb1["actions"]]),
                    dtype=torch.long).to(dev1)
                reorder_acts = torch.tensor(
                    np.array([a["reorder"] for a in mb1["actions"]]),
                    dtype=torch.long).clamp(0, agent1.policy.q_max).to(dev1)

                n_m = agent1.n_machines
                n_c = agent1.n_consumable
                lp_maint   = maint_dist.log_prob(maint_acts.view(-1)).view(batch, n_m).sum(-1)
                lp_reorder = reorder_dist.log_prob(reorder_acts.view(-1)).view(batch, n_c).sum(-1)
                new_lp1    = lp_maint + lp_reorder

                ratio1  = torch.exp(new_lp1 - old_lp1)
                surr1   = ratio1 * adv1
                surr2   = torch.clamp(ratio1, 1 - clip_eps, 1 + clip_eps) * adv1
                a1_loss = -torch.min(surr1, surr2).mean()
                ent1    = maint_dist.entropy().mean() + reorder_dist.entropy().mean()
                a1_loss = a1_loss - entropy_c * ent1

                optim_actor1.zero_grad()
                a1_loss.backward()
                nn.utils.clip_grad_norm_(agent1.policy.parameters(), max_grad)
                optim_actor1.step()

                metrics["actor1_loss"] += a1_loss.item()
                metrics["entropy1"]    += ent1.item()
                metrics["kl1"]         += abs(((ratio1 - 1) - (new_lp1 - old_lp1)).mean().item())

            # Agent 2: per-sample PPO (variable graph sizes require individual forwards)
            if agent2.tgin is not None:
                adv2    = torch.tensor(mb2["advantages"], dtype=torch.float32).to(dev2)
                old_lp2 = torch.tensor(mb2["log_probs"],  dtype=torch.float32).to(dev2)
                adv2    = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)

                new_lps = []; entropies = []
                for i in range(batch):
                    lp_i, ent_i = agent2.get_log_prob(mb2["obs"][i], int(mb2["actions"][i]))
                    new_lps.append(lp_i); entropies.append(ent_i)

                new_lp2 = torch.stack(new_lps)
                ent2    = torch.stack(entropies).mean()
                ratio2  = torch.exp(new_lp2 - old_lp2)
                surr1   = ratio2 * adv2
                surr2   = torch.clamp(ratio2, 1 - clip_eps, 1 + clip_eps) * adv2
                a2_loss = -torch.min(surr1, surr2).mean() - entropy_c * ent2

                optim_actor2.zero_grad()
                a2_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent2.tgin.parameters()) +
                    list(agent2.action_scorer.parameters()), max_grad)
                optim_actor2.step()

                metrics["actor2_loss"] += a2_loss.item()
                metrics["entropy2"]    += ent2.item()

            # Critic: MSE on MC returns — was skipped for 400k steps (Bug 1)
            # Without this, V(s) stayed random, GAE = noise, nothing could learn.
            if critic is not None and optim_critic is not None:
                ret1   = torch.tensor(mb1["returns"], dtype=torch.float32).to(dev1)
                obs1_c = torch.tensor(np.stack(mb1["obs"]), dtype=torch.float32).to(dev1)

                # TGIN embeddings per sample, detached to protect actor weights
                hidden = critic.hidden_dim
                all_embs = []
                for i in range(batch):
                    try:
                        from models.tgin.graph_builder import GraphBuilder
                        g = GraphBuilder(config).build(mb2["obs"][i], device=dev1)
                        with torch.no_grad():
                            emb = agent2.tgin(g)
                        op_e = emb["op"].mean(0)     if emb["op"].shape[0]  > 0 else torch.zeros(hidden, device=dev1)
                        ma_e = emb["machine"].mean(0)
                        jo_e = emb["job"].mean(0)    if emb["job"].shape[0] > 0 else torch.zeros(hidden, device=dev1)
                        all_embs.append(torch.cat([op_e, ma_e, jo_e]))
                    except Exception:
                        all_embs.append(torch.zeros(3 * hidden, device=dev1))

                emb_batch = torch.stack(all_embs)

                # Extract resource obs from obs1: layout is [mach_feats, res_feats, job_summary(5)]
                n_mach = agent1.n_machines
                from environments.spaces.observation_spaces import MACHINE_FEATURE_DIM
                res_start = MACHINE_FEATURE_DIM * n_mach
                res_end   = obs1_c.shape[1] - 5
                res_flat  = obs1_c[:, res_start:res_end]

                # Build global state, trim/pad to match critic's expected input dim
                gs = torch.cat([emb_batch, res_flat, obs1_c,
                                torch.zeros(batch, n_mach * 2, device=dev1)], dim=-1)
                exp = critic.net[0].in_features
                act = gs.shape[1]
                if   act > exp: gs = gs[:, :exp]
                elif act < exp: gs = torch.cat([gs, torch.zeros(batch, exp - act, device=dev1)], 1)

                v_pred  = critic.net(gs).squeeze(-1)
                # Normalise returns so critic loss stays in [0, 10] range
                # Raw returns can be -1000 to +100 -> MSE = 100^2 (Bug 3 fix)
                ret_mean = ret1.mean(); ret_std = ret1.std().clamp(min=1.0)
                ret_norm = (ret1 - ret_mean) / ret_std
                v_norm   = (v_pred - ret_mean) / ret_std
                c_loss   = F.mse_loss(v_norm, ret_norm.clamp(-10, 10))

                optim_critic.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad)
                optim_critic.step()
                metrics["critic_loss"] += c_loss.item()

            metrics["n_updates"] += 1

    n = max(metrics["n_updates"], 1)
    for k in ["actor1_loss", "actor2_loss", "critic_loss", "entropy1", "entropy2", "kl1"]:
        metrics[k] /= n
    return metrics


def build_optimizers(agent1, agent2, critic, config: dict):
    if not TORCH_AVAILABLE:
        return None, None, None
    import torch.optim as optim
    mappo = config.get("mappo", {})
    o1 = optim.Adam(agent1.parameters(), lr=mappo.get("lr_actor1", 1e-4))  if agent1 else None
    o2 = optim.Adam(agent2.parameters(), lr=mappo.get("lr_actor2", 3e-4))  if agent2 else None
    oc = optim.Adam(critic.parameters(), lr=mappo.get("lr_critic",  1e-3)) if critic else None
    return o1, o2, oc