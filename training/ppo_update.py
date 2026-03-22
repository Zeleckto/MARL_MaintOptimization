"""
training/ppo_update.py
=======================
PPO loss computation and gradient updates for both actors + critic.

Implements:
    L_actor_i = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] - c_ent * H(pi)
    L_critic   = E[(R_t + gamma*V(s_{t+1}) - V(s_t))^2]
    L_total    = L_actor1 + L_actor2 + c_v * L_critic

Key notes:
    - Separate optimizers for theta1, theta2, phi
    - K epochs over minibatches per rollout
    - Advantage normalisation per agent independently
    - Gradient clipping (max_grad_norm=0.5)
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PPOUpdater:
    """
    Handles all PPO gradient updates for one training iteration.
    Called after rollout buffer is full (every rollout_steps environment steps).
    """

    def __init__(self, config: dict, actor1, actor2, critic, device: str = "cpu"):
        """
        Args:
            config:  Full config dict — reads mappo hyperparameters
            actor1:  MLPPolicy (Agent 1)
            actor2:  TGIN + ActionScorer (Agent 2) — pass as tuple or wrapper
            critic:  CentralizedCritic
            device:  'cuda' or 'cpu'
        """
        if not TORCH_AVAILABLE:
            return

        mappo = config.get("mappo", {})
        self.clip_eps      = mappo.get("clip_eps",        0.2)
        self.entropy_coef  = mappo.get("entropy_coef",    0.01)
        self.value_coef    = mappo.get("value_loss_coef", 0.5)
        self.max_grad_norm = mappo.get("max_grad_norm",   0.5)
        self.ppo_epochs    = mappo.get("ppo_epochs",      10)
        self.minibatch_sz  = mappo.get("minibatch_size",  64)

        self.actor1 = actor1
        self.actor2 = actor2
        self.critic = critic
        self.device = device

        # Separate optimizers per agent (different learning rates)
        self.optim_actor1 = torch.optim.Adam(
            actor1.parameters(), lr=mappo.get("lr_actor1", 1e-4)
        )
        self.optim_actor2 = torch.optim.Adam(
            actor2.parameters(), lr=mappo.get("lr_actor2", 3e-4)
        )
        self.optim_critic = torch.optim.Adam(
            critic.parameters(), lr=mappo.get("lr_critic", 1e-3)
        )


    def update(self, buffer) -> dict:
        """
        Runs K epochs of PPO updates over the rollout buffer.

        Args:
            buffer: RolloutBuffer with filled buffer1 (Agent 1) and buffer2 (Agent 2)

        Returns:
            Dict of mean losses for TensorBoard logging
        """
        if not TORCH_AVAILABLE:
            return {}

        import torch

        losses = {
            "actor1_loss": [], "actor2_loss": [],
            "critic_loss": [], "entropy1":    [], "entropy2": [],
        }

        for epoch in range(self.ppo_epochs):
            # Normalise advantages per agent independently
            adv1 = np.array(buffer.buffer1.advantages, dtype=np.float32)
            adv2 = np.array(buffer.buffer2.advantages, dtype=np.float32)
            adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)
            adv2 = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)
            buffer.buffer1.advantages = adv1
            buffer.buffer2.advantages = adv2

            # Iterate minibatches — zip both agent buffers
            for mb1, mb2 in zip(
                buffer.buffer1.get_minibatches(self.minibatch_sz),
                buffer.buffer2.get_minibatches(self.minibatch_sz),
            ):
                # ─── Agent 1 (MLP) loss ───────────────────────────────────
                l_a1, ent1 = self._actor1_loss(mb1)

                # ─── Agent 2 (TGIN) loss ──────────────────────────────────
                l_a2, ent2 = self._actor2_loss(mb2)

                # ─── Critic loss ──────────────────────────────────────────
                l_c = self._critic_loss(mb1, mb2)

                # ─── Combined update ──────────────────────────────────────
                total_loss = l_a1 + l_a2 + self.value_coef * l_c

                self.optim_actor1.zero_grad()
                self.optim_actor2.zero_grad()
                self.optim_critic.zero_grad()

                total_loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor1.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor2.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optim_actor1.step()
                self.optim_actor2.step()
                self.optim_critic.step()

                losses["actor1_loss"].append(l_a1.item())
                losses["actor2_loss"].append(l_a2.item())
                losses["critic_loss"].append(l_c.item())
                losses["entropy1"].append(ent1.item())
                losses["entropy2"].append(ent2.item())

        return {k: float(np.mean(v)) for k, v in losses.items()}


    def _actor1_loss(self, mb: dict) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        PPO clipped objective for Agent 1.
        Recomputes log probs under current policy for ratio computation.
        """
        import torch

        obs_list   = mb["obs"]
        old_lps    = torch.tensor(mb["log_probs"], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(mb["advantages"], dtype=torch.float32).to(self.device)

        # Stack observations
        obs_t = torch.tensor(
            np.array([o for o in obs_list]), dtype=torch.float32
        ).to(self.device)

        # Recompute log probs under current policy (no masking during update)
        maint_dist, reorder_dist = self.actor1(obs_t, None, None)

        actions = mb["actions"]  # list of action dicts
        maint_actions  = torch.tensor(
            np.array([a["maintenance"].flatten() for a in actions]), dtype=torch.long
        ).to(self.device)
        reorder_actions = torch.tensor(
            np.array([a["reorder"].flatten() for a in actions]), dtype=torch.long
        ).to(self.device)

        batch = obs_t.shape[0]
        n_mach = self.actor1.n_machines
        n_con  = self.actor1.n_consumable

        lp_maint = maint_dist.log_prob(
            maint_actions.view(-1)
        ).view(batch, n_mach).sum(-1)
        lp_reorder = reorder_dist.log_prob(
            reorder_actions.view(-1)
        ).view(batch, n_con).sum(-1)
        new_lps = lp_maint + lp_reorder

        entropy = maint_dist.entropy().mean() + reorder_dist.entropy().mean()

        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        return loss, entropy


    def _actor2_loss(self, mb: dict) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        PPO clipped objective for Agent 2.
        Agent 2's actions are integer indices into valid_pairs.
        Recomputes scores through TGIN + ActionScorer.

        Note: This is simplified — full implementation requires
        storing graph obs and rebuilding for each minibatch step.
        In practice, store logits during rollout for efficiency.
        """
        import torch

        old_lps    = torch.tensor(mb["log_probs"], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(mb["advantages"], dtype=torch.float32).to(self.device)

        # Placeholder: in full implementation, recompute through TGIN
        # For now, use importance sampling approximation with stored logits
        # TODO: store graph obs in buffer and recompute properly
        new_lps  = old_lps  # identity — no update without graph replay
        entropy  = torch.tensor(0.01)

        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        return loss, entropy


    def _critic_loss(self, mb1: dict, mb2: dict) -> "torch.Tensor":
        """
        MSE critic loss on joint returns R_joint = returns1 + returns2.
        Uses clipped value loss (PPO-style) for stability.
        """
        import torch

        returns = torch.tensor(
            mb1["returns"] + mb2["returns"], dtype=torch.float32
        ).to(self.device) / 2.0   # average joint return

        old_values = torch.tensor(
            mb1["values"], dtype=torch.float32
        ).to(self.device)

        # Critic recomputation requires global state — use stored values
        # TODO: recompute V(s) through critic with stored global state obs
        # For now, use stored values as proxy
        new_values = old_values

        # Clipped value loss (PPO trick)
        value_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_eps, self.clip_eps
        )
        loss1 = (new_values   - returns) ** 2
        loss2 = (value_clipped - returns) ** 2
        loss  = 0.5 * torch.max(loss1, loss2).mean()

        return loss
