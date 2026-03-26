from __future__ import annotations
"""
models/critic.py
================
Centralized critic V_phi(s_global) for MAPPO.

Input (all concatenated):
    - Per-type mean-pooled TGIN embeddings (DETACHED from TGIN graph)
      → prevents critic loss from corrupting TGIN weights used by Agent 2 actor
    - Resource flat state vector
    - Agent 1's full observation
    - Agent 1's last maintenance action (one-hot per machine, 3 actions each)
      → critic can reason about consequences of what Agent 1 just decided

Why detach():
    TGIN weights θ₂ are shared between Agent 2 actor and critic.
    Without detach(), critic loss backpropagates into θ₂, creating a
    moving-target problem: critic and actor fight over the same weights.
    Solution: detach embeddings before passing to critic MLP.
    TGIN learns only from actor loss. Critic trains a linear map on top.
    Slightly less expressive but stable. Upgrade to separate TGIN if instability seen.

Why add action1:
    If Agent 1 just sent 3 machines to PM, future availability drops.
    The critic seeing only the pre-action state cannot estimate this correctly.
    Adding one-hot action1 gives the critic causal information it needs.
"""

from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from environments.spaces.observation_spaces import (
    OP_FEATURE_DIM, MACHINE_FEATURE_DIM, JOB_FEATURE_DIM
)


class CentralizedCritic(nn.Module if TORCH_AVAILABLE else object):
    """
    V_phi(s_global) — scalar value estimate from global state.
    Trained on R_joint = r1 + r2 + R_shared.
    """

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        hidden_dim   = config.get("tgin", {}).get("hidden_dim", 256)
        critic_hdims = config.get("critic", {}).get("hidden_dims", [512, 256])

        # Resource state dim
        n_ren     = len(config["resources"]["renewable"])
        n_con     = len(config["resources"]["consumable"])
        max_lead  = max(r["lead_time_shifts"] for r in config["resources"]["consumable"])
        res_dim   = n_ren + n_con * (1 + max_lead)

        # Agent 1 obs dim
        n_machines = len(config.get("machines", []))
        agent1_obs_dim = MACHINE_FEATURE_DIM * n_machines + res_dim + 5  # +5 job summary

        # Agent 1 action dim (one-hot: n_machines × 3 maintenance actions)
        action1_dim = n_machines * 3

        # Input: 3 × hidden (pooled graph) + resource + agent1_obs + action1
        input_dim = 3 * hidden_dim + res_dim + agent1_obs_dim + action1_dim

        # MLP
        layers = []
        in_d = input_dim
        for h in critic_hdims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.LayerNorm(h)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

        self.n_machines  = n_machines
        self.hidden_dim  = hidden_dim


    def forward(
        self,
        embeddings:    Dict[str, "torch.Tensor"],  # from TGIN
        resource_flat: "torch.Tensor",              # [batch, res_dim]
        agent1_obs:    "torch.Tensor",              # [batch, obs_dim]
        action1_maint: Optional["torch.Tensor"] = None,  # [batch, n_machines] int
    ) -> "torch.Tensor":
        """
        Computes V(s_global).

        Args:
            embeddings:    TGIN output — WILL BE DETACHED inside this method
            resource_flat: Flattened resource state
            agent1_obs:    Agent 1 full observation
            action1_maint: Agent 1 maintenance actions [batch, n_machines]
                           Each in {0,1,2}. None = zeros (no action info).

        Returns:
            [batch, 1] value estimates
        """
        import torch

        # DETACH embeddings to prevent critic loss from corrupting TGIN actor weights
        h_op      = embeddings["op"].detach().mean(dim=0, keepdim=True)
        h_machine = embeddings["machine"].detach().mean(dim=0, keepdim=True)
        h_job     = embeddings["job"].detach().mean(dim=0, keepdim=True)

        # Handle batch dim
        if resource_flat.dim() == 1: resource_flat = resource_flat.unsqueeze(0)
        if agent1_obs.dim()    == 1: agent1_obs    = agent1_obs.unsqueeze(0)

        batch = resource_flat.shape[0]

        h_op      = h_op.expand(batch, -1)
        h_machine = h_machine.expand(batch, -1)
        h_job     = h_job.expand(batch, -1)

        # One-hot encode Agent 1 maintenance actions
        if action1_maint is not None:
            # action1_maint: [batch, n_machines] each in {0,1,2}
            action1_oh = torch.zeros(batch, self.n_machines * 3, device=resource_flat.device)
            for m in range(self.n_machines):
                action_col = action1_maint[:, m].long()  # [batch]
                idx = m * 3 + action_col
                action1_oh.scatter_(1, idx.unsqueeze(1), 1.0)
        else:
            action1_oh = torch.zeros(batch, self.n_machines * 3, device=resource_flat.device)

        global_state = torch.cat([
            h_op, h_machine, h_job,
            resource_flat,
            agent1_obs,
            action1_oh,
        ], dim=-1)

        return self.net(global_state)   # [batch, 1]