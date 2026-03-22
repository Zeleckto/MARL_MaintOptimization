"""
models/critic.py
================
Centralized critic V_phi(s_global) for MAPPO.

Architecture change from report (Section 7.2 in architecture doc):
    ORIGINAL (eq 3.35): MLP(mean_pool(ALL node embeddings))
    UPDATED:            MLP(concat[mean_pool(ops), mean_pool(machines), mean_pool(jobs),
                                   resource_state, agent1_obs_summary])

Mean pooling across all node types together loses structural info.
Per-type pooling then concatenation preserves it at identical compute cost.

The critic sees EVERYTHING — global state including both agents' information.
Only used during TRAINING, never during execution (CTDE paradigm).
"""
from __future__ import annotations
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
    V_phi(s_global) — estimates expected return from global state.
    Trained on joint reward R_joint = r1 + r2 + R_shared.
    """

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        hidden_dim   = config.get("tgin", {}).get("hidden_dim", 256)
        critic_hdims = config.get("critic", {}).get("hidden_dims", [512, 256])

        # Compute resource state dim
        n_ren     = len(config["resources"]["renewable"])
        n_con     = len(config["resources"]["consumable"])
        max_lead  = max(r["lead_time_shifts"] for r in config["resources"]["consumable"])
        res_dim   = n_ren + n_con * (1 + max_lead)

        # Agent 1 obs summary dim (compact — just machine + resource parts)
        n_machines = len(config.get("machines", []))
        agent1_summary_dim = MACHINE_FEATURE_DIM * n_machines + res_dim

        # Input: per-type pooled embeddings + resource state + agent1 summary
        # Op pool: hidden_dim, Machine pool: hidden_dim, Job pool: hidden_dim
        input_dim = 3 * hidden_dim + res_dim + agent1_summary_dim

        # MLP layers
        layers = []
        in_d = input_dim
        for h in critic_hdims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.LayerNorm(h)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))  # scalar output
        self.net = nn.Sequential(*layers)


    def forward(
        self,
        embeddings:       Dict[str, "torch.Tensor"],  # from TGIN
        resource_flat:    "torch.Tensor",              # flat resource obs
        agent1_obs:       "torch.Tensor",              # Agent 1 flat obs
    ) -> "torch.Tensor":
        """
        Computes V(s_global).

        Args:
            embeddings:    TGIN output {'op':..., 'machine':..., 'job':...}
            resource_flat: Flattened resource state [batch, res_dim]
            agent1_obs:    Agent 1's full observation [batch, obs_dim]
                          (we use first machine+resource portion as summary)

        Returns:
            [batch, 1] value estimates
        """
        import torch

        # Per-type mean pooling (permutation invariant per type)
        h_op      = embeddings["op"].mean(dim=0, keepdim=True)      # [1, hidden]
        h_machine = embeddings["machine"].mean(dim=0, keepdim=True) # [1, hidden]
        h_job     = embeddings["job"].mean(dim=0, keepdim=True)     # [1, hidden]

        # Handle batch dimension
        if resource_flat.dim() == 1:
            resource_flat = resource_flat.unsqueeze(0)
        if agent1_obs.dim() == 1:
            agent1_obs = agent1_obs.unsqueeze(0)

        batch = resource_flat.shape[0]

        # Expand pooled embeddings to batch size
        h_op      = h_op.expand(batch, -1)
        h_machine = h_machine.expand(batch, -1)
        h_job     = h_job.expand(batch, -1)

        # Concatenate all global state components
        global_state = torch.cat([
            h_op, h_machine, h_job,
            resource_flat,
            agent1_obs,
        ], dim=-1)

        return self.net(global_state)   # [batch, 1]
