"""
models/mlp_policy.py
=====================
Agent 1 (PDM) actor network — MLP policy.

Input:  Agent 1's flat observation vector
Output: Two heads:
    - Maintenance logits: [n_machines, 3] (none/PM/CM per machine)
    - Reorder logits:     [n_consumable, Q_max+1] (order quantity per resource)

Masking applied BEFORE softmax (set invalid logits to -inf).
This ensures invalid actions have exactly zero probability.
"""
from __future__ import annotations
from typing import Tuple, Optional, List

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLPPolicy(nn.Module if TORCH_AVAILABLE else object):
    """
    Agent 1 actor: flat obs -> maintenance + reorder action distributions.
    """

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        # Compute input dimension
        from environments.spaces.observation_spaces import compute_agent1_obs_dim
        obs_dim = compute_agent1_obs_dim(config)

        n_machines   = len(config.get("machines", []))
        n_consumable = len(config["resources"]["consumable"])
        q_max        = config.get("q_max", 10)

        hidden_dims = config.get("mlp_policy", {}).get("hidden_dims", [256, 256])

        # Shared trunk
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Maintenance head: [n_machines * 3] logits
        # Reshape to [n_machines, 3] after forward
        self.maint_head = nn.Linear(in_dim, n_machines * 3)
        self.n_machines = n_machines

        # Reorder head: [n_consumable * (q_max+1)] logits
        self.reorder_head = nn.Linear(in_dim, n_consumable * (q_max + 1))
        self.n_consumable = n_consumable
        self.q_max        = q_max


    def forward(
        self,
        obs:          "torch.Tensor",            # [batch, obs_dim]
        maint_mask:   Optional["torch.Tensor"],  # [batch, n_machines, 3] bool
        reorder_mask: Optional["torch.Tensor"],  # [batch, n_consumable] bool
    ) -> Tuple["Categorical", "Categorical"]:
        """
        Computes action distributions for Agent 1.

        Args:
            obs:          Flat observation tensor
            maint_mask:   True = action ALLOWED. None = all allowed.
            reorder_mask: True = reorder ALLOWED for this resource.

        Returns:
            (maint_dist, reorder_dist)
            maint_dist:  Categorical over 3 actions per machine
                         Shape: [batch * n_machines, 3] -> sample [batch, n_machines]
            reorder_dist: Categorical over Q_max+1 quantities per resource
        """
        import torch

        features = self.trunk(obs)  # [batch, hidden]

        # --- Maintenance head ---
        maint_logits = self.maint_head(features)              # [batch, n_machines*3]
        maint_logits = maint_logits.view(-1, self.n_machines, 3)  # [batch, n_mach, 3]

        if maint_mask is not None:
            # Set invalid action logits to -inf
            maint_logits = maint_logits.masked_fill(~maint_mask, float("-inf"))

        # Flatten batch*machines for Categorical
        batch = maint_logits.shape[0]
        maint_logits_flat = maint_logits.view(batch * self.n_machines, 3)
        maint_dist = Categorical(logits=maint_logits_flat)

        # --- Reorder head ---
        reorder_logits = self.reorder_head(features)          # [batch, n_con*(Q+1)]
        reorder_logits = reorder_logits.view(
            -1, self.n_consumable, self.q_max + 1
        )  # [batch, n_con, Q+1]

        if reorder_mask is not None:
            # Block all reorder quantities for masked resources
            block = (~reorder_mask).unsqueeze(-1).expand_as(reorder_logits)
            # Allow only action=0 (no order) for blocked resources
            no_order_mask = torch.zeros_like(reorder_logits, dtype=torch.bool)
            no_order_mask[:, :, 1:] = block[:, :, 1:]  # block qty >= 1
            reorder_logits = reorder_logits.masked_fill(no_order_mask, float("-inf"))

        reorder_logits_flat = reorder_logits.view(batch * self.n_consumable, self.q_max + 1)
        reorder_dist = Categorical(logits=reorder_logits_flat)

        return maint_dist, reorder_dist


    def get_action(
        self,
        obs:          "torch.Tensor",
        maint_mask:   Optional["torch.Tensor"],
        reorder_mask: Optional["torch.Tensor"],
    ) -> Tuple[dict, "torch.Tensor", "torch.Tensor"]:
        """
        Samples actions and returns log probs for rollout collection.

        Returns:
            (action_dict, log_prob, entropy)
            action_dict: {'maintenance': [n_mach], 'reorder': [n_con]}
            log_prob:    sum of log probs across all action dims
            entropy:     sum of entropies (for entropy bonus in PPO)
        """
        import torch

        maint_dist, reorder_dist = self.forward(obs, maint_mask, reorder_mask)

        maint_action  = maint_dist.sample()    # [batch * n_mach]
        reorder_action = reorder_dist.sample()  # [batch * n_con]

        batch = obs.shape[0]
        maint_action   = maint_action.view(batch, self.n_machines)
        reorder_action = reorder_action.view(batch, self.n_consumable)

        # Log prob: sum across all independent action dimensions
        lp_maint  = maint_dist.log_prob(maint_action.view(-1)).view(batch, self.n_machines).sum(-1)
        lp_reorder = reorder_dist.log_prob(reorder_action.view(-1)).view(batch, self.n_consumable).sum(-1)
        log_prob  = lp_maint + lp_reorder

        entropy = maint_dist.entropy().mean() + reorder_dist.entropy().mean()

        action_dict = {
            "maintenance": maint_action,
            "reorder":     reorder_action,
        }

        return action_dict, log_prob, entropy
