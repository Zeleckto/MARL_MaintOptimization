"""
agents/pdm_agent.py
====================
Agent 1 (PDM) wrapper.
Orchestrates: obs -> action masking -> MLPPolicy -> action
Stores (obs, action, logprob, value) for rollout buffer.
"""

from typing import Tuple, Optional, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.mlp_policy import MLPPolicy
from environments.spaces.action_spaces import (
    build_agent1_maintenance_mask,
    build_agent1_reorder_mask,
)


class PDMAgent:
    """
    Agent 1: Predictive Maintenance + Resource Ordering decisions.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.n_machines   = len(config.get("machines", []))
        self.n_consumable = len(config["resources"]["consumable"])
        self.n_renewable  = len(config["resources"]["renewable"])

        if TORCH_AVAILABLE:
            self.policy = MLPPolicy(config).to(device)
        else:
            self.policy = None


    def act(
        self,
        obs_np:        np.ndarray,
        machine_states,
        machine_busy:  List[bool],
        resource_state,
        rho_PM:        np.ndarray,
        rho_CM:        np.ndarray,
    ) -> Tuple[dict, float, float]:
        """
        Selects Agent 1's action with masking applied.

        Args:
            obs_np:        Agent 1 flat observation [obs_dim]
            machine_states: Current machine states
            machine_busy:  [n_machines] True if machine processing job
            resource_state: Current resource state
            rho_PM:        Resource requirements for PM
            rho_CM:        Resource requirements for CM

        Returns:
            (action_dict, log_prob, entropy)
            action_dict: {'maintenance': np.ndarray [n_mach],
                          'reorder':     np.ndarray [n_con]}
        """
        if not TORCH_AVAILABLE or self.policy is None:
            # Random action for testing without torch
            return self._random_action(), 0.0, 0.0

        import torch

        obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Build masks
        maint_mask_np = build_agent1_maintenance_mask(
            machine_states, machine_busy, resource_state,
            rho_PM, rho_CM, self.n_renewable
        )
        rho_CM_max = rho_CM[:, self.n_renewable:].max(axis=0)
        reorder_mask_np = build_agent1_reorder_mask(resource_state, rho_CM_max)

        maint_mask  = torch.tensor(maint_mask_np,  dtype=torch.bool).unsqueeze(0).to(self.device)
        reorder_mask = torch.tensor(reorder_mask_np, dtype=torch.bool).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_dict, log_prob, entropy = self.policy.get_action(
                obs_t, maint_mask, reorder_mask
            )

        return {
            "maintenance": action_dict["maintenance"].squeeze(0).cpu().numpy(),
            "reorder":     action_dict["reorder"].squeeze(0).cpu().numpy(),
        }, log_prob.item(), entropy.item()


    def _random_action(self) -> dict:
        """Random action for testing (no torch)."""
        return {
            "maintenance": np.zeros(self.n_machines, dtype=int),
            "reorder":     np.zeros(self.n_consumable, dtype=float),
        }


    def parameters(self):
        """Returns policy parameters for optimizer."""
        if self.policy:
            return self.policy.parameters()
        return []
