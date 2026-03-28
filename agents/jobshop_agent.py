from __future__ import annotations
"""
agents/jobshop_agent.py
========================
Agent 2 (Job Shop) wrapper.

Key addition: get_log_prob() method used during PPO update to compute
π_new(a|obs) — the log probability of the stored action under the
CURRENT (updated) policy weights. This enables proper PPO importance
sampling ratio: r_t = exp(new_logp - old_logp).
"""

from typing import Tuple, Optional, List, Dict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.tgin.graph_builder import GraphBuilder
from models.tgin.tgin import TGIN
from models.tgin.action_scorer import ActionScorer


class JobShopAgent:
    """Agent 2: Job-Machine assignment using TGIN policy."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        if TORCH_AVAILABLE:
            self.graph_builder = GraphBuilder(config)
            self.tgin          = TGIN(config).to(device)
            self.action_scorer = ActionScorer(config).to(device)
        else:
            self.graph_builder = None
            self.tgin          = None
            self.action_scorer = None


    def _forward(self, obs: dict) -> Tuple:
        """
        Shared forward pass: obs dict → graph → TGIN → (dist, op_id_map).
        Used by both act() and get_log_prob() to avoid code duplication.

        Returns:
            (dist, valid_pairs, op_id_map)
        """
        import torch

        valid_pairs = obs.get("valid_pairs", [])
        graph       = self.graph_builder.build(obs, device=self.device)
        embeddings  = self.tgin(graph)

        # Build op_id_map — maps (job_id, op_idx) → node index in graph
        # Op nodes are ordered as all pending ops in job_id order
        # We approximate by clamping to graph size (exact fix in Phase 2)
        n_op_nodes = embeddings["op"].shape[0]
        op_id_map  = {}
        for i, (job_id, op_idx, _) in enumerate(valid_pairs):
            key = (job_id, op_idx)
            if key not in op_id_map:
                op_id_map[key] = min(i, n_op_nodes - 1)

        dist, logits = self.action_scorer(embeddings, valid_pairs, op_id_map)
        return dist, valid_pairs, op_id_map


    def act(
        self,
        obs:         dict,
        valid_pairs: List[Tuple[int, int, int]],
    ) -> Tuple[Optional[Tuple[int, int, int]], int, float, float]:
        """
        Selects action. Called during rollout collection.

        Returns:
            (semantic_action, action_idx, log_prob, entropy)
        """
        if not TORCH_AVAILABLE:
            return None, len(valid_pairs), 0.0, 0.0

        import torch

        with torch.no_grad():
            dist, vp, _ = self._forward(obs)
            action_idx  = dist.sample().item()
            log_prob    = dist.log_prob(
                torch.tensor(action_idx, device=self.device)
            ).item()
            entropy = dist.entropy().item()

        semantic = vp[action_idx] if action_idx < len(vp) else None
        return semantic, action_idx, log_prob, entropy


    def get_log_prob(
        self,
        obs:        dict,
        action_idx: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Computes log prob and entropy for a stored action under CURRENT weights.

        This IS differentiable — the forward pass through self.tgin and
        self.action_scorer creates a live computation graph so .backward()
        will update TGIN weights.

        Called during PPO update (not during rollout — no torch.no_grad() here).

        Args:
            obs:        Stored obs dict from rollout buffer
            action_idx: The action that was taken (stored in buffer)

        Returns:
            (log_prob_tensor, entropy_tensor) — both with gradient
        """
        import torch

        dist, vp, _ = self._forward(obs)

        # Clamp action_idx to valid distribution size
        n_actions   = len(vp) + 1   # valid pairs + WAIT
        action_idx  = min(int(action_idx), n_actions - 1)
        action_t    = torch.tensor(action_idx, device=self.device)

        log_prob = dist.log_prob(action_t)    # has gradient ✓
        entropy  = dist.entropy()             # has gradient ✓

        return log_prob, entropy


    def parameters(self):
        if self.tgin and self.action_scorer:
            return list(self.tgin.parameters()) + \
                   list(self.action_scorer.parameters())
        return []