from __future__ import annotations
"""
agents/jobshop_agent.py
========================
Agent 2 (Job Shop) wrapper.
Orchestrates: obs -> graph_builder -> TGIN -> action_scorer -> (j,k,m)
Stores (obs, action, logprob) for rollout buffer.
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
    """
    Agent 2: Job-Machine assignment and operation sequencing.
    Uses TGIN policy network on tripartite graph representation.
    """

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


    def act(
        self,
        obs:         dict,   # Agent 2 obs dict from mfg_env._build_agent2_obs()
        valid_pairs: List[Tuple[int, int, int]],
    ) -> Tuple[Optional[Tuple[int, int, int]], int, float, float]:
        """
        Selects Agent 2's scheduling action.

        Args:
            obs:         Graph observation dict
            valid_pairs: Valid (job_id, op_idx, machine_id) tuples

        Returns:
            (semantic_action, action_idx, log_prob, entropy)
            semantic_action: (job_id, op_idx, machine_id) or None (WAIT)
            action_idx:      Index sampled (last idx = WAIT)
        """
        if not TORCH_AVAILABLE:
            return None, len(valid_pairs), 0.0, 0.0

        import torch

        # Build graph
        graph = self.graph_builder.build(obs, device=self.device)

        # Build op_id_map: (job_id, op_idx) -> node index
        op_id_map = self._build_op_id_map(obs)

        # TGIN forward pass
        with torch.no_grad():
            embeddings = self.tgin(graph)
            dist, logits = self.action_scorer(embeddings, valid_pairs, op_id_map)
            action_idx = dist.sample().item()
            log_prob   = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
            entropy    = dist.entropy().item()

        # Decode action
        if action_idx < len(valid_pairs):
            semantic_action = valid_pairs[action_idx]
        else:
            semantic_action = None  # WAIT

        return semantic_action, action_idx, log_prob, entropy


    def _build_op_id_map(self, obs: dict) -> Dict[Tuple[int, int], int]:
        """
        Reconstructs (job_id, op_idx) -> node_index mapping from obs dict.
        The obs dict from mfg_env contains valid_pairs which give us the mapping.
        """
        # The op nodes in the graph are built in the same order as
        # pending ops in mfg_env._build_agent2_obs()
        # We reconstruct the mapping from valid_pairs
        op_id_map = {}
        for i, (job_id, op_idx, _) in enumerate(obs.get("valid_pairs", [])):
            key = (job_id, op_idx)
            if key not in op_id_map:
                op_id_map[key] = i  # approximate — exact mapping via env
        return op_id_map


    def parameters(self):
        """Returns all trainable parameters for optimizer."""
        if self.tgin and self.action_scorer:
            return list(self.tgin.parameters()) + list(self.action_scorer.parameters())
        return []