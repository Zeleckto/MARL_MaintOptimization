"""
models/tgin/action_scorer.py
==============================
Scores (op, machine) pairs using TGIN embeddings.
Applies mask and returns Categorical distribution for PPO sampling.

This is the policy head for Agent 2.
Input:  TGIN output embeddings + valid pair indices
Output: Categorical distribution over valid pairs + WAIT action
"""

from typing import List, Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ActionScorer(nn.Module if TORCH_AVAILABLE else object):
    """
    Scores valid (op, machine) pairs and returns action distribution.

    Architecture:
        score(o, m) = MLP_score(concat[h_o, h_m]) -> scalar
        pi(o,m|s) = softmax(masked_scores) over valid pairs
        WAIT action always appended as last option.
    """

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        hidden_dim = config.get("tgin", {}).get("hidden_dim", 256)

        # MLP: concat(h_op, h_machine) [2*hidden] -> score [1]
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable score for WAIT action
        # Initialised to small negative value — WAIT only if nothing better
        self.wait_score = nn.Parameter(torch.tensor(-1.0))


    def forward(
        self,
        embeddings:  Dict[str, "torch.Tensor"],
        valid_pairs: List[Tuple[int, int, int]],   # (job_id, op_idx, machine_id)
        op_id_map:   Dict[Tuple[int, int], int],   # (job_id, op_idx) -> node index
    ) -> Tuple["Categorical", "torch.Tensor"]:
        """
        Computes action distribution over valid (op, machine) pairs.

        Args:
            embeddings:  Dict from TGIN.forward() {'op':..., 'machine':..., 'job':...}
            valid_pairs: List of valid (job_id, op_idx, machine_id) tuples
            op_id_map:   Maps (job_id, op_idx) -> op node index in graph

        Returns:
            (distribution, logits)
            distribution: Categorical over len(valid_pairs)+1 actions
                         (last index = WAIT)
            logits: raw scores before softmax [n_valid+1]
        """
        import torch

        h_op      = embeddings["op"]      # [n_ops, hidden_dim]
        h_machine = embeddings["machine"] # [n_machines, hidden_dim]

        if not valid_pairs:
            # Only WAIT available
            logits = self.wait_score.unsqueeze(0)
            dist   = Categorical(logits=logits)
            return dist, logits

        # Score each valid (op, machine) pair
        scores = []
        for job_id, op_idx, machine_id in valid_pairs:
            op_node_idx = op_id_map.get((job_id, op_idx), 0)

            h_o = h_op[op_node_idx]          # [hidden_dim]
            h_m = h_machine[machine_id]      # [hidden_dim]

            pair_feat = torch.cat([h_o, h_m], dim=-1)  # [2*hidden_dim]
            score = self.score_mlp(pair_feat).squeeze(-1)
            scores.append(score)

        pair_scores = torch.stack(scores)             # [n_valid]
        all_scores  = torch.cat([pair_scores,
                                  self.wait_score.unsqueeze(0)])  # [n_valid+1]

        dist = Categorical(logits=all_scores)
        return dist, all_scores


    def get_log_prob(
        self,
        embeddings:  Dict[str, "torch.Tensor"],
        valid_pairs: List[Tuple[int, int, int]],
        op_id_map:   Dict[Tuple[int, int], int],
        action:      "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Computes log probability and entropy for a given action.
        Used during PPO update to compute the policy ratio.

        Args:
            action: Sampled action index [batch] or scalar

        Returns:
            (log_prob, entropy)
        """
        dist, _ = self.forward(embeddings, valid_pairs, op_id_map)
        return dist.log_prob(action), dist.entropy()
