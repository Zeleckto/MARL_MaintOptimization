from __future__ import annotations
"""
models/tgin/action_scorer.py — v4 (Feature MLP, smart init)
=============================================================
Pure hand-crafted features → score. No TGIN for action selection.
Initialized to approximate fewest-ops-left from step 0.

Architecture: Linear(7→1) + MLP(7→64→32→1), blended.
Linear head starts with fewest-ops-left weights.
MLP head starts random, learns during RL.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

PAIR_FEATURE_DIM = 7


class ActionScorer(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        self.hidden_dim = config.get("tgin", {}).get("hidden_dim", 256)

        # ── Linear head: initialized to fewest-ops-left ──────────
        # Produces a score from day 0 that approximates the expert
        self.linear_head = nn.Linear(PAIR_FEATURE_DIM, 1)
        with torch.no_grad():
            # [remaining_ops, proc_time, slack, health, is_last_op, progress, urgency]
            self.linear_head.weight[0] = torch.tensor([-2.5, -3.5, 0.5, 0.5, 3.0, 2.0, 0.5])
            self.linear_head.bias[0] = 1.0

        # ── MLP head: learns residual corrections during RL ──────
        self.mlp_head = nn.Sequential(
            nn.Linear(PAIR_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Init MLP to output ~0 so linear head dominates at start
        with torch.no_grad():
            self.mlp_head[-1].weight.mul_(0.01)
            self.mlp_head[-1].bias.zero_()

        self.wait_score = nn.Parameter(torch.tensor(-2.0))

    def _build_pair_features(self, valid_pairs, obs=None):
        import torch
        from environments.transitions.job_dynamics import OpStatus

        features = []
        jobs = obs.get("_jobs", []) if obs else []
        machine_states = obs.get("_machine_states", []) if obs else []
        current_step = obs.get("_current_step", 0) if obs else 0
        t_max = obs.get("_t_max", 150) if obs else 150

        for job_id, op_idx, machine_id in valid_pairs:
            job = None
            for j in jobs:
                if j.job_id == job_id:
                    job = j; break

            if job is None:
                features.append([0.5, 0.5, 0.0, 0.8, 0.0, 0.5, 0.0])
                continue

            n_total = len(job.operations)
            n_done = sum(1 for op in job.operations if op.status == OpStatus.DONE)
            remaining = n_total - n_done
            op = job.operations[op_idx]
            pt = min(op.nominal_proc_times.values()) / 8.0 if op.nominal_proc_times else 5.0
            remaining_work = sum(
                min(o.nominal_proc_times.values()) / 8.0 if o.nominal_proc_times else 5.0
                for o in job.operations if o.status != OpStatus.DONE
            )
            slack = (job.due_date - current_step - remaining_work) / max(t_max, 1)
            health = machine_states[machine_id].health / 100.0 if machine_id < len(machine_states) else 0.8

            features.append([
                remaining / max(n_total, 1),
                pt / 10.0,
                float(np.clip(slack, -1.0, 1.0)),
                health,
                1.0 if remaining == 1 else 0.0,
                n_done / max(n_total, 1),
                float(np.clip(-slack / max(pt / 10, 0.01), -2.0, 2.0)),
            ])

        if not features:
            return torch.zeros(1, PAIR_FEATURE_DIM)
        return torch.tensor(features, dtype=torch.float32)

    def forward(self, embeddings, valid_pairs, op_id_map, obs=None):
        import torch

        if not valid_pairs:
            logits = self.wait_score.unsqueeze(0)
            return Categorical(logits=logits), logits

        device = next(self.linear_head.parameters()).device
        feats = self._build_pair_features(valid_pairs, obs).to(device)

        # Linear head (initialized, dominates early)
        linear_scores = self.linear_head(feats).squeeze(-1)
        # MLP head (random→0 initially, learns corrections)
        mlp_scores = self.mlp_head(feats).squeeze(-1)
        # Combined
        scores = linear_scores + mlp_scores

        all_scores = torch.cat([scores, self.wait_score.unsqueeze(0)])
        return Categorical(logits=all_scores), all_scores

    def get_log_prob(self, embeddings, valid_pairs, op_id_map, action, obs=None):
        dist, _ = self.forward(embeddings, valid_pairs, op_id_map, obs=obs)
        return dist.log_prob(action), dist.entropy()