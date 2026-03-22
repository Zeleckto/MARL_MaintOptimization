"""
models/tgin/tgin.py
====================
Tripartite Graph Isomorphism Network (TGIN) for Agent 2.

Architecture (Section 3.3.4 in report):
    - 3 node types: Operations, Machines, Jobs
    - 3 edge types: Op->Machine, Machine->Job, Op->Job
    - L=3 rounds of bidirectional message passing
    - Forward: Op->Machine->Job
    - Backward: Job->Machine->Op
    - GIN update rule: h_v^(l+1) = MLP((1+eps)*h_v^(l) + sum(MLP(h_u + e_uv)))

Why GIN over GAT/GraphSAGE:
    GIN is maximally expressive among message-passing GNNs (Xu et al. 2019).
    Two scheduling states that look similar but differ in due dates must be
    distinguished — GIN's sum aggregation preserves this structural info.
    GAT attention can wash out minority-feature bottleneck machines.

Output:
    Per-node embeddings {h_op, h_machine, h_job} of shape [n_nodes, hidden_dim]
    These are passed to action_scorer.py for (op, machine) pair scoring.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import HeteroConv, GINEConv, Linear
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GINELayer(nn.Module if TORCH_AVAILABLE else object):
    """
    Single GIN-E (GIN with Edge features) message passing layer.
    Implements: h_v^(l+1) = MLP((1+eps)*h_v^(l) + sum_{u in N(v)} MLP(h_u + e_uv))

    Using GINE (GIN with edge features) instead of plain GIN because
    our edges carry meaningful features (processing times, progress).
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, eps: float = 0.0):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        # MLP for message aggregation: maps (node + edge) features -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        # Edge feature projection to match node feature dim
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:          [n_nodes, in_dim] node features
            edge_index: [2, n_edges] (src, dst)
            edge_attr:  [n_edges, edge_dim] edge features

        Returns:
            [n_nodes, out_dim] updated node embeddings
        """
        from torch_geometric.utils import scatter
        import torch

        # Project edge features to node feature space
        edge_feat = self.edge_proj(edge_attr)   # [n_edges, in_dim]

        # Aggregate: for each dst node, sum (src_features + edge_features)
        src, dst = edge_index
        messages = x[src] + edge_feat           # [n_edges, in_dim]

        # Sum aggregation at destination nodes
        aggr = torch.zeros_like(x)              # [n_nodes, in_dim]
        aggr.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # GIN update: (1+eps) * self + aggregated neighbours
        out = self.mlp((1 + self.eps) * x + aggr)
        return out


class TGIN(nn.Module if TORCH_AVAILABLE else object):
    """
    Full Tripartite GIN: L rounds of bidirectional message passing.

    Input:  HeteroData graph with op/machine/job nodes and 3 edge types
    Output: Dict of updated node embeddings {'op': ..., 'machine': ..., 'job': ...}
    """

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        tgin_cfg    = config.get("tgin", {})
        self.n_layers   = tgin_cfg.get("n_layers", 3)
        self.hidden_dim = tgin_cfg.get("hidden_dim", 256)
        eps             = tgin_cfg.get("eps", 0.0)

        op_dim      = tgin_cfg.get("op_feature_dim",      10)
        machine_dim = tgin_cfg.get("machine_feature_dim", 15)
        job_dim     = tgin_cfg.get("job_feature_dim",     7)
        edge_dim    = 2   # all edge types have 2-dim features

        # Input projection: map raw features to hidden_dim
        self.op_proj      = nn.Linear(op_dim,      self.hidden_dim)
        self.machine_proj = nn.Linear(machine_dim, self.hidden_dim)
        self.job_proj     = nn.Linear(job_dim,     self.hidden_dim)

        # Message passing layers for each direction and each round
        # Forward:  op->machine, machine->job, op->job
        # Backward: job->machine, machine->op
        self.forward_op_to_mach  = nn.ModuleList()
        self.forward_mach_to_job = nn.ModuleList()
        self.forward_op_to_job   = nn.ModuleList()
        self.backward_job_to_mach = nn.ModuleList()
        self.backward_mach_to_op  = nn.ModuleList()

        for _ in range(self.n_layers):
            self.forward_op_to_mach.append(
                GINELayer(self.hidden_dim, self.hidden_dim, edge_dim, eps)
            )
            self.forward_mach_to_job.append(
                GINELayer(self.hidden_dim, self.hidden_dim, edge_dim, eps)
            )
            self.forward_op_to_job.append(
                GINELayer(self.hidden_dim, self.hidden_dim, edge_dim, eps)
            )
            self.backward_job_to_mach.append(
                GINELayer(self.hidden_dim, self.hidden_dim, edge_dim, eps)
            )
            self.backward_mach_to_op.append(
                GINELayer(self.hidden_dim, self.hidden_dim, edge_dim, eps)
            )


    def forward(self, data) -> Dict[str, "torch.Tensor"]:
        """
        Full forward pass: L rounds of bidirectional message passing.

        Args:
            data: PyG HeteroData (or Batch for parallel envs)

        Returns:
            Dict with keys 'op', 'machine', 'job' -> [n_nodes, hidden_dim] tensors
        """
        import torch

        # --- Input projection ---
        h_op      = torch.relu(self.op_proj(data["op"].x))
        h_machine = torch.relu(self.machine_proj(data["machine"].x))
        h_job     = torch.relu(self.job_proj(data["job"].x))

        # --- L rounds of message passing ---
        for l in range(self.n_layers):

            # === FORWARD PASS: Op -> Machine -> Job ===

            # Step 1: Op messages to Machine
            ei_om = data["op", "to", "machine"].edge_index
            ea_om = data["op", "to", "machine"].edge_attr
            if ei_om.shape[1] > 0:
                # Build message: each machine receives from its eligible ops
                # GINELayer handles aggregation internally
                h_machine_new = self._message_pass(
                    src_features=h_op,
                    dst_features=h_machine,
                    edge_index=ei_om,
                    edge_attr=ea_om,
                    layer=self.forward_op_to_mach[l],
                    n_dst=h_machine.shape[0],
                )
            else:
                h_machine_new = h_machine

            # Step 2: Machine messages to Job
            ei_mj = data["machine", "to", "job"].edge_index
            ea_mj = data["machine", "to", "job"].edge_attr
            if ei_mj.shape[1] > 0:
                h_job_new = self._message_pass(
                    src_features=h_machine_new,
                    dst_features=h_job,
                    edge_index=ei_mj,
                    edge_attr=ea_mj,
                    layer=self.forward_mach_to_job[l],
                    n_dst=h_job.shape[0],
                )
            else:
                h_job_new = h_job

            # Step 3: Op structural messages to Job (parallel with step 2)
            ei_oj = data["op", "to", "job"].edge_index
            ea_oj = data["op", "to", "job"].edge_attr
            if ei_oj.shape[1] > 0:
                h_job_from_op = self._message_pass(
                    src_features=h_op,
                    dst_features=h_job,
                    edge_index=ei_oj,
                    edge_attr=ea_oj,
                    layer=self.forward_op_to_job[l],
                    n_dst=h_job.shape[0],
                )
                h_job_new = h_job_new + h_job_from_op  # combine both signals

            # === BACKWARD PASS: Job -> Machine -> Op ===

            # Step 4: Job messages back to Machine
            # Reverse edge direction for backward pass
            if ei_mj.shape[1] > 0:
                ei_jm_rev = torch.stack([ei_mj[1], ei_mj[0]])
                h_machine_new = self._message_pass(
                    src_features=h_job_new,
                    dst_features=h_machine_new,
                    edge_index=ei_jm_rev,
                    edge_attr=ea_mj,
                    layer=self.backward_job_to_mach[l],
                    n_dst=h_machine_new.shape[0],
                )

            # Step 5: Machine messages back to Op
            if ei_om.shape[1] > 0:
                ei_mo_rev = torch.stack([ei_om[1], ei_om[0]])
                h_op = self._message_pass(
                    src_features=h_machine_new,
                    dst_features=h_op,
                    edge_index=ei_mo_rev,
                    edge_attr=ea_om,
                    layer=self.backward_mach_to_op[l],
                    n_dst=h_op.shape[0],
                )

            # Update embeddings for next round
            h_machine = h_machine_new
            h_job     = h_job_new

        return {
            "op":      h_op,
            "machine": h_machine,
            "job":     h_job,
        }


    def _message_pass(
        self,
        src_features: "torch.Tensor",
        dst_features: "torch.Tensor",
        edge_index:   "torch.Tensor",
        edge_attr:    "torch.Tensor",
        layer:        GINELayer,
        n_dst:        int,
    ) -> "torch.Tensor":
        """
        Performs one direction of message passing.
        Aggregates src messages at dst nodes, then applies GIN update to dst.

        Args:
            src_features: [n_src, hidden_dim]
            dst_features: [n_dst, hidden_dim]
            edge_index:   [2, n_edges] — row 0 = src, row 1 = dst
            edge_attr:    [n_edges, edge_dim]
            layer:        GINELayer to apply
            n_dst:        Number of destination nodes

        Returns:
            [n_dst, hidden_dim] updated destination embeddings
        """
        import torch

        src_idx, dst_idx = edge_index

        # Project edge features
        edge_feat = layer.edge_proj(edge_attr)   # [n_edges, hidden_dim]

        # Compute messages: src_features + edge_features
        messages = src_features[src_idx] + edge_feat  # [n_edges, hidden_dim]

        # Aggregate at destination (sum)
        aggr = torch.zeros(n_dst, layer.mlp[0].in_features,
                           device=src_features.device)
        aggr.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(messages), messages)

        # GIN update at destination
        return layer.mlp((1 + layer.eps) * dst_features + aggr)
