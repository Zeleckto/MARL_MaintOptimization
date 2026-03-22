"""
models/tgin/graph_builder.py
==============================
Converts environment observation dict (from mfg_env._build_agent2_obs())
into a PyG HeteroData object for TGIN input.

This is the trickiest file in the project. Key challenges:
    1. Graph size changes every step (Poisson arrivals, op completions)
    2. Must be called every step after Agent 1 half-step
    3. Must support batching across 4 parallel envs (PyG Batch.from_data_list)
    4. Edge validity changes with machine status changes

Kept separate from tgin.py so graph construction can be tested independently
of the neural network.

Usage:
    builder = GraphBuilder(config)
    graph = builder.build(obs_dict)     # single env
    batch = builder.build_batch([o1, o2, o3, o4])   # 4 parallel envs
"""

from typing import List, Dict, Optional

try:
    import torch
    from torch_geometric.data import HeteroData, Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


class GraphBuilder:
    """
    Converts Agent 2 observation dict to PyG HeteroData.

    The obs dict comes from mfg_env._build_agent2_obs() and contains
    pre-computed numpy arrays for nodes and edges.
    This class just packages them into HeteroData format.
    """

    def __init__(self, config: dict):
        self.device = config.get("device", "cpu")

    def build(self, obs: dict, device: Optional[str] = None) -> "HeteroData":
        """
        Builds a single HeteroData graph from one env's observation.

        Args:
            obs:    Agent 2 observation dict from mfg_env._build_agent2_obs()
            device: PyTorch device string ('cpu' or 'cuda')

        Returns:
            PyG HeteroData with node types: 'op', 'machine', 'job'
            and edge types: ('op','to','machine'), ('machine','to','job'),
                            ('op','to','job')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torch-geometric required for GraphBuilder")

        dev = device or self.device
        data = HeteroData()

        # --- Node features ---
        data["op"].x      = torch.tensor(obs["op_features"],      dtype=torch.float32).to(dev)
        data["machine"].x = torch.tensor(obs["machine_features"], dtype=torch.float32).to(dev)
        data["job"].x     = torch.tensor(obs["job_features"],     dtype=torch.float32).to(dev)

        # --- Edges: Op -> Machine ---
        ei = obs["edge_op_mach"]     # [2, n_edges]
        ea = obs["edge_attr_op_mach"] # [n_edges, 2]
        if ei.shape[1] > 0:
            data["op", "to", "machine"].edge_index = torch.tensor(
                ei, dtype=torch.long
            ).to(dev)
            data["op", "to", "machine"].edge_attr = torch.tensor(
                ea, dtype=torch.float32
            ).to(dev)
        else:
            # Empty edges — PyG requires at least empty tensors
            data["op", "to", "machine"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            ).to(dev)
            data["op", "to", "machine"].edge_attr = torch.zeros(
                (0, 2), dtype=torch.float32
            ).to(dev)

        # --- Edges: Machine -> Job ---
        ei = obs["edge_mach_job"]
        ea = obs["edge_attr_mach_job"]
        if ei.shape[1] > 0:
            data["machine", "to", "job"].edge_index = torch.tensor(
                ei, dtype=torch.long
            ).to(dev)
            data["machine", "to", "job"].edge_attr = torch.tensor(
                ea, dtype=torch.float32
            ).to(dev)
        else:
            data["machine", "to", "job"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            ).to(dev)
            data["machine", "to", "job"].edge_attr = torch.zeros(
                (0, 2), dtype=torch.float32
            ).to(dev)

        # --- Edges: Op -> Job ---
        ei = obs["edge_op_job"]
        ea = obs["edge_attr_op_job"]
        if ei.shape[1] > 0:
            data["op", "to", "job"].edge_index = torch.tensor(
                ei, dtype=torch.long
            ).to(dev)
            data["op", "to", "job"].edge_attr = torch.tensor(
                ea, dtype=torch.float32
            ).to(dev)
        else:
            data["op", "to", "job"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            ).to(dev)
            data["op", "to", "job"].edge_attr = torch.zeros(
                (0, 2), dtype=torch.float32
            ).to(dev)

        return data


    def build_batch(
        self,
        obs_list: List[dict],
        device:   Optional[str] = None,
    ) -> "Batch":
        """
        Builds a batched PyG graph from multiple env observations.
        Used for parallel env training — 4 envs -> 1 batched forward pass.

        PyG's Batch.from_data_list() handles node index offsetting automatically.
        Nodes from different graphs are isolated — no cross-env message passing.

        Args:
            obs_list: List of obs dicts, one per parallel env
            device:   PyTorch device

        Returns:
            PyG Batch object (behaves like HeteroData for TGIN forward pass)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torch-geometric required")

        graphs = [self.build(obs, device) for obs in obs_list]
        return Batch.from_data_list(graphs)
