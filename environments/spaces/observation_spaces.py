"""
environments/spaces/observation_spaces.py
==========================================
Defines observation space dimensions for both agents.

Agent 1 (PDM): flat vector
    Per-machine features (15 * n_machines)
    + resource state (n_renewable + n_consumable * (1 + max_lead_time))
    + job summary (5 aggregate stats)

Agent 2 (Job Shop): HeteroData graph
    Node feature dims are defined here; graph structure in graph_builder.py

These dims must stay consistent with:
    - MachineState.to_feature_vector()  -> 15 dims
    - ResourceState.to_flat_vector()    -> computed from config
    - Operation.to_feature_vector()     -> 10 dims
    - Job.to_feature_vector()           -> 7 dims
"""

from typing import Dict, Tuple


# Node feature dimensions for TGIN (Table 3.9 in report)
OP_FEATURE_DIM      = 10
MACHINE_FEATURE_DIM = 15
JOB_FEATURE_DIM     = 7

# Edge feature dimensions (Table 3.10 in report)
EDGE_OP_MACH_DIM    = 2   # (processing_time_norm, compatibility_score)
EDGE_MACH_JOB_DIM   = 2   # (progress_pct, est_completion_time)
EDGE_OP_JOB_DIM     = 2   # (precedence_position, is_ready_flag)


def compute_agent1_obs_dim(config: dict) -> int:
    """
    Computes flat observation dimension for Agent 1.

    Args:
        config: Full config dict

    Returns:
        Integer dimension of Agent 1's observation vector
    """
    n_machines  = len(config.get("machines", []))
    machine_dim = MACHINE_FEATURE_DIM * n_machines

    ren_cfgs = config["resources"]["renewable"]
    con_cfgs = config["resources"]["consumable"]
    n_ren    = len(ren_cfgs)
    n_con    = len(con_cfgs)
    max_lead = max(r["lead_time_shifts"] for r in con_cfgs)

    # resource_dim = n_ren (available) + n_con (inventory) + n_con*max_lead (pipeline)
    resource_dim = n_ren + n_con * (1 + max_lead)

    # Job summary: 5 aggregate stats
    # [n_jobs_active, n_jobs_at_risk, avg_completion_ratio, avg_slack, n_ready_ops]
    job_summary_dim = 5

    return machine_dim + resource_dim + job_summary_dim


def compute_agent1_action_dim(config: dict) -> Tuple[int, int]:
    """
    Returns (maintenance_action_dim, reorder_action_dim) for Agent 1.

    maintenance_action_dim: n_machines * 3 (none/PM/CM per machine)
    reorder_action_dim:     n_consumable (quantity per resource)

    Returns:
        (maintenance_dim, reorder_dim)
    """
    n_machines = len(config.get("machines", []))
    n_con      = len(config["resources"]["consumable"])
    return n_machines * 3, n_con
