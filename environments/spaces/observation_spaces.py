"""
environments/spaces/observation_spaces.py
===========================================
Observation space dimension constants.
All observation arrays are built from these constants — single source of truth.

CHANGE LOG:
    Phase 0: MACHINE_FEATURE_DIM 15->15 (same count but features changed)
             H is now Weibull survival, hazard_rate is feature [4]
    Phase 1: OP_FEATURE_DIM unchanged at 10
             JOB_FEATURE_DIM unchanged at 7
"""

# Node feature dimensions (must match to_feature_vector() implementations)
MACHINE_FEATURE_DIM = 15   # degradation.py MachineState.to_feature_vector()
OP_FEATURE_DIM      = 10   # job_dynamics.py Operation.to_feature_vector()
JOB_FEATURE_DIM     = 7    # job_dynamics.py Job.to_feature_vector()

# Agent 1 flat observation structure:
#   [machine_features (15 * n_machines),
#    resource_renewable (n_renewable * 2),      [available, capacity]
#    resource_consumable (n_consumable * 2),     [inventory, pipeline_sum]
#    resource_pipeline (n_consumable * max_lead_time),
#    job_summary (6 stats)]
JOB_SUMMARY_DIM = 6


def compute_agent1_obs_dim(config_or_n_machines, n_renewable=None, n_consumable=None, max_lead_time=None) -> int:
    """
    Computes Agent 1's flat observation vector dimension.

    Accepts either:
        compute_agent1_obs_dim(config_dict)                          # preferred
        compute_agent1_obs_dim(n_machines, n_renewable, n_consumable, max_lead_time)  # legacy

    Returns:
        Total observation dimension
    """
    # Accept config dict (new calling convention from mfg_env.py)
    # OR accept 4 explicit ints (legacy calling convention from mlp_policy.py)
    if isinstance(config_or_n_machines, dict):
        config = config_or_n_machines
        n_machines   = len(config.get("machines", []))
        res_cfg      = config.get("resources", {})
        n_renewable  = len(res_cfg.get("renewable", []))
        n_consumable = len(res_cfg.get("consumable", []))
        max_lead_time = max(
            (r.get("lead_time_shifts", 5) for r in res_cfg.get("consumable", [])),
            default=7
        )
    else:
        n_machines = config_or_n_machines
        # n_renewable, n_consumable, max_lead_time passed explicitly

    machine_dim  = MACHINE_FEATURE_DIM * n_machines
    resource_ren = n_renewable * 2
    resource_con = n_consumable * 2
    pipeline_dim = n_consumable * max_lead_time
    job_dim      = JOB_SUMMARY_DIM

    return machine_dim + resource_ren + resource_con + pipeline_dim + job_dim