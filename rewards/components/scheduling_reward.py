"""
rewards/components/scheduling_reward.py
=========================================
Agent 2 (Job Shop) reward component.

r2_t = -w_tard * sum_j(w_j * max(0, C_j - d_j)) / T_max  [normalised tardiness]
       +w_comp * |{j: completed this step}|                 [DENSE completion bonus]
       +w_health * mean(h_m) over assigned machines          [DENSE health-aware bonus]
       +lambda * R_shared_t                                  [shared failure penalty]

The completion bonus and health bonus are dense — they fire every step
and prevent the critic value function from collapsing to near-zero.
"""

import numpy as np
from typing import List, Tuple
from environments.transitions.job_dynamics import Job
from environments.transitions.degradation import MachineState


def compute_scheduling_reward(
    jobs:                List[Job],
    completed_job_ids:   List[int],    # jobs completed THIS step
    assignment:          Tuple[int, int, int],  # (job_id, op_idx, machine_id) or None
    machine_states:      List[MachineState],
    shared_reward:       float,
    t_max:               int,
    weights:             dict,
) -> float:
    """
    Computes Agent 2's reward for this timestep.

    Args:
        jobs:              All active jobs
        completed_job_ids: Jobs that completed this step (for completion bonus)
        assignment:        Agent 2's action this step — (j,k,m) or None (WAIT)
        machine_states:    Post-tick machine states (for health bonus)
        shared_reward:     R_shared_t
        t_max:             Episode length (for normalising tardiness)
        weights:           Reward weight coefficients

    Returns:
        r2_t scalar reward for Agent 2
    """
    w_tard   = weights.get("w_tard", 5.0)
    w_comp   = weights.get("w_comp", 3.0)
    w_health = weights.get("w_health", 0.5)
    lam      = weights.get("lambda_shared", 0.3)

    # Normalised weighted tardiness across all completed jobs
    total_tardiness = sum(
        job.weight * job.tardiness
        for job in jobs
        if job.completion_time is not None
    )
    tard_penalty = -w_tard * total_tardiness / max(t_max, 1)

    # Completion bonus: reward for each job finished this step
    comp_bonus = w_comp * len(completed_job_ids)

    # Health-aware dispatch bonus: reward for assigning to healthy machines
    # Encourages Agent 2 to cooperate with Agent 1 by preferring healthy machines
    health_bonus = 0.0
    if assignment is not None:
        _, _, machine_id = assignment
        if machine_id < len(machine_states):
            health_norm = machine_states[machine_id].health / 100.0
            health_bonus = w_health * health_norm

    r2 = tard_penalty + comp_bonus + health_bonus + lam * shared_reward

    return float(r2)
