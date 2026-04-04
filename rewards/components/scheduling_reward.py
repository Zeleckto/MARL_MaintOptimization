"""
rewards/components/scheduling_reward.py
==========================================
Agent 2 reward: tardiness + completion + slack + health-aware dispatch.

KEY REDESIGN (Phase 2):
    OLD: -w_tard * total_tardiness / T_max  (sparse — only fires at completion)
    NEW: -w_tard * delta_tardiness_risk      (dense projected tardiness)
         + w_slack * n_on_track / J          (fraction of jobs with positive slack)

    delta_tardiness_risk = change in (tardiness + projected_excess) this step.
    This fires every step based on how at-risk the schedule looks.

r2_t = -w_tard * sum_j(w_j * max(0, C_j - d_j)) / T_max  [completed jobs]
       - w_tard * 0.1 * projected_tardiness                 [at-risk jobs, dense]
       + w_slack * n_on_track / J                           [DENSE slack signal]
       + w_comp * n_completions_this_step                   [DENSE completion]
       + w_health * health_assigned_machine / 100           [DENSE health bonus]
       + lambda * R_shared
"""

from typing import List, Optional, Tuple
import numpy as np

from environments.transitions.job_dynamics import Job, OpStatus
from environments.transitions.degradation import MachineState


def compute_scheduling_reward(
    jobs:              List[Job],
    completed_job_ids: List[int],
    assignment:        Optional[Tuple[int, int, int]],  # (job_id, op_idx, machine_id) or None
    machine_states:    List[MachineState],
    shared_reward:     float,
    current_time:      float,
    t_max:             int,
    weights:           dict,
) -> float:
    """
    Computes Agent 2's reward for one timestep.

    Args:
        jobs:              All active jobs (including completed)
        completed_job_ids: Jobs finished this step
        assignment:        Agent 2's action — (j,k,m) or None (WAIT)
        machine_states:    Post-tick states (for health bonus)
        shared_reward:     R_shared_t
        current_time:      Current timestep
        t_max:             Episode length
        weights:           Reward weight dict

    Returns:
        r2 scalar
    """
    w_tard   = weights.get("w_tard",   5.0)
    w_comp   = weights.get("w_comp",   3.0)
    w_health = weights.get("w_health", 0.5)
    w_slack  = weights.get("w_slack",  0.2)
    lam      = weights.get("lambda_shared", 0.3)

    # --- Actual tardiness (completed jobs only) ---
    total_tard = sum(
        job.weight * job.tardiness
        for job in jobs
        if job.completion_time is not None
    )
    tard_penalty = -w_tard * total_tard / max(t_max, 1)

    # --- Projected tardiness (at-risk active jobs) ---
    # Dense signal: penalise jobs whose remaining work > remaining time
    projected = 0.0
    active_jobs = [j for j in jobs if not j.is_complete]
    n_jobs = max(len(active_jobs), 1)
    n_on_track = 0

    for job in active_jobs:
        remaining_work = sum(
            min(op.nominal_proc_times.values()) if op.nominal_proc_times else 2.0
            for op in job.operations
            if op.status in (OpStatus.PENDING, OpStatus.READY, OpStatus.IN_PROGRESS)
        )
        time_left = job.due_date - current_time
        excess = max(remaining_work - time_left, 0.0)
        projected += job.weight * excess / max(t_max, 1)

        if excess <= 0:
            n_on_track += 1

    projected_penalty = -w_tard * 0.1 * projected

    # --- Slack signal: fraction of jobs currently on track ---
    slack_reward = w_slack * (n_on_track / n_jobs)

    # --- Completion bonus ---
    comp_bonus = w_comp * len(completed_job_ids)

    # --- Health-aware dispatch bonus ---
    health_bonus = 0.0
    if assignment is not None:
        _, _, machine_id = assignment
        if machine_id < len(machine_states):
            health_bonus = w_health * (machine_states[machine_id].health / 100.0)

    r2 = tard_penalty + projected_penalty + slack_reward + comp_bonus + health_bonus + lam * shared_reward

    return float(r2)
