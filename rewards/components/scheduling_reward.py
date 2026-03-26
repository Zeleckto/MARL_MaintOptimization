from __future__ import annotations
"""
rewards/components/scheduling_reward.py
=========================================
Agent 2 (Job Shop) reward — aligned with report objective eq. 3.14.

r2_t = −α · Σwⱼ·max(0, Cⱼ−dⱼ) / T_max   [normalised weighted tardiness]
       −β · Cmax_est / T_max               [running makespan estimate — NEW]
       +w_comp · completions_this_step      [DENSE completion bonus]
       +w_health · health_assigned_mach     [DENSE health-aware dispatch bonus]
       +λ · R_shared_t                     [shared failure penalty]

Makespan estimate: rather than waiting for episode end (sparse),
we use estimated_Cmax = max over active jobs of (estimated_completion_time).
Small β keeps this from dominating tardiness.
"""

import numpy as np
from typing import List, Tuple, Optional
from environments.transitions.job_dynamics import Job
from environments.transitions.degradation import MachineState


def estimate_makespan(
    jobs:         List[Job],
    current_step: int,
) -> float:
    """
    Running estimate of Cmax — maximum expected completion time.

    For completed jobs: use actual completion_time.
    For active jobs: estimate = current_step + remaining_ops * avg_proc_time.
    For pending jobs not started: estimate = due_date (conservative).

    Returns normalised estimate (raw step count, caller divides by T_max).
    """
    max_completion = 0.0

    for job in jobs:
        if job.completion_time is not None:
            # Already done — use actual
            max_completion = max(max_completion, job.completion_time)
        else:
            # Estimate remaining time
            n_remaining = sum(1 for op in job.operations if op.status in (0, 1, 2))
            # Rough average: 1 op takes ~1-2 shifts
            est_done = current_step + n_remaining * 1.5
            max_completion = max(max_completion, est_done)

    return float(max_completion)


def compute_scheduling_reward(
    jobs:                List[Job],
    completed_job_ids:   List[int],
    assignment:          Optional[Tuple[int, int, int]],
    machine_states:      List[MachineState],
    shared_reward:       float,
    t_max:               int,
    current_step:        int,
    weights:             dict,
) -> float:
    """
    Computes Agent 2's reward — aligned with eq. 3.14.

    Args:
        jobs:              All jobs (active + done)
        completed_job_ids: Jobs completed THIS step (for completion bonus)
        assignment:        Agent 2's action — (job_id, op_idx, machine_id) or None
        machine_states:    Post-tick machine states
        shared_reward:     R_shared_t (criticality-weighted)
        t_max:             Episode length for normalisation
        current_step:      Current timestep (for makespan estimate)
        weights:           From reward_weights.yaml

    Returns:
        r2_t scalar
    """
    alpha         = weights.get("alpha", 1.0)
    w_tard        = weights.get("w_tard", 5.0)
    beta_makespan = weights.get("beta_makespan", 0.2)
    w_comp        = weights.get("w_comp", 3.0)
    w_health      = weights.get("w_health", 0.5)
    lam           = weights.get("lambda_shared", 0.3)

    # ── Normalised weighted tardiness (α-weighted, eq. 3.14) ──────────────
    # Only count jobs that have already completed — running jobs don't have
    # final tardiness yet. Prevents penalising jobs still in progress.
    completed_jobs = [j for j in jobs if j.completion_time is not None]
    total_tardiness = sum(
        j.weight * max(0.0, j.completion_time - j.due_date)
        for j in completed_jobs
    )
    tard_penalty = -alpha * w_tard * total_tardiness / max(t_max, 1)

    # ── Running makespan estimate penalty (β-weighted, eq. 3.14) ──────────
    # Small β so it doesn't dominate tardiness — guides Agent 2 toward
    # minimising global span not just individual tardiness.
    cmax_est     = estimate_makespan(jobs, current_step)
    make_penalty = -beta_makespan * cmax_est / max(t_max, 1)

    # ── Completion bonus (dense — fires when a job finishes this step) ─────
    comp_bonus = w_comp * len(completed_job_ids)

    # ── Health-aware dispatch bonus (dense — fires on every assignment) ────
    # Rewards Agent 2 for preferring healthy machines → cooperative with Agent 1.
    health_bonus = 0.0
    if assignment is not None:
        _, _, machine_id = assignment
        if machine_id < len(machine_states):
            health_norm  = machine_states[machine_id].health / 100.0
            health_bonus = w_health * health_norm

    r2 = tard_penalty + make_penalty + comp_bonus + health_bonus + lam * shared_reward
    return float(r2)