"""
rewards/components/scheduling_reward.py — FINAL
================================================
Agent 2 (Job Shop) reward.

r2 = -α * w_tard * NEW_tardiness / T   [incremental: one-shot per late job]
     +w_comp * n_completed_this_step    [dense completion bonus]
     +w_health * health_of_machine      [health-aware dispatch]
     +λ * R_shared                      [failure penalty + completion reward]

FIX: Tardiness is INCREMENTAL, not cumulative.
Old code summed ALL completed jobs' tardiness EVERY step → grew to -300/ep.
Now each job's tardiness is charged exactly once, when it completes.

REMOVED:
  makespan estimate — noisy, dominated by tardiness
  slack signal — noisy, rare to fire, low information
"""
import numpy as np
from typing import List, Tuple, Optional
from environments.transitions.job_dynamics import Job
from environments.transitions.degradation import MachineState


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
    alpha    = weights.get("alpha", 1.0)
    w_tard   = weights.get("w_tard", 5.0)
    w_comp   = weights.get("w_comp", 3.0)
    w_health = weights.get("w_health", 0.5)
    lam      = weights.get("lambda_shared", 0.4)

    completed_set = set(completed_job_ids)

    # ── INCREMENTAL tardiness (one-shot per late job) ─────────────────────
    new_tardiness = 0.0
    for j in jobs:
        if j.job_id in completed_set and j.completion_time is not None:
            new_tardiness += j.weight * max(0.0, j.completion_time - j.due_date)
    tard_penalty = -alpha * w_tard * new_tardiness / max(t_max, 1)

    # ── Completion bonus ──────────────────────────────────────────────────
    comp_bonus = w_comp * len(completed_job_ids)

    # ── Health-aware dispatch ─────────────────────────────────────────────
    health_bonus = 0.0
    if assignment is not None:
        _, _, machine_id = assignment
        if machine_id < len(machine_states):
            health_bonus = w_health * (machine_states[machine_id].health / 100.0)

    r2 = tard_penalty + comp_bonus + health_bonus + lam * shared_reward
    return float(r2)