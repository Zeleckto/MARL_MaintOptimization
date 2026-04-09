"""
rewards/components/scheduling_reward.py — v5
=============================================
Agent 2 (Job Shop) reward.

r2 = +w_assign    per assignment made           [DENSE: fires 80/150 steps]
     -w_wait      per unnecessary wait          [DENSE: penalizes inaction]
     +w_comp      per job completed             [SPARSE: ~20/150 steps]
     -w_tard      per late job (incremental)    [SPARSE: ~10/150 steps]
     +w_health    per healthy machine dispatch  [DENSE: steers machine choice]
     +λ * R_shared                              [cooperation]

KEY INSIGHT: Agent 2's main problem was that WAIT and ASSIGN had nearly
identical per-step reward (0.0 vs +0.39). With MC returns over 150 steps,
this 0.39 gap was invisible. Adding w_assign=0.5 and w_wait=-0.3 creates
a 1.19 gap that's 3x more learnable.
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
    n_valid_pairs:       int = 0,
) -> float:
    alpha    = weights.get("alpha", 1.0)
    w_tard   = weights.get("w_tard", 8.0)
    w_comp   = weights.get("w_comp", 5.0)
    w_health = weights.get("w_health", 0.5)
    w_assign = weights.get("w_assign", 0.5)
    w_wait   = weights.get("w_wait", 0.3)
    lam      = weights.get("lambda_shared", 0.4)

    completed_set = set(completed_job_ids)

    # ── INCREMENTAL tardiness ─────────────────────────────────────────
    new_tardiness = 0.0
    for j in jobs:
        if j.job_id in completed_set and j.completion_time is not None:
            new_tardiness += j.weight * max(0.0, j.completion_time - j.due_date)
    tard_penalty = -alpha * w_tard * new_tardiness / max(t_max, 1)

    # ── Completion bonus ──────────────────────────────────────────────
    comp_bonus = w_comp * len(completed_job_ids)

    # ── Assignment bonus (DENSE — fires EVERY step agent assigns) ─────
    # This is the key signal: "assigning work is better than waiting"
    assign_bonus = 0.0
    wait_penalty = 0.0
    if assignment is not None:
        assign_bonus = w_assign
    elif n_valid_pairs > 0:
        # Agent chose to WAIT when there were valid options
        wait_penalty = -w_wait

    # ── Health-aware dispatch ─────────────────────────────────────────
    health_bonus = 0.0
    if assignment is not None:
        _, _, machine_id = assignment
        if machine_id < len(machine_states):
            health_bonus = w_health * (machine_states[machine_id].health / 100.0)

    r2 = (tard_penalty + comp_bonus + assign_bonus + wait_penalty 
          + health_bonus + lam * shared_reward)
    return float(r2)