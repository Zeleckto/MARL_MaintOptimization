"""
rewards/components/maintenance_reward.py
==========================================
Agent 1 (PDM) reward component.

r1_t = -sum_m(c_PM * z_PM_{m,t} + c_CM * z_CM_{m,t})   [maintenance costs]
       -sum_r(c_r * a_order_{r,t})                        [ordering cost]
       +w_avail * A_system(s_{t+1})                       [DENSE availability bonus]
       +lambda * R_shared_t                               [shared failure penalty]

The availability bonus w_avail * A_system is critical — without it,
most steps return r1~=0 and the critic learns nothing useful.
"""

import numpy as np
from typing import List
from environments.transitions.degradation import MachineState, MachineStatus


def compute_system_availability(machine_states: List[MachineState]) -> float:
    """
    Computes system availability A_system at current timestep.
    Simple definition: fraction of machines that are operational (OP status).

    More sophisticated version (eq 3.40 in report) uses Weibull reliability
    per machine and operation eligibility. Using simple version for now
    as it provides the same gradient signal direction.

    Args:
        machine_states: Post-tick machine states

    Returns:
        Float in [0, 1]. 1.0 = all machines operational.
    """
    n_op = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    return n_op / max(len(machine_states), 1)


def compute_maintenance_reward(
    maintenance_actions: List[int],   # [n_machines] 0=none,1=PM,2=CM
    ordering_cost:       float,       # from resource_dynamics.step()
    machine_states:      List[MachineState],  # post-tick states
    shared_reward:       float,
    weights:             dict,        # from reward_weights.yaml
) -> float:
    """
    Computes Agent 1's reward for this timestep.

    Args:
        maintenance_actions: Agent 1's maintenance decisions this step
        ordering_cost:       Total consumable ordering cost this step
        machine_states:      Post-tick machine states (for availability)
        shared_reward:       R_shared_t (failure penalty)
        weights:             Reward weight coefficients

    Returns:
        r1_t scalar reward for Agent 1
    """
    c_PM     = weights.get("c_PM", 1.0)
    c_CM     = weights.get("c_CM", 3.0)
    w_avail  = weights.get("w_avail", 2.0)
    lam      = weights.get("lambda_shared", 0.3)

    # Maintenance action costs
    maint_cost = 0.0
    for action in maintenance_actions:
        if action == 1:   # PM
            maint_cost += c_PM
        elif action == 2: # CM
            maint_cost += c_CM

    # System availability bonus (dense signal every step)
    avail_bonus = w_avail * compute_system_availability(machine_states)

    # Combine
    r1 = -maint_cost - ordering_cost + avail_bonus + lam * shared_reward

    return float(r1)
