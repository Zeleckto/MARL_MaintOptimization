from __future__ import annotations
"""
rewards/components/maintenance_reward.py
==========================================
Agent 1 (PDM) reward — aligned with report objective eq. 3.14.

r1_t = −c_PM·Σzᴾᴹₘ − c_CM·Σzᶜᴹₘ       [maintenance action costs]
       −δ · Σcᵣ·Qᵣ                       [resource ordering cost, δ-weighted]
       +w_avail · A_system(s_{t+1})       [DENSE availability bonus]
       +w_RUL · RUL_bonus(s_{t+1})        [DENSE RUL preservation bonus — NEW]
       +λ · R_shared_t                    [shared failure penalty]

RUL bonus: rewards Agent 1 for keeping machines in the useful-life regime.
Computed as mean(RUL_m / η_m) across machines — normalised remaining life fraction.
Fires every step; small but consistent signal toward proactive maintenance.
"""

import numpy as np
from typing import List
from environments.transitions.degradation import MachineState, MachineStatus


def compute_system_availability(machine_states: List[MachineState]) -> float:
    """Fraction of machines in OP status."""
    n_op = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    return n_op / max(len(machine_states), 1)


def compute_rul_bonus(
    machine_states: List[MachineState],
    eta_values:     List[float],     # characteristic life per machine
    rul_threshold:  float = 0.3,     # below this fraction of eta, no bonus
) -> float:
    """
    Mean normalised RUL across operational machines.
    Returns value in [0, 1].

    Only counts OP machines — machines in PM/CM have no RUL risk right now.
    If RUL < rul_threshold * eta, that machine contributes 0 (already degraded).

    This directly encodes Weibull reliability:
    high RUL fraction = machines in useful-life phase (low hazard rate)
    low RUL fraction  = machines approaching wear-out (high hazard rate)
    """
    rul_fracs = []
    for s, eta in zip(machine_states, eta_values):
        if s.status != MachineStatus.OP:
            continue
        frac = s.rul / max(eta, 1.0)
        if frac >= rul_threshold:
            rul_fracs.append(min(frac, 1.0))
        else:
            rul_fracs.append(0.0)

    return float(np.mean(rul_fracs)) if rul_fracs else 0.0


def compute_maintenance_reward(
    maintenance_actions: List[int],    # [n_machines] 0=none,1=PM,2=CM
    ordering_cost:       float,        # raw ordering cost from resource_dynamics
    machine_states:      List[MachineState],
    eta_values:          List[float],  # characteristic life per machine
    shared_reward:       float,
    weights:             dict,
) -> float:
    """
    Computes Agent 1's reward — aligned with eq. 3.14 objective.

    Args:
        maintenance_actions: Agent 1 maintenance decisions this step
        ordering_cost:       Total consumable ordering cost (before δ weighting)
        machine_states:      Post-tick machine states
        eta_values:          Weibull η per machine (for RUL normalisation)
        shared_reward:       R_shared_t (criticality-weighted failure penalty)
        weights:             Reward weight coefficients from reward_weights.yaml

    Returns:
        r1_t scalar
    """
    c_PM        = weights.get("c_PM", 1.0)
    c_CM        = weights.get("c_CM", 3.0)
    delta_obj   = weights.get("delta_obj", 0.5)
    w_avail     = weights.get("w_avail", 2.0)
    w_RUL       = weights.get("w_RUL", 0.5)
    rul_thresh  = weights.get("rul_threshold", 0.3)
    lam         = weights.get("lambda_shared", 0.3)

    # Maintenance action costs
    maint_cost = 0.0
    for action in maintenance_actions:
        if action == 1:    maint_cost += c_PM
        elif action == 2:  maint_cost += c_CM

    # Resource ordering cost (δ-weighted per eq. 3.14)
    resource_cost = delta_obj * ordering_cost

    # Dense availability bonus (every step)
    avail_bonus = w_avail * compute_system_availability(machine_states)

    # Dense RUL preservation bonus (every step) — NEW
    # Ties reward to Weibull reliability: preserve RUL = stay in useful-life phase
    rul_bonus = w_RUL * compute_rul_bonus(machine_states, eta_values, rul_thresh)

    r1 = -maint_cost - resource_cost + avail_bonus + rul_bonus + lam * shared_reward
    return float(r1)