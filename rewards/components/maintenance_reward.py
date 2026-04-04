"""
rewards/components/maintenance_reward.py
==========================================
Agent 1 reward: maintenance costs + ΔRUL dense signal + availability + holding cost.

KEY REDESIGN (Phase 2):
    OLD: w_avail * A_system_level  (step-level availability — not dense enough)
    NEW: w_RUL * ΔRUL_fleet  (per-step RUL change across all machines)

    ΔRUL fires every step with magnitude proportional to how much the
    fleet's remaining useful life changed. This gives Agent 1 a continuous
    gradient signal without needing a failure event.

    Added: w_hold * sum(inventory) — holding cost discourages over-ordering.

r1_t = -c_PM * n_PM
       - c_CM * n_CM
       - delta * ordering_cost
       + w_RUL * ΔRUL_fleet
       + w_avail * (fraction of OP machines)
       - w_hold * sum(consumable_inventory)
       + lambda * R_shared
"""

from typing import List, Optional
import numpy as np

from environments.transitions.degradation import MachineState, MachineStatus, estimate_rul


def compute_maintenance_reward(
    maintenance_actions:  List[int],         # [n_mach] 0=none, 1=PM, 2=CM
    ordering_cost:        float,             # total cost of orders placed
    machine_states:       List[MachineState],
    prev_ruls:            Optional[List[float]],  # RUL before this step
    consumable_inventory: Optional[List[float]],  # current inventory levels
    shared_reward:        float,
    weights:              dict,
) -> float:
    """
    Computes Agent 1's reward for one timestep.

    Args:
        maintenance_actions:  Actions taken this step
        ordering_cost:        Cost of reorder actions
        machine_states:       Post-tick machine states
        prev_ruls:            RUL values before this step (for ΔRUL)
        consumable_inventory: Current inventory [n_consumable]
        shared_reward:        R_shared_t (failure penalty)
        weights:              Reward weight dict from reward_weights.yaml

    Returns:
        r1 scalar
    """
    c_PM    = weights.get("c_PM",    1.0)
    c_CM    = weights.get("c_CM",    7.0)
    delta   = weights.get("delta",   0.5)
    w_RUL   = weights.get("w_RUL",   0.05)
    w_avail = weights.get("w_avail", 2.0)
    w_hold  = weights.get("w_hold",  0.005)
    lam     = weights.get("lambda_shared", 0.3)

    # --- Maintenance action costs ---
    n_PM = sum(1 for a in maintenance_actions if a == 1)
    n_CM = sum(1 for a in maintenance_actions if a == 2)
    maint_cost = -(c_PM * n_PM + c_CM * n_CM)

    # --- Ordering cost ---
    order_cost = -delta * ordering_cost

    # --- ΔRUL dense signal ---
    # Positive when PM/CM happened (RUL increased)
    # Negative when machines aged without maintenance
    if prev_ruls is not None and len(prev_ruls) == len(machine_states):
        curr_ruls = [estimate_rul(s) for s in machine_states]
        delta_rul = sum(
            curr - prev
            for curr, prev in zip(curr_ruls, prev_ruls)
        )
        rul_reward = w_RUL * delta_rul
    else:
        rul_reward = 0.0

    # --- Availability signal (fraction of OP machines) ---
    n_op     = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    avail_r  = w_avail * (n_op / max(len(machine_states), 1))

    # --- Holding cost ---
    if consumable_inventory is not None:
        hold_cost = -w_hold * sum(max(inv, 0.0) for inv in consumable_inventory)
    else:
        hold_cost = 0.0

    r1 = maint_cost + order_cost + rul_reward + avail_r + hold_cost + lam * shared_reward

    return float(r1)
