"""
rewards/components/maintenance_reward.py
=========================================
Agent 1 (PDM) reward — r1.

r1 = pm_bonus - c_PM*n_PM - c_CM*n_CM       # maintenance incentives
   - delta_obj*ordering_cost                  # ordering cost (small)
   - w_hold*inventory_total                   # holding cost
   + w_reorder_bonus*units_ordered_below_rop  # ordering incentive
   - w_hazard*mean_hazard_rate                # continuous health signal
   + w_RUL*sum(delta_RUL)                     # RUL preservation
   + w_avail*availability                     # dense availability
   + lam*shared_reward                        # shared failure penalty

DESIGN NOTE (v3):
  w_stockout removed. It created a catch-22: PM depleted inventory,
  stockout penalised inventory depletion, so PM caused huge penalties.
  The agent correctly learned to never PM. Fix: use reorder_bonus instead
  (pull toward ordering) rather than stockout (push away from depletion).
"""
from typing import List, Optional
import numpy as np
from environments.transitions.degradation import MachineState


def compute_maintenance_reward(
    maintenance_actions: List[int],
    ordering_cost:       float,
    machine_states:      List[MachineState],
    eta_values:          List[float],
    shared_reward:       float,
    weights:             dict,
    inventory_total:     float = 0.0,
    delta_ruls:          List[float] = None,
    n_auto_cm:           int = 0,
    units_ordered:       float = 0.0,
    inv_below_rop:       bool = False,
    pre_maint_health:    list = None,   # health BEFORE tick_all (avoids health=100 reset bug)
) -> float:
    """Compute Agent 1 reward for one timestep."""
    c_PM         = weights.get("c_PM",           1.0)
    c_CM         = weights.get("c_CM",           7.0)
    delta_obj    = weights.get("delta_obj",       0.05)
    w_avail      = weights.get("w_avail",         0.5)
    w_RUL        = weights.get("w_RUL",           0.15)
    lam          = weights.get("lambda_shared",   0.3)
    w_hold       = weights.get("w_hold",          0.005)
    w_pm_bonus   = weights.get("w_pm_bonus",      3.0)
    w_hazard     = weights.get("w_hazard",        0.5)
    w_reorder_b  = weights.get("w_reorder_bonus", 0.5)

    # -- Maintenance costs --------------------------------------------------
    n_PM       = sum(1 if a == 1 else 0 for a in maintenance_actions)
    maint_cost = n_PM * c_PM

    # -- PM initiation bonus: health-conditional to prevent PM spam -----------
    # pm_bonus = w_pm_bonus × (1 - health/100) per PM taken.
    # At health=100: bonus=0 < c_PM=1 → noop preferred (no spam).
    # At health=80:  bonus=1.0 = c_PM → breakeven.
    # At health<80:  bonus > c_PM → PM strictly preferred.
    # This gives the agent a natural, learned PM threshold around h=80.
    pm_bonus = 0.0
    if w_pm_bonus > 0 and machine_states is not None:
        # Use pre_maint_health (health before tick_all) because tick_all resets
        # health=100 at PM initiation (Weibull formula with time_since_maint=0).
        # Without this, pm_bonus always fires at urgency=0 (h=100 after tick).
        health_source = pre_maint_health if pre_maint_health is not None else [
            s.health for s in machine_states]
        for i, a in enumerate(maintenance_actions):
            if a == 1 and i < len(health_source):
                h_norm = health_source[i] / 100.0
                urgency = max(0.0, 1.0 - h_norm)
                pm_bonus += w_pm_bonus * urgency

    # -- Auto-CM cost -------------------------------------------------------
    auto_cm_cost = c_CM * n_auto_cm

    # -- Ordering -----------------------------------------------------------
    resource_cost   = delta_obj * ordering_cost
    holding_cost    = w_hold * inventory_total
    # Reorder bonus: reward ordering when inventory is below reorder point
    reorder_bonus   = w_reorder_b * units_ordered if inv_below_rop else 0.0

    # -- Hazard penalty (continuous, grows as health declines) ---------------
    w_hazard_val = w_hazard
    hazard_penalty = 0.0
    if w_hazard_val > 0 and machine_states is not None:
        mean_h = float(sum(getattr(s, "hazard_rate", 0.0)
                           for s in machine_states) / max(len(machine_states), 1))
        hazard_penalty = w_hazard_val * mean_h

    # -- RUL preservation (dense during PM window) ---------------------------
    rul_bonus = 0.0
    if delta_ruls is not None and w_RUL > 0:
        rul_bonus = w_RUL * sum(max(0.0, d) for d in delta_ruls)

    # -- Availability --------------------------------------------------------
    from environments.transitions.degradation import MachineStatus
    n_op = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    avail_bonus = w_avail * (n_op / max(len(machine_states), 1))

    r1 = (pm_bonus + rul_bonus + avail_bonus + reorder_bonus
          - maint_cost - resource_cost - holding_cost
          - auto_cm_cost - hazard_penalty
          + lam * shared_reward)
    return r1


def compute_system_availability(machine_states) -> float:
    """
    Fraction of machines currently in OP status.
    Used by tests and analytics.
    """
    from environments.transitions.degradation import MachineStatus
    n_op = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    return n_op / max(len(machine_states), 1)