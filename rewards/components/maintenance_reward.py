"""
rewards/components/maintenance_reward.py — FINAL
=================================================
Agent 1 (PDM) reward. Clean, economically-grounded signals only.

r1 = +pm_bonus(health-gated)        [PM incentive, breaks even at h≈80%]
     -c_PM * n_PM                    [PM costs money]
     -c_CM * n_auto_cm               [CM costs money]
     +w_RUL * SUM(ΔRUL)             [fleet life preservation — fires at PM completion]
     -delta_obj * ordering_cost      [ordering costs money]
     -w_hold * inventory_total       [holding costs money]
     +λ * R_shared                   [failure penalty + completion reward]

REMOVED (caused reward hacking in previous versions):
  w_avail  — punished PM (availability drops when machine enters PM)
  w_hazard — redundant with ΔRUL, added noise to critic
  w_stockout — catch-22 (PM depletes inventory → stockout punishes PM)
  w_reorder_bonus — incentivised ordering for its own sake
"""
from typing import List, Optional
import numpy as np
from environments.transitions.degradation import MachineState, MachineStatus


def compute_system_availability(machine_states) -> float:
    n_op = sum(1 for s in machine_states if s.status == MachineStatus.OP)
    return n_op / max(len(machine_states), 1)


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
    pre_maint_health:    list = None,
    # Accept and ignore v0 params for compatibility
    units_ordered:       float = 0.0,
    inv_below_rop:       bool = False,
) -> float:
    c_PM       = weights.get("c_PM", 1.0)
    c_CM       = weights.get("c_CM", 7.0)
    delta_obj  = weights.get("delta_obj", 0.05)
    w_RUL      = weights.get("w_RUL", 0.15)
    lam        = weights.get("lambda_shared", 0.4)
    w_hold     = weights.get("w_hold", 0.005)
    w_pm_bonus = weights.get("w_pm_bonus", 5.0)

    # ── PM costs ────────────────────────────────────────────────────────
    n_PM = sum(1 if a == 1 else 0 for a in maintenance_actions)
    maint_cost = n_PM * c_PM
    auto_cm_cost = c_CM * n_auto_cm

    # ── Health-gated PM bonus (uses pre_maint_health to avoid h=100 reset) ──
    # pm_bonus = w_pm_bonus × (1 - h/100) per PM.
    # h=100: bonus=0, net=-1 → noop preferred
    # h=80:  bonus=1, net=0  → breakeven
    # h=70:  bonus=1.5, net=+0.5 → PM preferred
    pm_bonus = 0.0
    if w_pm_bonus > 0 and n_PM > 0:
        health_src = pre_maint_health if pre_maint_health is not None else [
            s.health for s in machine_states]
        for i, a in enumerate(maintenance_actions):
            if a == 1 and i < len(health_src):
                urgency = max(0.0, 1.0 - health_src[i] / 100.0)
                pm_bonus += w_pm_bonus * urgency

    # ── ΔRUL fleet signal (SUM, not mean) ───────────────────────────────
    # PM completion: one machine jumps +30 → sum = +26 (net of 4 others at -1)
    # Normal step: all -1 → sum = -5
    # Failure: one machine drops → sum = -large
    rul_bonus = 0.0
    if delta_ruls is not None and w_RUL > 0:
        rul_bonus = w_RUL * sum(delta_ruls)

    # ── Ordering + holding ──────────────────────────────────────────────
    resource_cost = delta_obj * ordering_cost
    holding_cost = w_hold * inventory_total

    # ── Assemble ────────────────────────────────────────────────────────
    r1 = (pm_bonus + rul_bonus
          - maint_cost - auto_cm_cost - resource_cost - holding_cost
          + lam * shared_reward)
    return float(r1)