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
    Returns value in [0, 1]. Gracefully handles missing rul attribute (v1).

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
        rul = getattr(s, "rul", None)
        if rul is None:
            # rul removed in v1: use hazard_rate as proxy, or skip
            hr = getattr(s, "hazard_rate", None)
            if hr is not None:
                peak_hr = s.beta / max(eta, 1.0)
                frac = max(0.0, 1.0 - hr / max(peak_hr, 1e-12))
            else:
                continue  # nothing to compute
        else:
            frac = rul / max(eta, 1.0)
        if frac >= rul_threshold:
            rul_fracs.append(min(frac, 1.0))
        else:
            rul_fracs.append(0.0)

    return float(np.mean(rul_fracs)) if rul_fracs else 0.0


def compute_maintenance_reward(
    maintenance_actions: List[int],       # [n_machines] 0=none,1=PM (CM removed)
    ordering_cost:       float,
    machine_states:      List[MachineState],
    eta_values:          List[float],
    shared_reward:       float,
    weights:             dict,
    inventory_total:     float = 0.0,
    delta_ruls:          List[float] = None,
    n_auto_cm:           int = 0,         # machines auto-CM initiated this step
) -> float:
    """
    Agent 1 reward — design doc v2 eq. 3.14 + Section 6.2 DELTA-RUL signal.

    r1 = -c_PM*n_PM - c_CM*n_CM          [maintenance action costs]
         - delta_obj * ordering_cost      [resource ordering]
         - w_hold * inventory_total       [holding cost, EOQ theory §7.5]
         + w_RUL * mean(DELTA_RUL)        [DENSE: change in RUL fleet §6.2]
         + w_avail * delta_availability   [DENSE: change in system availability]
         + lambda * R_shared              [shared failure penalty]

    DELTA_RUL per machine per step (design doc §6.2):
        PM this step:    DELTA_RUL = RUL_after - RUL_before ~ +30-40 shifts (positive)
        Normal operate:  DELTA_RUL = -1  (one shift of life consumed)
        Failed:          DELTA_RUL = large negative (life collapses to ~0)

    This is causally linked to Agent 1's actions: PM fires a positive DELTA_RUL
    spike every step of maintenance, giving immediate dense credit.
    """
    c_PM       = weights.get("c_PM",        1.0)
    c_CM       = weights.get("c_CM",        7.0)
    delta_obj  = weights.get("delta_obj",   0.5)
    w_avail    = weights.get("w_avail",     0.5)
    w_RUL      = weights.get("w_RUL",       0.05)
    lam        = weights.get("lambda_shared", 0.3)
    w_hold     = weights.get("w_hold",      0.005)
    # w_fail_idle removed — CM is now auto-handled by environment
    # c_CM still charges when auto-CM is triggered (real operational cost)

    # ── Maintenance action costs (agent only decides noop/PM now) ────────
    maint_cost = sum(c_PM if a == 1 else 0
                     for a in maintenance_actions)

    # ── Resource ordering cost ──────────────────────────────────────────
    resource_cost = delta_obj * ordering_cost

    # ── Inventory holding cost (EOQ theory §7.5) ────────────────────────
    holding_cost = w_hold * inventory_total

    # ── Auto-CM cost: charge when environment initiates CM on a failed machine
    # This makes failure costly without requiring the agent to decide on CM.
    # c_CM fires once per auto-CM event (matches real cost: parts, technicians)
    auto_cm_cost = c_CM * n_auto_cm

    # ── DELTA-RUL fleet signal (design doc §6.2) ────────────────────────
    # mean(ΔRUL_m) across OP machines — fires positive on PM, -1 normally
    if delta_ruls is not None and len(delta_ruls) > 0:
        op_deltas = [
            dr for dr, s in zip(delta_ruls, machine_states)
            if s.status == MachineStatus.OP
        ]
        delta_rul_signal = float(np.mean(op_deltas)) if op_deltas else 0.0
    else:
        # Fallback: use RUL level fraction if delta not available
        delta_rul_signal = compute_rul_bonus(machine_states, eta_values, 0.3)

    rul_bonus = w_RUL * delta_rul_signal

    # ── Availability change bonus ───────────────────────────────────────
    # w_avail rewards MAINTAINING high availability (level, per design doc)
    avail_bonus = w_avail * compute_system_availability(machine_states)

    r1 = (-maint_cost - resource_cost - holding_cost - auto_cm_cost
          + rul_bonus + avail_bonus + lam * shared_reward)
    return float(r1)