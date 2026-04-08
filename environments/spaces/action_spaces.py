from __future__ import annotations
"""
environments/spaces/action_spaces.py
=====================================
Action space definitions and masking logic for both agents.

Agent 1 actions:
    maintenance: [n_machines] each in {0=none, 1=PM, 2=CM}
    reorder:     [n_consumable] each in {0..Q_max}

Agent 2 actions:
    (job_id, op_idx, machine_id) sampled from valid pairs
    OR WAIT action (index = n_valid_pairs, always available)

Masking rules (from architecture doc):
    Agent 1 PM mask:     machine must be OP and not processing a job
    Agent 1 CM mask:     machine must be FAIL
    Agent 1 reorder mask: pipeline already covers projected need
    Agent 2 mask:         derived from post-Agent-1 machine states
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from environments.transitions.degradation import MachineState, MachineStatus
from environments.transitions.job_dynamics import Job, OpStatus
from environments.transitions.resource_dynamics import ResourceState


# Q_max: max units orderable per consumable per step
Q_MAX = 10

# Action indices for Agent 1 maintenance decisions
# NOTE: CM is no longer an agent decision — it is triggered automatically
# by the environment when a machine fails (see mfg_env._attempt_auto_cm).
ACTION_NONE = 0
ACTION_PM   = 1
# ACTION_CM = 2  ← removed from agent action space; kept only as MachineStatus value


def build_agent1_maintenance_mask(
    machine_states: List[MachineState],
    machine_busy:   List[bool],
    resource_state: ResourceState,
    rho_PM:         np.ndarray,
    rho_CM:         np.ndarray,   # kept in signature for API compatibility
    n_renewable:    int,
) -> np.ndarray:
    """
    Builds maintenance action mask for Agent 1.
    Shape: [n_machines, 2] — columns are [noop, PM].

    CM is NO LONGER an agent action.  When a machine fails the environment
    automatically initiates CM (see mfg_env._attempt_auto_cm).

    Rules:
        noop (0): always allowed
        PM   (1): machine must be OP AND idle AND resources available

    Returns:
        [n_machines, 2] bool mask  (True = allowed)
    """
    n_machines = len(machine_states)
    mask = np.zeros((n_machines, 2), dtype=bool)

    for i, s in enumerate(machine_states):
        mask[i, ACTION_NONE] = True          # noop always valid

        # PM: machine must be OP, not busy, resources ok, AND health < 75%
        # Health gate matches mfg_env._step_agent1 hard constraint.
        # Adding it here means the policy network learns "PM not available at h>75"
        # instead of outputting PM actions that get silently rejected.
        if (s.status == MachineStatus.OP
                and not machine_busy[i]
                and s.health < 75
                and resource_state.can_do_maintenance(
                    rho_PM[i, :n_renewable].astype(int),
                    rho_PM[i, n_renewable:]
                )):
            mask[i, ACTION_PM] = True

    return mask


def build_agent1_reorder_mask(
    resource_state: ResourceState,
    rho_CM_max:     np.ndarray,   # [n_consumable] max CM needs across machines
) -> np.ndarray:
    """
    Builds reorder action mask for Agent 1.
    Shape: [n_consumable] — True means reordering is ALLOWED.

    FIX (v2): Old implementation used horizon=max_lead_time (=5 steps), which
    created a safety_stock of only 10 units. Starting inventory (20,15,10) was
    above this, so the mask was False at training start and the agent could never
    order. Result: order_cost=0 for all 400k steps, consumables silently depleted,
    CM eventually blocked.

    New policy: allow ordering when (inventory + pipeline) < safety_stock.
    safety_stock = rho_CM_max * t_ep_buffer, where t_ep_buffer covers ~one full
    episode of worst-case failures. Agent is ALWAYS free to order; the w_hold
    holding cost in the reward function provides incentive not to over-order.
    This correctly separates the mask (feasibility) from the cost (optimality).

    Args:
        resource_state: Current resource state
        rho_CM_max:     Max CM consumable requirement per resource [n_consumable]

    Returns:
        [n_consumable] bool mask — True = ordering is allowed (not over-stocked)
    """
    # safety_stock: 3 episodes × ~13 units/ep consumption = 39 units per resource
    # Capped at 40 so agent can always order during the critical learning phase.
    # Holding cost (w_hold=0.005) discourages over-ordering beyond this.
    safety_stock = rho_CM_max * 20  # 2*20=40 > all initial inventories → always orderable at start

    # Current effective supply = on-hand + in-pipeline
    pipeline = resource_state.pending_orders.sum(axis=1)
    effective_supply = resource_state.consumable_inventory + pipeline

    # Allow ordering if below safety stock — agent pays w_hold to discourage excess
    return effective_supply < safety_stock


def build_agent2_valid_actions(
    jobs:           List[Job],
    machine_states: List[MachineState],
    machine_busy:   List[bool],
) -> Dict[Tuple[int, int], List[int]]:
    """
    Builds complete valid action dict for Agent 2.
    Called AFTER Agent 1 half-step so machine states are updated.

    Returns:
        Dict {(job_id, op_idx): [valid_machine_ids]}
        Empty dict -> only WAIT is valid this step
    """
    # Machine availability: OP and not busy
    avail = {
        s.machine_id: (
            s.status == MachineStatus.OP
            and not machine_busy[s.machine_id]
        )
        for s in machine_states
    }

    valid_actions = {}
    for job in jobs:
        for op in job.operations:
            if op.status != OpStatus.READY:
                continue
            valid_machines = [
                m for m in op.eligible_machines
                if avail.get(m, False)
            ]
            if valid_machines:
                valid_actions[(job.job_id, op.op_idx)] = valid_machines

    return valid_actions


def flatten_agent2_actions(
    valid_actions: Dict[Tuple[int, int], List[int]]
) -> List[Tuple[int, int, int]]:
    """
    Flattens valid action dict to list of (job_id, op_idx, machine_id) tuples.
    The index into this list is what the policy samples over.
    Last index = WAIT action.

    Returns:
        List of (j, k, m) tuples. Sample index i -> valid_pairs[i].
        len(result) = total valid (op, machine) pairs.
    """
    pairs = []
    for (job_id, op_idx), machines in valid_actions.items():
        for m in machines:
            pairs.append((job_id, op_idx, m))
    return pairs