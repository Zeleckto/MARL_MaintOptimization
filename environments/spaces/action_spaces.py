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

# Action indices for maintenance
ACTION_NONE = 0
ACTION_PM   = 1
ACTION_CM   = 2


def build_agent1_maintenance_mask(
    machine_states: List[MachineState],
    machine_busy:   List[bool],         # True if machine processing a job
    resource_state: ResourceState,
    rho_PM:         np.ndarray,         # [n_machines, n_ren+n_con]
    rho_CM:         np.ndarray,
    n_renewable:    int,
) -> np.ndarray:
    """
    Builds maintenance action mask for Agent 1.
    Shape: [n_machines, 3] — True means action is ALLOWED.

    Rules:
        PM (action=1): machine must be OP AND idle AND resources available
        CM (action=2): machine must be FAIL AND resources available
        NONE (action=0): always allowed

    Args:
        machine_states: Current machine states
        machine_busy:   [n_machines] True if machine currently processing op
        resource_state: Current resource state
        rho_PM:         Resource requirements for PM [n_mach, n_ren+n_con]
        rho_CM:         Resource requirements for CM [n_mach, n_ren+n_con]
        n_renewable:    Number of renewable resources

    Returns:
        [n_machines, 3] bool mask
    """
    n_machines = len(machine_states)
    mask = np.zeros((n_machines, 3), dtype=bool)

    for i, s in enumerate(machine_states):
        # NONE always allowed
        mask[i, ACTION_NONE] = True

        # PM: must be OP, idle, and have resources
        if (s.status == MachineStatus.OP
                and not machine_busy[i]
                and resource_state.can_do_maintenance(
                    rho_PM[i, :n_renewable].astype(int),
                    rho_PM[i, n_renewable:]
                )):
            mask[i, ACTION_PM] = True

        # CM: must be FAIL and have resources
        if (s.status == MachineStatus.FAIL
                and resource_state.can_do_maintenance(
                    rho_CM[i, :n_renewable].astype(int),
                    rho_CM[i, n_renewable:]
                )):
            mask[i, ACTION_CM] = True

    return mask


def build_agent1_reorder_mask(
    resource_state: ResourceState,
    rho_CM_max:     np.ndarray,   # [n_consumable] max CM needs across machines
) -> np.ndarray:
    """
    Builds reorder action mask for Agent 1.
    Shape: [n_consumable] — True means reordering is ALLOWED.

    Blocks reorder if pending pipeline already covers projected need.
    Prevents over-ordering reward hacking.

    Args:
        resource_state: Current resource state (contains pending_orders)
        rho_CM_max:     Max CM consumable requirement per resource

    Returns:
        [n_consumable] bool mask
    """
    from environments.transitions.resource_dynamics import ResourceManager
    # Use projected need calculation from ResourceState
    net_need = resource_state.projected_consumable_need(
        horizon=resource_state.max_lead_time,
        rho_CM=rho_CM_max,
    )
    return net_need > 0.0


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
