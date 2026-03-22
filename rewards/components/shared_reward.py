"""
rewards/components/shared_reward.py
=====================================
Shared failure penalty — applied to BOTH agents.

Both agents are jointly responsible for machine failures:
    - Agent 1 for not doing PM early enough
    - Agent 2 for overloading degraded machines

R_shared_t = -c_fail * number_of_failures_this_step
"""

from typing import List
from environments.transitions.degradation import MachineStatus


def compute_shared_reward(
    newly_failed_machine_ids: List[int],
    c_fail: float,
) -> float:
    """
    Computes shared failure penalty for this timestep.

    Args:
        newly_failed_machine_ids: Machines that transitioned to FAIL this step
        c_fail: Failure penalty coefficient (from reward_weights.yaml)

    Returns:
        Negative penalty value (0.0 if no failures)
    """
    n_failures = len(newly_failed_machine_ids)
    return -c_fail * n_failures
