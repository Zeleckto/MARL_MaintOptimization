"""
rewards/components/shared_reward.py
======================================
Shared failure penalty: both agents receive this for every machine failure.

DESIGN DECISION: No criticality weighting.
    Old design had a per-failure severity multiplier.
    Removed because: (a) TGIN learns bottlenecks implicitly, (b) criticality
    weights are hard to calibrate without domain data, (c) the multiplier
    interacts with the cost ratio c_PM:c_CM:c_fail in non-obvious ways.

R_shared = -c_fail * n_failures_this_step
"""

from typing import List


def compute_shared_reward(
    newly_failed_machine_ids: List[int],
    c_fail: float = 25.0,
) -> float:
    """
    Computes the shared failure penalty.

    Args:
        newly_failed_machine_ids: Machines that failed this step
        c_fail:                   Cost per failure (from reward_weights.yaml)

    Returns:
        R_shared scalar (always <= 0)
    """
    return -c_fail * len(newly_failed_machine_ids)
