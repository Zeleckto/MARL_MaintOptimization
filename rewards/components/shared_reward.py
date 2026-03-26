from __future__ import annotations
"""
rewards/components/shared_reward.py
=====================================
Shared failure penalty — applied to BOTH agents.

Now criticality-weighted: bottleneck machine failure costs more.
Machine criticality = fraction of pending ops that ONLY that machine can do.
A machine with criticality=1 is the sole machine for all pending ops → failure
is catastrophic and should be penalised far more than a redundant machine.

R_shared_t = −c_fail · Σ_{m failed} (1 + crit_mult · criticality_m)
"""

from typing import List, Dict


def compute_machine_criticality(
    newly_failed:  List[int],
    eligible_map:  Dict[int, List[int]],   # machine_id -> list of ops that need it
    n_pending_ops: int,
) -> Dict[int, float]:
    """
    Computes criticality score for each newly failed machine.

    criticality_m = |ops where m is the ONLY eligible machine| / n_pending_ops

    Args:
        newly_failed:  Machine IDs that failed this step
        eligible_map:  {machine_id: [op_idx, ...]} — ops that can run on each machine
        n_pending_ops: Total pending ops right now

    Returns:
        Dict {machine_id: criticality_score in [0,1]}
    """
    if not newly_failed or n_pending_ops == 0:
        return {m: 0.0 for m in newly_failed}

    criticality = {}
    for m in newly_failed:
        # Count ops where m is the only eligible machine
        bottleneck_ops = sum(
            1 for machine_ops in eligible_map.values()
            if m in machine_ops
        )
        # Rough proxy — full computation requires iterating all ops
        criticality[m] = min(bottleneck_ops / max(n_pending_ops, 1), 1.0)

    return criticality


def compute_shared_reward(
    newly_failed_machine_ids: List[int],
    c_fail:                   float,
    criticality_multiplier:   float = 5.0,
    machine_criticality:      Dict[int, float] = None,
) -> float:
    """
    Criticality-weighted failure penalty.

    Penalty per machine = c_fail × (1 + crit_mult × criticality_m)
    → A pure bottleneck machine (crit=1) costs 6× a redundant machine.

    Args:
        newly_failed_machine_ids: Machines that transitioned to FAIL this step
        c_fail:                   Base failure penalty coefficient
        criticality_multiplier:   How much more bottleneck failures cost
        machine_criticality:      {machine_id: criticality} — None means all = 0

    Returns:
        Negative penalty value (0.0 if no failures)
    """
    if not newly_failed_machine_ids:
        return 0.0

    machine_criticality = machine_criticality or {}
    total_penalty = 0.0
    for m in newly_failed_machine_ids:
        crit = machine_criticality.get(m, 0.0)
        weight = 1.0 + criticality_multiplier * crit
        total_penalty += c_fail * weight

    return -total_penalty