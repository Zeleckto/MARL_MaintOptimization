"""
rewards/components/shared_reward.py — FINAL
=============================================
Shared signal that BOTH agents receive. Creates cooperative incentive.

R_shared = -c_fail * n_failures              [catastrophic — both agents penalized]
           +w_comp_shared * n_completions     [both agents rewarded for throughput]

DESIGN CHANGE: Added completion bonus to R_shared.
Previously R_shared was only negative (failure penalty). This meant the shared
signal only fired on catastrophic events. Adding a positive completion signal
means both agents continuously benefit from throughput:
  - Agent 1 learns: keeping machines available helps Agent 2 complete jobs → +reward
  - Agent 2 learns: routing to healthy machines avoids failures → avoids -reward

With λ=0.4:
  - Failure cost to each agent:    0.4 × (-25) = -10 per failure
  - Completion reward to each:     0.4 × (+1)  = +0.4 per job completed
  - Over an episode with 20 completions: +8 from completions, -20 per failure
"""
from typing import List


def compute_shared_reward(
    newly_failed_machine_ids: List[int],
    c_fail: float = 25.0,
    n_completions: int = 0,
    w_comp_shared: float = 1.0,
) -> float:
    """
    R_shared = -c_fail * n_failures + w_comp_shared * n_completions
    """
    failure_penalty = -c_fail * len(newly_failed_machine_ids)
    completion_bonus = w_comp_shared * n_completions
    return failure_penalty + completion_bonus


def compute_machine_criticality(newly_failed, eligible_map, n_pending_ops):
    """Stub for compatibility — criticality weighting disabled."""
    return {m: 0.0 for m in newly_failed}