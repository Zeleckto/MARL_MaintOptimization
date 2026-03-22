"""
benchmarks/metrics.py
======================
Shared metrics computation for Tier 1 vs Tier 3 comparison.
Both solvers produce output in the same format so this file
can evaluate either without modification.
"""

import numpy as np
from typing import List, Dict
from environments.transitions.job_dynamics import Job
from environments.transitions.degradation import MachineState, MachineStatus


def compute_makespan(jobs: List[Job]) -> float:
    """Max completion time across all jobs."""
    times = [j.completion_time for j in jobs if j.completion_time is not None]
    return max(times) if times else 0.0


def compute_weighted_tardiness(jobs: List[Job]) -> float:
    """Sum of w_j * max(0, C_j - d_j)."""
    return sum(
        j.weight * max(0.0, (j.completion_time or 0.0) - j.due_date)
        for j in jobs
    )


def compute_system_availability(
    machine_states: List[MachineState],
    episode_length: int,
) -> float:
    """
    Fraction of (machine, timestep) pairs where machine was operational.
    Approximated from final state cumulative_op_time.
    """
    total_possible = len(machine_states) * episode_length * 8.0  # hours
    total_op = sum(s.cumulative_op_time for s in machine_states)
    return total_op / max(total_possible, 1.0)


def compute_mtbf(failure_times: List[float]) -> float:
    """Mean Time Between Failures from list of failure timestamps."""
    if len(failure_times) < 2:
        return float("inf")
    gaps = np.diff(sorted(failure_times))
    return float(np.mean(gaps))


def compute_total_cost(
    weighted_tardiness: float,
    n_PM:               int,
    n_CM:               int,
    n_failures:         int,
    ordering_cost:      float,
    weights:            Dict,
) -> float:
    """Total cost combining all components (eq 3.14 in report)."""
    return (
        weights.get("alpha",    1.0) * weighted_tardiness
      + weights.get("gamma_obj",1.0) * (
            weights.get("c_PM", 1.0) * n_PM
          + weights.get("c_CM", 3.0) * n_CM
          + weights.get("c_fail",30.0) * n_failures
        )
      + weights.get("delta_obj",0.0) * ordering_cost
    )


def summarise_episode(
    jobs:           List[Job],
    machine_states: List[MachineState],
    n_failures:     int,
    n_PM:           int,
    n_CM:           int,
    ordering_cost:  float,
    episode_length: int,
    weights:        Dict,
) -> Dict:
    """
    Computes all benchmark metrics for one episode.
    Returns dict suitable for printing or CSV export.
    """
    wt   = compute_weighted_tardiness(jobs)
    ms   = compute_makespan(jobs)
    avail = compute_system_availability(machine_states, episode_length)
    cost  = compute_total_cost(wt, n_PM, n_CM, n_failures, ordering_cost, weights)

    n_completed = sum(1 for j in jobs if j.is_complete)

    return {
        "makespan":           ms,
        "weighted_tardiness": wt,
        "system_availability": avail,
        "n_failures":         n_failures,
        "n_PM":               n_PM,
        "n_CM":               n_CM,
        "n_jobs_completed":   n_completed,
        "ordering_cost":      ordering_cost,
        "total_cost":         cost,
    }