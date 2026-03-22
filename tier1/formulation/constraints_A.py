"""
tier1/formulation/constraints_A.py
====================================
Constraints C1-C12: Job Shop Scheduling.
Section 3.2.4 Group A in report.
"""

from ortools.sat.python import cp_model


def add_constraints_A(
    model:   cp_model.CpModel,
    vars_s:  dict,    # scheduling variables
    jobs:    list,
    horizon: int,
) -> None:
    """
    Adds all Group A constraints to the CP-SAT model.

    C1:  Each op assigned to exactly one eligible machine
    C2:  (handled by IntervalVar is_present logic)
    C3:  (handled by IntervalVar is_present logic)
    C4:  (handled by IntervalVar definition)
    C5:  Completion time definition
    C6:  Job completion = last op completion
    C7:  Tardiness definition
    C8:  Makespan definition
    C9:  Release time constraint
    C10: Operation precedence within job
    C11-C12: Disjunctive no-overlap constraints per machine
    """
    y        = vars_s["y"]
    s        = vars_s["s"]
    C        = vars_s["C"]
    T_tard   = vars_s["T"]
    C_max    = vars_s["C_max"]
    intervals = vars_s["intervals"]

    # ─── C1: Each operation assigned to exactly one eligible machine ───────
    for j, job in enumerate(jobs):
        for k in range(job["n_ops"]):
            eligible = job["eligible_machines"][k]
            model.AddExactlyOne([y[j][k][m] for m in eligible])

    # ─── C5: Completion time definition ───────────────────────────────────
    for j, job in enumerate(jobs):
        for k in range(job["n_ops"]):
            eligible = job["eligible_machines"][k]
            # C[j][k] = s[j][k] + sum(proc_time[m] * y[j][k][m])
            proc_expr = sum(
                job["proc_times"][k][m] * y[j][k][m]
                for m in eligible
            )
            model.Add(C[j][k] == s[j][k] + proc_expr)

    # ─── C7: Tardiness T[j] = max(0, C[j][n_j] - d_j) ──────────────────
    for j, job in enumerate(jobs):
        last_k = job["n_ops"] - 1
        d_j    = job["due_date"]
        model.AddMaxEquality(T_tard[j], [0, C[j][last_k] - d_j])

    # ─── C8: Makespan C_max >= C[j][last_op] for all j ───────────────────
    for j, job in enumerate(jobs):
        last_k = job["n_ops"] - 1
        model.Add(C_max >= C[j][last_k])

    # ─── C9: Release time s[j][0] >= r_j ─────────────────────────────────
    for j, job in enumerate(jobs):
        r_j = job.get("release_time", 0)
        model.Add(s[j][0] >= r_j)

    # ─── C10: Operation precedence s[j][k+1] >= C[j][k] ─────────────────
    for j, job in enumerate(jobs):
        for k in range(job["n_ops"] - 1):
            model.Add(s[j][k + 1] >= C[j][k])

    # ─── C11-C12: No-overlap on each machine (disjunctive constraint) ─────
    # Group all interval vars by machine
    machine_intervals = {}
    for j, job in enumerate(jobs):
        for k in range(job["n_ops"]):
            for m in job["eligible_machines"][k]:
                if m not in machine_intervals:
                    machine_intervals[m] = []
                machine_intervals[m].append(intervals[j][k][m])

    for m, ivs in machine_intervals.items():
        model.AddNoOverlap(ivs)
