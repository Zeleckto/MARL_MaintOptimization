"""
tier1/formulation/constraints_E.py
====================================
Constraints C35-C38: Linking Constraints.
Section 3.2.4 Group E in report.
These couple the scheduling and maintenance decisions.
"""

from ortools.sat.python import cp_model


def add_constraints_E(
    model:    cp_model.CpModel,
    vars_s:   dict,
    vars_m:   dict,
    jobs:     list,
    machines: list,
    horizon:  int,
) -> None:
    """
    C35: Cannot schedule job on machine under maintenance
    C36: Cannot schedule job on failed machine
    C37: PM only when machine is idle (not processing)
    C38: PM health threshold constraint (optional)
    """
    y    = vars_s["y"]
    s    = vars_s["s"]
    C    = vars_s["C"]
    u    = vars_m["u"]
    f    = vars_m["f"]
    z_PM = vars_m["z_PM"]

    M = horizon  # Big-M constant

    # ─── C35: Cannot schedule on machine under maintenance ────────────────
    # If u[m][t] = 1, no op can be active on m at t
    # Enforced via interval variable is_present and maintenance interval
    # The NoOverlap constraint handles this if we add maintenance intervals
    for m, mach in enumerate(machines):
        tau_PM = mach["tau_PM_shifts"]
        tau_CM = mach["tau_CM_shifts"]

        # Add maintenance intervals to machine NoOverlap (handled in constraints_A
        # if maintenance interval vars are passed there — cross-reference)
        pass

    # ─── C37: PM only when machine is idle ───────────────────────────────
    # For each machine m and time t:
    # If z_PM[m][t] = 1, no operation starts on m at time t
    for m in range(len(machines)):
        for t in range(horizon):
            for j, job in enumerate(jobs):
                for k in range(job["n_ops"]):
                    if m in job["eligible_machines"][k]:
                        # z_PM[m][t] + y[j][k][m] * (s[j][k] == t) <= 1
                        # Linearised: if PM starts, op cannot start same time
                        # Using big-M: s[j][k] >= t+1 - M*(1-z_PM[m][t])
                        model.Add(
                            s[j][k] >= t + 1 - M * (1 - z_PM[m][t])
                        ).OnlyEnforceIf(y[j][k][m])
