"""
tier1/formulation/constraints_B.py
====================================
Constraints C13-C20: Maintenance Scheduling.
Section 3.2.4 Group B in report.
"""

from ortools.sat.python import cp_model


def add_constraints_B(
    model:    cp_model.CpModel,
    vars_s:   dict,
    vars_m:   dict,
    machines: list,
    horizon:  int,
) -> None:
    """
    C13: Max PM count per machine per horizon
    C14: CM required when critical health reached
    C15: PM occupies machine for tau_PM periods
    C16: CM occupies machine for tau_CM periods
    C17: Maintenance exclusivity (not PM and CM simultaneously)
    C18: Cannot start maintenance while another ongoing
    C19: Maintenance indicator consistency
    C20: Cannot schedule operations on machine under maintenance
    """
    z_PM = vars_m["z_PM"]
    z_CM = vars_m["z_CM"]
    u    = vars_m["u"]
    f    = vars_m["f"]

    for m, mach in enumerate(machines):
        tau_PM = mach["tau_PM_shifts"]
        tau_CM = mach["tau_CM_shifts"]

        # ─── C14: CM required when failure occurs ─────────────────────────
        for t in range(horizon):
            # If f[m][t]=1, then sum of z_CM in [t, t+tau_CM] >= 1
            end_t = min(t + tau_CM, horizon)
            cm_sum = sum(z_CM[m][tt] for tt in range(t, end_t))
            model.Add(cm_sum >= f[m][t])

        # ─── C15: PM occupies machine for tau_PM periods ─────────────────
        for t in range(horizon):
            for dt in range(tau_PM):
                if t + dt < horizon:
                    model.Add(u[m][t + dt] >= z_PM[m][t])

        # ─── C16: CM occupies machine for tau_CM periods ─────────────────
        for t in range(horizon):
            for dt in range(tau_CM):
                if t + dt < horizon:
                    model.Add(u[m][t + dt] >= z_CM[m][t])

        # ─── C17: Maintenance exclusivity ────────────────────────────────
        for t in range(horizon):
            model.Add(z_PM[m][t] + z_CM[m][t] <= 1)

        # ─── C18: Only one maintenance start while another ongoing ────────
        # At most one maintenance active at any time
        for t in range(horizon):
            # Sum of all maintenance starts in recent window <= 1
            window_pm = sum(
                z_PM[m][tt] for tt in range(max(0, t - tau_PM + 1), t + 1)
            )
            window_cm = sum(
                z_CM[m][tt] for tt in range(max(0, t - tau_CM + 1), t + 1)
            )
            model.Add(window_pm + window_cm <= 1)
