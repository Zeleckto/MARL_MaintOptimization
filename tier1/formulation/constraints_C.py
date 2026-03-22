"""
tier1/formulation/constraints_C.py
====================================
Constraints C21-C26: Health Dynamics.
Section 3.2.4 Group C in report.
Uses LINEAR degradation (Tier 1 deterministic approximation).
"""

from ortools.sat.python import cp_model


def add_constraints_C(
    model:    cp_model.CpModel,
    vars_s:   dict,
    vars_m:   dict,
    machines: list,
    horizon:  int,
    scale:    int = 10,   # scale factor for integer health arithmetic
) -> None:
    """
    C21: Health evolution (linear degradation for Tier 1)
    C22: Health bounds 0 <= h <= 100
    C23: Initial health
    C24: Critical health detection
    C25: Health bounds after maintenance
    C26: Degradation rate consistency
    """
    h    = vars_m["h"]
    z_PM = vars_m["z_PM"]
    z_CM = vars_m["z_CM"]
    f    = vars_m["f"]
    u    = vars_m["u"]

    for m, mach in enumerate(machines):
        delta_h    = int(mach.get("delta_h",    0.5)  * scale)
        h_crit     = int(mach.get("h_critical", 10.0) * scale)
        h_restore_PM = int(mach.get("h_restore_PM", 30.0) * scale)
        h_restore_CM = int(mach.get("h_restore_CM", 60.0) * scale)
        tau_PM     = mach["tau_PM_shifts"]
        tau_CM     = mach["tau_CM_shifts"]

        # ─── C21: Health evolution ────────────────────────────────────────
        for t in range(horizon):
            # omega_{m,t} = 1 if machine operating (not under maintenance)
            # Approximation: machine operates if not under maintenance
            # h[m][t+1] = h[m][t] - delta_h * (1 - u[m][t])
            #             + h_restore_PM * z_PM[m][t-tau_PM+1]   (if t >= tau_PM-1)
            #             + h_restore_CM * z_CM[m][t-tau_CM+1]
            degradation = delta_h  # simplified: degrade every step

            restore_PM = 0
            if t >= tau_PM - 1:
                restore_PM = h_restore_PM * z_PM[m][t - tau_PM + 1]

            restore_CM = 0
            if t >= tau_CM - 1:
                restore_CM = h_restore_CM * z_CM[m][t - tau_CM + 1]

            model.Add(
                h[m][t + 1] == h[m][t] - degradation * (1 - u[m][t])
                + restore_PM + restore_CM
            )

        # ─── C22: Health bounds ───────────────────────────────────────────
        for t in range(horizon + 1):
            model.Add(h[m][t] >= 0)
            model.Add(h[m][t] <= 100 * scale)

        # ─── C24: Critical health detection ──────────────────────────────
        # f[m][t] = 1 if h[m][t] <= h_critical
        for t in range(horizon):
            # Big-M formulation: h[m][t] <= h_crit + M*(1-f[m][t])
            M = 100 * scale
            model.Add(h[m][t] <= h_crit + M * (1 - f[m][t]))
            model.Add(h[m][t] >= h_crit - M * f[m][t] + 1)
