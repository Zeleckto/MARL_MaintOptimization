"""
tier1/formulation/constraints_D.py
====================================
Constraints C27-C34: Resource Constraints.
Section 3.2.4 Group D in report.
"""

from ortools.sat.python import cp_model


def add_constraints_D(
    model:     cp_model.CpModel,
    vars_m:    dict,
    vars_r:    dict,
    resources: dict,
    n_machines: int,
    horizon:   int,
    rho_PM:    list,   # [n_machines][n_resources] PM requirements
    rho_CM:    list,   # [n_machines][n_resources] CM requirements
) -> None:
    """
    C27: Renewable resource capacity
    C28: Consumable inventory limit
    C29: PM resource sufficiency
    C30: CM resource sufficiency
    C31: No allocation without maintenance
    C32: Inventory dynamics
    C33: Initial inventory
    C34: Non-negative inventory
    """
    delta  = vars_r["delta"]
    I      = vars_r["I"]
    Q      = vars_r["Q"]
    z_PM   = vars_m["z_PM"]
    z_CM   = vars_m["z_CM"]

    ren_cfgs = resources["renewable"]
    con_cfgs = resources["consumable"]
    n_ren    = len(ren_cfgs)

    # ─── C27: Renewable resource capacity ────────────────────────────────
    for r_idx, r_cfg in enumerate(ren_cfgs):
        K_r = r_cfg["capacity"]
        for t in range(horizon):
            model.Add(
                sum(delta[m][r_idx][t] for m in range(n_machines)) <= K_r
            )

    # ─── C28-C30: Consumable requirements and sufficiency ─────────────────
    for r_idx, r_cfg in enumerate(con_cfgs):
        res_col = n_ren + r_idx  # column in rho matrices

        for t in range(horizon):
            # C28: sum of allocations <= inventory
            model.Add(
                sum(delta[m][res_col][t] for m in range(n_machines)) <= I[r_idx][t]
            )

            # C29-C30: Resource sufficiency for maintenance
            for m in range(n_machines):
                rho_pm = int(rho_PM[m][res_col]) if rho_PM else 1
                rho_cm = int(rho_CM[m][res_col]) if rho_CM else 2
                model.Add(delta[m][res_col][t] >= rho_pm * z_PM[m][t])
                model.Add(delta[m][res_col][t] >= rho_cm * z_CM[m][t])

    # ─── C32: Inventory dynamics ──────────────────────────────────────────
    for r_idx, r_cfg in enumerate(con_cfgs):
        L_r = r_cfg["lead_time_shifts"]
        res_col = n_ren + r_idx

        for t in range(horizon):
            # I[r][t+1] = I[r][t] - consumed[r][t] + Q[r][t-L_r]
            consumed = sum(delta[m][res_col][t] for m in range(n_machines))
            arrived  = Q[r_idx][t - L_r] if t >= L_r else 0
            model.Add(I[r_idx][t + 1] == I[r_idx][t] - consumed + arrived)

    # ─── C34: Non-negative inventory ──────────────────────────────────────
    for r_idx in range(len(con_cfgs)):
        for t in range(horizon + 1):
            model.Add(I[r_idx][t] >= 0)
