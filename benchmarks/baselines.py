"""
benchmarks/baselines.py
========================
Four comparison baselines for the manufacturing MARL environment.

Baselines:
  1. Reactive + FCFS       — CM only, FIFO scheduling, order when empty
  2. Rule-Based EDF        — PM when H<45, EDF scheduling, safety stock
  3. Fixed-Interval PM     — block replacement, SPT, periodic order
  4. ABR + MDD + (Q,R)     — analytically optimal per subproblem independently
                             (KEY comparison: beats independent OR optima)

Baseline 4 uses:
  - ABR (Age-Based Replacement): t* = argmin C(t)/Lambda(t) via scipy
  - MDD (Modified Due Date): priority = max(d_j, now + p_j)
  - (Q,R) inventory policy: EOQ quantity, safety stock reorder point

All baselines implement the same interface:
    agent1_action(env) -> dict with 'maintenance' and 'reorder' keys
    agent2_action(env) -> int (index into valid_pairs, or len for WAIT)
"""

import numpy as np
from typing import List, Optional
import math

from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
from environments.transitions.degradation import MachineStatus
from environments.transitions.job_dynamics import OpStatus

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# BASE INTERFACE
# ---------------------------------------------------------------------------
class BaselinePolicy:
    """Base class for all baselines."""
    name: str = "Baseline"

    def agent1_action(self, env: ManufacturingEnv) -> dict:
        raise NotImplementedError

    def agent2_action(self, env: ManufacturingEnv) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BASELINE 1: Reactive + FCFS
# Lower bound — CM only, FIFO, order when empty
# ---------------------------------------------------------------------------
class ReactiveBaseline(BaselinePolicy):
    """
    Reactive maintenance: only do CM when machine fails.
    Scheduling: FIFO (first-in-first-out by job_id).
    Ordering: only when inventory hits zero.
    """
    name = "Reactive + FCFS"

    def agent1_action(self, env: ManufacturingEnv) -> dict:
        maint = []
        for s in env.machine_states:
            if s.status == MachineStatus.FAIL:
                maint.append(2)  # CM
            else:
                maint.append(0)  # nothing

        # Order only when any consumable hits zero
        n_con = env._n_consumable
        reorder = np.zeros(n_con, dtype=float)
        if env.resource_state is not None:
            for i in range(n_con):
                if env.resource_state.consumable_inventory[i] <= 0:
                    con_cfg = env.config.get("resources", {}).get("consumable", [])
                    if i < len(con_cfg):
                        reorder[i] = float(con_cfg[i].get("reorder_qty", 8))

        return {"maintenance": np.array(maint, dtype=int), "reorder": reorder}

    def agent2_action(self, env: ManufacturingEnv) -> int:
        """FIFO: assign lowest job_id READY op."""
        pairs = env._valid_pairs
        if not pairs:
            return 0  # WAIT
        # Sort by job_id ascending (FIFO)
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        j, k, m = sorted_pairs[0]
        for i, (pj, pk, pm) in enumerate(pairs):
            if pj == j and pk == k and pm == m:
                return i
        return len(pairs)  # WAIT


# ---------------------------------------------------------------------------
# BASELINE 2: Rule-Based EDF
# PM when H<45, EDF scheduling, safety stock ordering
# ---------------------------------------------------------------------------
class RuleBasedEDFBaseline(BaselinePolicy):
    """
    PM triggered when health drops below 45%.
    Scheduling: Earliest Due Date (EDF).
    Ordering: reorder when inventory < ROP.
    """
    name = "Rule-Based EDF"
    PM_THRESHOLD = 45.0

    def agent1_action(self, env: ManufacturingEnv) -> dict:
        maint = []
        for s in env.machine_states:
            if s.status == MachineStatus.FAIL:
                maint.append(2)
            elif (s.status == MachineStatus.OP
                  and not env.machine_busy[s.machine_id]
                  and s.health < self.PM_THRESHOLD):
                maint.append(1)
            else:
                maint.append(0)

        n_con = env._n_consumable
        reorder = np.zeros(n_con, dtype=float)
        if env.resource_state is not None:
            con_cfgs = env.config.get("resources", {}).get("consumable", [])
            for i in range(min(n_con, len(con_cfgs))):
                rop = float(con_cfgs[i].get("reorder_point", 5))
                if env.resource_state.consumable_inventory[i] <= rop:
                    reorder[i] = float(con_cfgs[i].get("reorder_qty", 8))

        return {"maintenance": np.array(maint, dtype=int), "reorder": reorder}

    def agent2_action(self, env: ManufacturingEnv) -> int:
        """EDF: earliest due date first."""
        pairs = env._valid_pairs
        if not pairs:
            return 0

        job_map = {j.job_id: j for j in env.jobs}
        def edf_key(pair):
            j, k, m = pair
            job = job_map.get(j)
            return job.due_date if job else float("inf")

        sorted_pairs = sorted(pairs, key=edf_key)
        j, k, m = sorted_pairs[0]
        for i, (pj, pk, pm) in enumerate(pairs):
            if pj == j and pk == k and pm == m:
                return i
        return len(pairs)


# ---------------------------------------------------------------------------
# BASELINE 3: Fixed-Interval PM + SPT
# Block replacement at fixed interval, SPT scheduling, periodic order
# ---------------------------------------------------------------------------
class FixedIntervalSPTBaseline(BaselinePolicy):
    """
    Fixed-interval preventive maintenance (block replacement).
    Interval = eta * 0.6 for each machine type.
    Scheduling: Shortest Processing Time (SPT).
    Ordering: fixed-period (every 20 shifts).
    """
    name = "Fixed-Interval PM + SPT"

    def __init__(self):
        self._last_pm_times: dict = {}

    def reset(self):
        self._last_pm_times = {}

    def agent1_action(self, env: ManufacturingEnv) -> dict:
        maint = []
        t = env.current_step

        for s in env.machine_states:
            last_pm = self._last_pm_times.get(s.machine_id, 0)
            interval = int(s.eta * 0.6)

            if s.status == MachineStatus.FAIL:
                maint.append(2)
            elif (s.status == MachineStatus.OP
                  and not env.machine_busy[s.machine_id]
                  and (t - last_pm) >= interval):
                maint.append(1)
                self._last_pm_times[s.machine_id] = t
            else:
                maint.append(0)

        n_con = env._n_consumable
        reorder = np.zeros(n_con, dtype=float)
        # Periodic ordering every 20 shifts
        if t % 20 == 0 and env.resource_state is not None:
            con_cfgs = env.config.get("resources", {}).get("consumable", [])
            for i in range(min(n_con, len(con_cfgs))):
                reorder[i] = float(con_cfgs[i].get("reorder_qty", 8))

        return {"maintenance": np.array(maint, dtype=int), "reorder": reorder}

    def agent2_action(self, env: ManufacturingEnv) -> int:
        """SPT: shortest processing time first."""
        pairs = env._valid_pairs
        if not pairs:
            return 0

        job_map = {j.job_id: j for j in env.jobs}
        def spt_key(pair):
            j, k, m = pair
            job = job_map.get(j)
            if job and k < len(job.operations):
                op = job.operations[k]
                return op.nominal_proc_times.get(m, float("inf"))
            return float("inf")

        sorted_pairs = sorted(pairs, key=spt_key)
        j, k, m = sorted_pairs[0]
        for i, (pj, pk, pm) in enumerate(pairs):
            if pj == j and pk == k and pm == m:
                return i
        return len(pairs)


# ---------------------------------------------------------------------------
# BASELINE 4: ABR + MDD + (Q,R)
# Analytically optimal per subproblem independently.
# KEY comparison: MARL should beat this because it solves jointly.
# ---------------------------------------------------------------------------
class ABRMDDQRBaseline(BaselinePolicy):
    """
    ABR: Age-Based Replacement — optimal PM time per machine.
         t* = argmin C(t) / Lambda(t)
         C(t) = c_PM + (c_CM - c_PM) * P(fail before t)
         Lambda(t) = E[renewal length] ≈ t * (1 - P_f) + (t + tau_CM) * P_f

    MDD: Modified Due Date scheduling priority.
         priority(j) = max(d_j, now + p_j)
         Lower priority = schedule first.

    (Q,R): EOQ quantity, reorder at safety stock point.
    """
    name = "ABR + MDD + (Q,R)"

    def __init__(self):
        self._abr_intervals: dict = {}   # machine_id -> optimal interval

    def reset(self):
        self._abr_intervals = {}

    def _compute_abr_interval(
        self, beta: float, eta: float,
        c_PM: float, c_CM: float, tau_PM: int, tau_CM: int
    ) -> float:
        """
        Solves ABR: t* = argmin expected_cost_per_unit_time.
        Uses scipy.optimize.minimize_scalar on cost_rate(t).
        """
        if not SCIPY_AVAILABLE:
            # Fallback: 60% of characteristic life
            return eta * 0.6

        def weibull_cdf(t):
            return 1.0 - math.exp(-(t / eta) ** beta)

        def cost_rate(t):
            if t <= 0:
                return float("inf")
            p_fail = weibull_cdf(t)
            p_surv = 1.0 - p_fail
            # Expected cost in one cycle
            expected_cost = c_PM * p_surv + c_CM * p_fail
            # Expected cycle length
            expected_len = t * p_surv + (t + tau_CM) * p_fail + tau_PM * p_surv
            return expected_cost / max(expected_len, 1e-6)

        result = minimize_scalar(cost_rate, bounds=(1.0, eta * 3.0), method="bounded")
        return max(result.x, 1.0)

    def agent1_action(self, env: ManufacturingEnv) -> dict:
        t = env.current_step

        # Load cost ratios from reward_weights.yaml (or defaults)
        import os, yaml
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "rewards", "reward_weights.yaml"
        )
        try:
            with open(weights_path) as f:
                w = yaml.safe_load(f)
        except Exception:
            w = {}
        c_PM = w.get("c_PM", 1.0)
        c_CM = w.get("c_CM", 7.0)

        maint = []
        for s in env.machine_states:
            mid = s.machine_id

            # Compute ABR interval on first step for each machine
            if mid not in self._abr_intervals:
                self._abr_intervals[mid] = self._compute_abr_interval(
                    s.beta, s.eta, c_PM, c_CM, s.tau_PM_shifts, s.tau_CM_shifts
                )

            t_star = self._abr_intervals[mid]

            if s.status == MachineStatus.FAIL:
                maint.append(2)  # CM
            elif (s.status == MachineStatus.OP
                  and not env.machine_busy[s.machine_id]
                  and s.time_since_maint >= t_star):
                maint.append(1)  # PM — ABR triggered
            else:
                maint.append(0)

        # (Q,R) inventory policy
        n_con = env._n_consumable
        reorder = np.zeros(n_con, dtype=float)
        if env.resource_state is not None:
            con_cfgs = env.config.get("resources", {}).get("consumable", [])
            for i in range(min(n_con, len(con_cfgs))):
                rop = float(con_cfgs[i].get("reorder_point", 5))
                qty = float(con_cfgs[i].get("reorder_qty", 8))
                # Check if position (inventory + pipeline) < ROP
                inv = env.resource_state.consumable_inventory[i]
                pipeline = env.resource_state.pending_orders[i].sum()
                if (inv + pipeline) <= rop:
                    reorder[i] = qty

        return {"maintenance": np.array(maint, dtype=int), "reorder": reorder}

    def agent2_action(self, env: ManufacturingEnv) -> int:
        """
        MDD: Modified Due Date priority.
        priority(j,k) = max(d_j, now + min_proc_time_of_op_k)
        Schedule the pair with the smallest priority.
        """
        pairs = env._valid_pairs
        if not pairs:
            return 0

        t       = float(env.current_step)
        job_map = {j.job_id: j for j in env.jobs}

        def mdd_key(pair):
            j, k, m = pair
            job = job_map.get(j)
            if job and k < len(job.operations):
                op = job.operations[k]
                p_jk = min(op.nominal_proc_times.values()) if op.nominal_proc_times else 2.0
                return max(job.due_date, t + p_jk)
            return float("inf")

        sorted_pairs = sorted(pairs, key=mdd_key)
        j, k, m = sorted_pairs[0]
        for i, (pj, pk, pm) in enumerate(pairs):
            if pj == j and pk == k and pm == m:
                return i
        return len(pairs)


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------
def get_all_baselines() -> List[BaselinePolicy]:
    """Returns all 4 baseline instances."""
    return [
        ReactiveBaseline(),
        RuleBasedEDFBaseline(),
        FixedIntervalSPTBaseline(),
        ABRMDDQRBaseline(),
    ]