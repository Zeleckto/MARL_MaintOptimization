"""
environments/transitions/degradation.py
=========================================
Weibull-based machine degradation with Kijima Type I imperfect repair.

KEY REDESIGN (Phase 0):
    OLD: H(t) = H_prev - delta_h  (linear counter, meaningless)
    NEW: H(t) = 100 * exp(-(effective_age / eta)^beta)
         where effective_age = virtual_age + time_since_last_maint

    H now represents the Weibull survival probability (%) — a principled
    definition. H=80 means the machine has an 80% survival probability
    at its current effective age.

    NO hardcoded PM threshold. Agent discovers optimal timing from:
        c_fail >> c_CM >> c_PM  (cost ratios drive the policy)

Failure sampling:
    P(fail this step | survived to effective_age) = conditional Weibull:
        p_fail = 1 - exp(-[(age+1/eta)^beta - (age/eta)^beta])
    This correctly samples from the hazard rate without any threshold.

Kijima Type I imperfect repair:
    V_n = V_{n-1} + q * X_n
    X_n = age at repair,  q = repair quality factor (0=perfect, 1=minimal)

Feature vector (15 dims — must match observation_spaces.py):
    [0]  health / 100                     normalised survival prob
    [1]  virtual_age / eta                age ratio (Kijima)
    [2]  time_since_maint / eta           recent age ratio
    [3]  effective_age / eta              total age ratio
    [4]  hazard_rate (normalised)         λ(t) / λ_max
    [5]  status (0=OP, 0.33=PM, 0.67=CM, 1=FAIL)
    [6]  maint_steps_remaining / tau_CM   maintenance progress
    [7]  is_OP binary
    [8]  is_FAIL binary
    [9]  is_under_maint binary (PM or CM)
    [10] cumulative_PM_count / 10         normalised PM count
    [11] cumulative_CM_count / 5          normalised CM count
    [12] cumulative_op_time / (T_max * dt) utilisation fraction
    [13] time_since_maint / 50            raw time since last maint
    [14] beta / 3.5                       Weibull shape (informational)
"""

import math
import copy
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# STATUS CONSTANTS
# ---------------------------------------------------------------------------
class MachineStatus:
    OP   = 0   # operational
    PM   = 1   # planned maintenance
    CM   = 2   # corrective maintenance
    FAIL = 3   # failed (waiting for CM)


# ---------------------------------------------------------------------------
# MACHINE STATE DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class MachineState:
    """
    Full state of one machine.
    All fields are updated by DegradationEngine.tick_one().
    """
    machine_id:   int
    name:         str
    machine_type: str          # "A", "B", "C", "D"

    # Weibull parameters
    beta:         float        # shape (>1 = wear-out dominant)
    eta:          float        # characteristic life in shifts
    q_PM:         float        # Kijima repair factor for PM
    q_CM:         float        # Kijima repair factor for CM
    h_critical:   float        # health below which failure prob jumps (for logging only)

    # Maintenance durations
    tau_PM:       int          # shifts for PM
    tau_CM:       int          # shifts for CM

    # Current status
    status:       int = MachineStatus.OP

    # Age tracking (Kijima)
    virtual_age:       float = 0.0   # V_n in Kijima model
    time_since_maint:  float = 0.0   # resets to 0 after each repair

    # Maintenance countdown
    maint_steps_remaining: int = 0   # >0 means machine is in PM/CM

    # Derived quantities (updated each step)
    health:       float = 100.0      # H(t) = 100 * exp(-(eff_age/eta)^beta)
    hazard_rate:  float = 0.0        # λ(t) = (beta/eta) * (eff_age/eta)^(beta-1)

    # Statistics
    cumulative_op_time:  float = 0.0
    cumulative_PM_count: int   = 0
    cumulative_CM_count: int   = 0

    def __post_init__(self):
        self._update_derived()

    @property
    def effective_age(self) -> float:
        """Age used in Weibull hazard: virtual_age + time_since_last_maint."""
        return self.virtual_age + self.time_since_maint

    def _update_derived(self) -> None:
        """Recompute health and hazard_rate from effective_age."""
        age = max(self.effective_age, 0.0)
        eta  = max(self.eta, 1e-6)
        beta = max(self.beta, 0.1)

        ratio = age / eta

        # Weibull survival function: H = 100 * exp(-(age/eta)^beta)
        self.health = 100.0 * math.exp(-ratio ** beta)

        # Weibull hazard rate: lambda(t) = (beta/eta) * (age/eta)^(beta-1)
        if age > 0:
            self.hazard_rate = (beta / eta) * (ratio ** (beta - 1.0))
        else:
            self.hazard_rate = 0.0

    def failure_probability_this_step(self) -> float:
        """
        P(fail in [age, age+1] | survived to age).
        Conditional Weibull probability — the correct way to sample failures.

        p = 1 - exp(-integral of lambda from age to age+1)
          = 1 - exp(-[(age+1/eta)^beta - (age/eta)^beta])
        """
        age  = max(self.effective_age, 0.0)
        eta  = max(self.eta, 1e-6)
        beta = max(self.beta, 0.1)

        rate_now  = (age / eta) ** beta
        rate_next = ((age + 1.0) / eta) ** beta

        return 1.0 - math.exp(-(rate_next - rate_now))

    def to_feature_vector(self) -> np.ndarray:
        """
        Returns 15-dim feature vector for Agent 1 flat observation.
        Must match MACHINE_FEATURE_DIM in observation_spaces.py.
        """
        eta  = max(self.eta, 1e-6)
        # Hazard rate normalisation: max hazard at age=T_max=150 shifts
        max_age  = 150.0
        max_hz   = (self.beta / eta) * (max_age / eta) ** (self.beta - 1.0)
        hz_norm  = min(self.hazard_rate / max(max_hz, 1e-12), 1.0)

        tau_max  = float(max(self.tau_CM, 1))

        return np.array([
            self.health / 100.0,                                    # [0]
            min(self.virtual_age / eta, 1.0),                       # [1]
            min(self.time_since_maint / eta, 1.0),                  # [2]
            min(self.effective_age / eta, 1.0),                     # [3]
            hz_norm,                                                 # [4]
            self.status / 3.0,                                      # [5]
            self.maint_steps_remaining / tau_max,                   # [6]
            float(self.status == MachineStatus.OP),                 # [7]
            float(self.status == MachineStatus.FAIL),               # [8]
            float(self.status in (MachineStatus.PM, MachineStatus.CM)),  # [9]
            min(self.cumulative_PM_count / 10.0, 1.0),              # [10]
            min(self.cumulative_CM_count / 5.0, 1.0),               # [11]
            min(self.cumulative_op_time / (150.0 * 8.0), 1.0),      # [12]
            min(self.time_since_maint / 50.0, 1.0),                 # [13]
            min(self.beta / 3.5, 1.0),                              # [14]
        ], dtype=np.float32)


MACHINE_FEATURE_DIM = 15


# ---------------------------------------------------------------------------
# DEGRADATION ENGINE
# ---------------------------------------------------------------------------
class DegradationEngine:
    """
    Manages Weibull degradation + Kijima repair for all machines.

    Called once per full timestep from mfg_env._resolve_physics().

    Sequence per tick_all():
        1. For each machine in PM/CM: decrement maint_steps_remaining
           If complete: restore age via Kijima, transition to OP.
        2. For machines just sent to PM/CM this step: set status, duration.
        3. For OP machines: increment time_since_maint, sample failure.
        4. Update health and hazard_rate for all machines.
    """

    def __init__(self, config: dict):
        self.config = config
        self.stoch_level = config.get("stochasticity_level", 1)
        machines_cfg = config.get("machines", [])
        self._machine_cfgs = {c["machine_id"]: c for c in machines_cfg}

    def tick_all(
        self,
        machine_states: List[MachineState],
        operating_flags: List[bool],     # True if machine was processing a job
        rng: np.random.Generator,
        actions_maintenance: List[int],  # [n_mach] 0=none, 1=PM, 2=CM
    ) -> List[MachineState]:
        """
        Advances all machine states by one shift.

        Args:
            machine_states:       Current states list
            operating_flags:      [n_mach] True = machine was busy this step
            rng:                  NumPy Generator for stochastic events
            actions_maintenance:  Agent 1's maintenance actions this step

        Returns:
            Updated machine_states list (same objects, modified in place)
        """
        for s in machine_states:
            i = s.machine_id
            action = actions_maintenance[i] if i < len(actions_maintenance) else 0

            if s.status in (MachineStatus.PM, MachineStatus.CM):
                # --- Maintenance in progress ---
                s.maint_steps_remaining -= 1

                if s.maint_steps_remaining <= 0:
                    # Maintenance complete — apply Kijima repair
                    s.maint_steps_remaining = 0
                    if s.status == MachineStatus.PM:
                        q = s.q_PM
                    else:  # CM
                        q = s.q_CM

                    # Kijima Type I: V_n = V_{n-1} + q * X_n
                    # X_n = age at repair = time_since_maint
                    s.virtual_age = s.virtual_age + q * s.time_since_maint
                    s.time_since_maint = 0.0
                    s.status = MachineStatus.OP

            elif s.status == MachineStatus.FAIL:
                # --- Failed, waiting for CM ---
                if action == 2:
                    # Agent 1 initiates CM
                    s.status = MachineStatus.CM
                    s.maint_steps_remaining = s.tau_CM
                    s.cumulative_CM_count += 1
                # If agent doesn't initiate CM, machine stays failed

            elif s.status == MachineStatus.OP:
                # --- Operational ---
                if action == 1 and not operating_flags[i]:
                    # Agent 1 initiates PM (only on idle machine)
                    s.status = MachineStatus.PM
                    s.maint_steps_remaining = s.tau_PM
                    s.cumulative_PM_count += 1
                else:
                    # Normal operation: age accumulates
                    s.time_since_maint += 1.0

                    if operating_flags[i]:
                        s.cumulative_op_time += 8.0  # dt_hours

                    # Sample failure via conditional Weibull
                    p_fail = s.failure_probability_this_step()
                    if rng.random() < p_fail:
                        s.status = MachineStatus.FAIL

            # Update derived quantities for all machines
            s._update_derived()

        return machine_states


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------
def build_machine_states(machines_cfg: List[dict]) -> List[MachineState]:
    """
    Constructs MachineState list from config dicts.

    Args:
        machines_cfg: List of machine config dicts from base.yaml

    Returns:
        List of fresh MachineState objects, all starting at OP, age=0
    """
    states = []
    for cfg in machines_cfg:
        s = MachineState(
            machine_id   = cfg["machine_id"],
            name         = cfg.get("name", f"M{cfg['machine_id']}"),
            machine_type = cfg.get("machine_type", "A"),
            beta         = float(cfg.get("beta",         2.5)),
            eta          = float(cfg.get("eta",          120.0)),
            q_PM         = float(cfg.get("q_PM",         0.4)),
            q_CM         = float(cfg.get("q_CM",         0.2)),
            h_critical   = float(cfg.get("h_critical",   15.0)),
            tau_PM       = int(cfg.get("tau_PM_shifts",  3)),
            tau_CM       = int(cfg.get("tau_CM_shifts",  8)),
        )
        states.append(s)
    return states


# ---------------------------------------------------------------------------
# RUL ESTIMATION UTILITY
# ---------------------------------------------------------------------------
def estimate_rul(state: MachineState, target_health: float = 50.0) -> float:
    """
    Estimates Remaining Useful Life (shifts) until health drops to target_health.

    RUL = age_at_target - effective_age_now
    where age_at_target = eta * (-ln(target_health/100))^(1/beta)

    Args:
        state:          MachineState
        target_health:  Health threshold (default 50%)

    Returns:
        RUL in shifts (0 if already below threshold)
    """
    if state.health <= target_health:
        return 0.0

    target_ratio = -math.log(target_health / 100.0)
    age_at_target = state.eta * (target_ratio ** (1.0 / state.beta))

    rul = age_at_target - state.effective_age
    return max(rul, 0.0)
