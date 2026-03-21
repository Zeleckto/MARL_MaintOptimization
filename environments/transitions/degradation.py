"""
environments/transitions/degradation.py
========================================
Weibull degradation + Kijima Type I imperfect repair physics.

THIS IS THE ONLY FILE IN THE CODEBASE THAT KNOWS ABOUT:
    - Weibull parameters (beta, eta) per machine
    - Kijima Type I virtual age update
    - Health score dynamics (linear degradation in Tier 1, Weibull-driven here)
    - Bathtub curve phase classification

Design:
    MachineState is a plain dataclass — no PyTorch, no gym, just numbers.
    degradation.py operates on MachineState objects and returns updated ones.
    The environment (mfg_env.py) owns a list of MachineState objects and calls
    this module at the end of each full timestep during the physics resolution phase.

Tier 1 vs Tier 3 degradation:
    Tier 1 (CP-SAT): linear degradation h_{t+1} = h_t - delta_h * omega_t
                     (deterministic, tractable for constraint programming)
    Tier 3 (this file): Weibull-driven stochastic failures + Kijima virtual age
                     (realistic, used in RL environment)

Status machine for each machine:
    OP   -> operating normally
    PM   -> undergoing preventive maintenance (scheduled)
    CM   -> undergoing corrective maintenance (after failure)
    FAIL -> failed, awaiting CM (transition state, resolved within same step)

Usage:
    from environments.transitions.degradation import MachineState, DegradationEngine
    engine = DegradationEngine(config)
    state = engine.tick(state, is_operating=True, rng=rng)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from utils.distributions import (
    sample_weibull_failure,
    sample_repair_effectiveness,
    compute_weibull_hazard_rate,
    compute_weibull_rul,
    classify_bathtub_phase,
)


# =============================================================================
# MACHINE STATUS CONSTANTS
# =============================================================================

class MachineStatus:
    """
    Discrete operational status of a machine.
    Using class constants instead of Enum for easy numpy indexing.
    """
    OP   = 0   # operating: can receive job assignments
    PM   = 1   # preventive maintenance in progress: unavailable
    CM   = 2   # corrective maintenance in progress: unavailable
    FAIL = 3   # just failed: Agent 1 must initiate CM


# =============================================================================
# MACHINE STATE DATACLASS
# =============================================================================

@dataclass
class MachineState:
    """
    Complete state of a single machine at one timestep.

    All fields map directly to the 15-dim machine node feature vector
    in the TGIN (Table 3.9 in report), plus internal tracking fields.

    Fields used in TGIN node features (15-dim):
        status, health, hazard_rate, rul, bathtub_phase,
        queue_length (set by job_dynamics), cumulative_op_time, time_since_maint

    Fields used internally (not in TGIN):
        virtual_age, maint_steps_remaining, machine_id
    """
    # --- Identity ---
    machine_id: int                    # index 0..M-1

    # --- Weibull parameters (machine-specific, set at init, never change) ---
    beta: float                        # shape parameter
    eta: float                         # scale parameter (hours)
    delta_h: float                     # health degradation rate per operating shift (health units/shift)
    h_PM_threshold: float              # health at which PM should be triggered (Agent 1 decides)
    h_critical: float                  # health at which machine fails deterministically
    tau_PM_shifts: int                 # PM duration in timesteps (shifts)
    tau_CM_shifts: int                 # CM duration in timesteps
    h_restore_PM: float                # health restored by PM (Kijima-modulated externally)
    h_restore_CM: float                # health restored by CM

    # --- Dynamic state (changes every step) ---
    health: float = 100.0             # current health score [0, 100]
    virtual_age: float = 0.0         # Kijima virtual age (hours equivalent)
    status: int = MachineStatus.OP    # current operational status

    # --- Derived features (recomputed each tick, stored for observation) ---
    hazard_rate: float = 0.0          # Weibull instantaneous hazard rate
    rul: float = 0.0                  # estimated remaining useful life (hours)
    bathtub_phase: int = 2            # 0=infant, 1=useful, 2=wearout

    # --- Tracking ---
    cumulative_op_time: float = 0.0   # total operating hours since episode start
    time_since_maint: float = 0.0     # hours since last PM or CM completed
    maint_steps_remaining: int = 0    # countdown for ongoing maintenance duration
    queue_length: int = 0             # number of pending ops assigned (set by job_dynamics)

    # --- Repair effectiveness (Kijima q, sampled per repair event) ---
    last_repair_q: float = 0.5       # stored for logging/debugging


    def to_feature_vector(self) -> np.ndarray:
        """
        Returns 15-dim numpy array for TGIN machine node features.
        Order must match Table 3.9 in report and graph_builder.py exactly.

        Features:
            [0]  status (one-hot index: 0=OP,1=PM,2=CM,3=FAIL)
            [1]  health (normalised to [0,1])
            [2]  hazard_rate (log-scaled to avoid huge values)
            [3]  rul (normalised by 5*eta)
            [4]  bathtub_phase (0/1/2, as float)
            [5]  queue_length (normalised by max_jobs=30)
            [6]  cumulative_op_time (normalised by eta)
            [7]  time_since_maint (normalised by tau_CM)
            [8]  maint_steps_remaining (normalised by max(tau_CM_shifts, 1))
            [9]  virtual_age (normalised by eta)
            [10] is_operating (binary)
            [11] is_under_maint (binary: PM or CM)
            [12] is_failed (binary)
            [13] health_below_pm_threshold (binary signal)
            [14] health_below_critical (binary signal)
        """
        eta_h = self.eta  # characteristic life in hours
        tau_cm_h = self.tau_CM_shifts * 8.0  # convert shifts to hours

        features = np.array([
            float(self.status),                                          # [0]
            self.health / 100.0,                                        # [1]
            float(np.log1p(self.hazard_rate)),                         # [2] log1p avoids log(0)
            self.rul / max(5.0 * eta_h, 1.0),                         # [3]
            float(self.bathtub_phase),                                  # [4]
            self.queue_length / 30.0,                                   # [5]
            self.cumulative_op_time / max(eta_h, 1.0),                 # [6]
            self.time_since_maint / max(tau_cm_h, 1.0),               # [7]
            self.maint_steps_remaining / max(self.tau_CM_shifts, 1),  # [8]
            self.virtual_age / max(eta_h, 1.0),                       # [9]
            float(self.status == MachineStatus.OP),                    # [10]
            float(self.status in [MachineStatus.PM, MachineStatus.CM]), # [11]
            float(self.status == MachineStatus.FAIL),                  # [12]
            float(self.health <= self.h_PM_threshold),                 # [13]
            float(self.health <= self.h_critical),                     # [14]
        ], dtype=np.float32)

        return features


# =============================================================================
# DEGRADATION ENGINE
# =============================================================================

class DegradationEngine:
    """
    Manages all physics for machine health degradation and maintenance recovery.

    One DegradationEngine is shared across all machines in the environment.
    Call tick_all() once per full timestep (after both agents have acted).

    Phase 1 stochasticity: only Weibull failure sampling is random.
                           q is fixed at q_fixed=0.5.
    Phase 2+:              q is sampled from Beta distribution.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict (loaded from configs/phase1.yaml etc.)
                    Reads:
                        stochasticity_level: int (1, 2, or 3)
                        degradation.dt_hours: float (hours per timestep, default 8.0)
                        degradation.q_fixed: float (repair effectiveness for phase 1)
                        degradation.alpha_q: float (Beta alpha for phase 2+)
                        degradation.beta_q:  float (Beta beta for phase 2+)
        """
        self.stoch_level = config.get("stochasticity_level", 1)
        self.dt = config.get("degradation", {}).get("dt_hours", 8.0)  # hours per timestep
        self.q_fixed = config.get("degradation", {}).get("q_fixed", 0.5)
        self.alpha_q = config.get("degradation", {}).get("alpha_q", 5.0)
        self.beta_q_param = config.get("degradation", {}).get("beta_q", 5.0)


    def tick(
        self,
        state: MachineState,
        is_operating: bool,
        rng: np.random.Generator,
        action_maintenance: int = 0,  # 0=none, 1=PM initiated, 2=CM initiated
    ) -> MachineState:
        """
        Advances one machine's state by one full timestep.

        Called AFTER both agents have acted. The sequence within tick():
            1. If maintenance just completed this step -> apply Kijima health restore
            2. If machine is operating -> degrade health, sample Weibull failure
            3. If maintenance in progress -> decrement countdown
            4. If new maintenance initiated -> set status and countdown
            5. Recompute derived features (hazard rate, RUL, bathtub phase)

        Args:
            state:             Current MachineState
            is_operating:      True if machine processed a job this step
                               (set by job_dynamics before calling tick)
            rng:               NumPy Generator (seeded)
            action_maintenance: Agent 1's action for this machine this step

        Returns:
            Updated MachineState (mutates in place AND returns for chaining)
        """

        # ------------------------------------------------------------------
        # STEP 1: Count down ongoing maintenance
        # ------------------------------------------------------------------
        if state.status in [MachineStatus.PM, MachineStatus.CM]:
            state.maint_steps_remaining -= 1

            if state.maint_steps_remaining <= 0:
                # Maintenance completed this step — apply health restoration
                state = self._apply_repair(state, rng)
                state.status = MachineStatus.OP
                state.maint_steps_remaining = 0
                state.time_since_maint = 0.0

            # Machine is under maintenance — no degradation this step
            # Accumulate no operating time
            state = self._recompute_derived(state)
            return state

        # ------------------------------------------------------------------
        # STEP 2: Handle FAIL status (CM must have been initiated by Agent 1)
        # ------------------------------------------------------------------
        if state.status == MachineStatus.FAIL:
            # Agent 1 should have set action_maintenance=2 (CM)
            # If not, machine stays FAIL — penalty accumulates via shared reward
            if action_maintenance == 2:  # CM initiated
                state.status = MachineStatus.CM
                state.maint_steps_remaining = state.tau_CM_shifts
            # Either way, no degradation or operating this step
            state = self._recompute_derived(state)
            return state

        # ------------------------------------------------------------------
        # STEP 3: Handle new maintenance initiation (Agent 1 action)
        # ------------------------------------------------------------------
        if action_maintenance == 1 and state.status == MachineStatus.OP:
            # PM initiated — machine must be idle (checked by action masking)
            state.status = MachineStatus.PM
            state.maint_steps_remaining = state.tau_PM_shifts
            state = self._recompute_derived(state)
            return state

        # action_maintenance == 2 on an OP machine should not happen
        # (CM mask only allows it on FAIL) — but guard anyway
        if action_maintenance == 2 and state.status == MachineStatus.OP:
            # Invalid action got through masking — do nothing, log warning
            # In production, raise ValueError; during training, be lenient
            pass

        # ------------------------------------------------------------------
        # STEP 4: Operating machine — degrade and check failure
        # ------------------------------------------------------------------
        if is_operating and state.status == MachineStatus.OP:
            # Linear health degradation (same as Tier 1 but can fail stochastically)
            state.health -= state.delta_h
            state.health = max(state.health, 0.0)  # floor at 0

            # Accumulate operating time
            state.cumulative_op_time += self.dt
            state.time_since_maint += self.dt

            # Weibull stochastic failure check
            failed = sample_weibull_failure(
                virtual_age=state.virtual_age,
                beta=state.beta,
                eta=state.eta,
                dt=self.dt,
                rng=rng,
            )

            # Deterministic failure if health hits critical threshold
            if state.health <= state.h_critical:
                failed = True

            if failed:
                state.status = MachineStatus.FAIL
                # Note: CM will be initiated next step when Agent 1 acts
                # The shock absorber mechanism handles this (see failure_handler.py)

        elif state.status == MachineStatus.OP and not is_operating:
            # Machine idle this step — still accumulate idle time
            state.time_since_maint += self.dt
            # No degradation when idle (common manufacturing assumption)

        # ------------------------------------------------------------------
        # STEP 5: Recompute derived features
        # ------------------------------------------------------------------
        state = self._recompute_derived(state)
        return state


    def _apply_repair(self, state: MachineState, rng: np.random.Generator) -> MachineState:
        """
        Applies Kijima Type I imperfect repair after maintenance completes.

        Kijima Type I:  v_new = v_old + q * X_n
        where:
            v_old = virtual age before this repair
            X_n   = operating time since last repair (approximated as time_since_maint)
            q     = repair effectiveness (0=perfect repair, 1=minimal repair)

        Health restoration is separate from virtual age — health is restored
        by h_restore amount (machine-specific, based on maintenance type).

        Phase 1: q = q_fixed (deterministic)
        Phase 2+: q sampled from Beta(alpha_q, beta_q)
        """
        # Sample repair effectiveness
        if self.stoch_level >= 2:
            q = sample_repair_effectiveness(self.alpha_q, self.beta_q_param, rng)
        else:
            q = self.q_fixed

        state.last_repair_q = q

        # Kijima Type I virtual age update
        # X_n approximated as time elapsed since last repair (in hours)
        X_n = state.time_since_maint
        state.virtual_age = state.virtual_age + q * X_n

        # Virtual age can't be negative (shouldn't happen but guard it)
        state.virtual_age = max(state.virtual_age, 0.0)

        # Health restoration
        if state.status == MachineStatus.PM:
            state.health = min(100.0, state.health + state.h_restore_PM)
        elif state.status == MachineStatus.CM:
            state.health = min(100.0, state.health + state.h_restore_CM)

        return state


    def _recompute_derived(self, state: MachineState) -> MachineState:
        """
        Recomputes hazard_rate, rul, bathtub_phase using EFFECTIVE age.

        Kijima virtual_age only updates at repair events (post-repair age).
        Between repairs, the machine effective age is:
            effective_age = virtual_age + time_since_maint

        This correctly captures that the machine has been running for
        time_since_maint hours starting from its post-repair virtual age.

        Example:
            After repair: virtual_age=1250, time_since_maint=0
            After 100h:   effective_age = 1250 + 100 = 1350  (correct)
            Using virtual_age alone would give 1250 (stale - BUG)
        """
        # Effective age = Kijima post-repair base + hours operated since last repair
        effective_age = state.virtual_age + state.time_since_maint

        state.hazard_rate = compute_weibull_hazard_rate(
            effective_age, state.beta, state.eta
        )
        state.rul = compute_weibull_rul(
            effective_age, state.beta, state.eta
        )
        state.bathtub_phase = classify_bathtub_phase(state.beta)
        return state


    def tick_all(
        self,
        machine_states: List[MachineState],
        operating_flags: List[bool],
        rng: np.random.Generator,
        actions_maintenance: List[int],
    ) -> List[MachineState]:
        """
        Convenience wrapper: ticks all machines in one call.

        Args:
            machine_states:      List of MachineState, one per machine
            operating_flags:     List[bool], True if machine processed a job this step
            rng:                 NumPy Generator
            actions_maintenance: List[int], Agent 1's maintenance action per machine

        Returns:
            Updated list of MachineState
        """
        assert len(machine_states) == len(operating_flags) == len(actions_maintenance), (
            f"Length mismatch: {len(machine_states)} states, "
            f"{len(operating_flags)} flags, {len(actions_maintenance)} actions"
        )

        updated = []
        for state, operating, action in zip(machine_states, operating_flags, actions_maintenance):
            updated.append(self.tick(state, operating, rng, action))
        return updated


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def build_machine_states(machine_configs: List[dict], stoch_level: int = 1) -> List[MachineState]:
    """
    Instantiates MachineState objects from the benchmark instance config.

    Args:
        machine_configs: List of dicts, one per machine. Each dict has keys:
            machine_id, beta, eta, delta_h, h_PM_threshold, h_critical,
            tau_PM_shifts, tau_CM_shifts, h_restore_PM, h_restore_CM
        stoch_level: Not used here but included for future extension

    Returns:
        List of MachineState, all initialised at health=100, virtual_age=0, status=OP

    Example machine_config dict (from small_3m_5j.yaml):
        {
            machine_id: 0,
            beta: 2.8, eta: 3000.0,
            delta_h: 0.5,          # loses 0.5 health per operating shift
            h_PM_threshold: 40.0,  # PM recommended below 40
            h_critical: 10.0,      # fails deterministically below 10
            tau_PM_shifts: 3,      # PM takes 3 shifts
            tau_CM_shifts: 8,      # CM takes 8 shifts
            h_restore_PM: 30.0,    # PM restores 30 health units
            h_restore_CM: 60.0,    # CM restores 60 health units
        }
    """
    states = []
    for cfg in machine_configs:
        state = MachineState(
            machine_id=cfg["machine_id"],
            beta=cfg["beta"],
            eta=cfg["eta"],
            delta_h=cfg["delta_h"],
            h_PM_threshold=cfg["h_PM_threshold"],
            h_critical=cfg["h_critical"],
            tau_PM_shifts=cfg["tau_PM_shifts"],
            tau_CM_shifts=cfg["tau_CM_shifts"],
            h_restore_PM=cfg["h_restore_PM"],
            h_restore_CM=cfg["h_restore_CM"],
            # All dynamic fields default to initial values in dataclass
        )
        # Compute initial derived features
        state.hazard_rate = compute_weibull_hazard_rate(0.0, state.beta, state.eta)
        state.rul = compute_weibull_rul(0.0, state.beta, state.eta)
        state.bathtub_phase = classify_bathtub_phase(state.beta)
        states.append(state)
    return states
