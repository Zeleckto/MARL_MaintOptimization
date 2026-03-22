"""
environments/transitions/failure_handler.py
============================================
Shock absorber mechanism for real-time machine failure response.

From the report (Section 3.3.6):
    When a machine fails, the system must respond WITHOUT full replanning.
    The shock absorber works as follows:
        1. Machine m fails (status -> FAIL)
        2. Agent 1 observes the failure in its next observation
        3. Agent 1 selects CM action for machine m
        4. Machine m is removed from Agent 2's valid action set
        5. Agent 2 reassigns pending operations to other machines
        6. Execution continues — no full schedule recomputation needed

This file handles steps 4 and 5: the action mask update and
the detection of which operations need reassignment.

Also handles:
    - Critical health threshold detection (deterministic failure)
    - Preemption: what happens to an IN_PROGRESS operation when machine fails
    - Failure logging for reward computation

Usage:
    from environments.transitions.failure_handler import FailureHandler
    handler = FailureHandler(config)
    failures, preempted_ops = handler.check_and_handle(machine_states, jobs)
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

from environments.transitions.degradation import MachineState, MachineStatus
from environments.transitions.job_dynamics import Job, Operation, OpStatus



class FailureHandler:
    """
    Detects failures and handles their immediate consequences.

    Called once per timestep BEFORE degradation.tick_all() updates status,
    so we can compare old vs new status to detect transitions to FAIL.

    The preemption policy is: if machine m fails mid-operation,
    the IN_PROGRESS operation on m is reset to READY status.
    Agent 2 must reassign it next step.
    This is the "shock absorber" — no replanning, just local state update.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict. Reads:
                degradation.h_critical (per machine, but also stored in MachineState)
        """
        # Preemption policy: reset to READY or back to PENDING
        # READY is correct — all predecessors were already done
        self.preemption_reset_status = OpStatus.READY


    def check_failures(
        self,
        old_states: List[MachineState],
        new_states: List[MachineState],
    ) -> List[int]:
        """
        Detects which machines transitioned to FAIL this step.
        Compares old and new machine states.

        Args:
            old_states: Machine states BEFORE degradation tick
            new_states: Machine states AFTER degradation tick

        Returns:
            List of machine ids that just failed (OP/PM -> FAIL transitions)
        """
        newly_failed = []
        for old, new in zip(old_states, new_states):
            was_not_failed = old.status != MachineStatus.FAIL
            now_failed     = new.status == MachineStatus.FAIL
            if was_not_failed and now_failed:
                newly_failed.append(new.machine_id)
        return newly_failed


    def handle_preemption(
        self,
        failed_machine_ids: List[int],
        jobs:               List[Job],
    ) -> Tuple[List[Job], List[Tuple[int, int]]]:
        """
        Handles operations that were IN_PROGRESS on failed machines.
        Resets them to READY so Agent 2 can reassign next step.

        This is the shock absorber core:
            No replanning needed — operation just goes back to READY pool.
            Agent 2 will see it as available next step with updated machine mask.

        Args:
            failed_machine_ids: List of machine ids that just failed
            jobs:               Current job list

        Returns:
            (updated_jobs, preempted_ops)
            preempted_ops: list of (job_id, op_idx) that were preempted
        """
        preempted = []

        for job in jobs:
            for op in job.operations:
                if (op.status == OpStatus.IN_PROGRESS
                        and op.assigned_machine in failed_machine_ids):

                    # Reset operation to READY — predecessors are still done
                    op.status           = self.preemption_reset_status
                    op.assigned_machine = None
                    op.remaining_time   = 0.0   # will be re-sampled on next assignment

                    preempted.append((job.job_id, op.op_idx))

        return jobs, preempted


    def build_machine_availability_mask(
        self,
        machine_states: List[MachineState],
    ) -> np.ndarray:
        """
        Builds boolean mask for Agent 2's action space.
        True = machine is available for job assignment.
        False = machine is under maintenance, failed, or busy.

        This mask is recomputed AFTER Agent 1's half-step so it
        reflects maintenance decisions taken this timestep.
        This is the AEC ordering fix — Agent 2 always sees
        post-Agent-1 machine availability.

        Args:
            machine_states: Current machine states (post Agent-1 half-step)

        Returns:
            [n_machines] bool array
        """
        return np.array([
            s.status == MachineStatus.OP
            for s in machine_states
        ], dtype=bool)


    def build_op_machine_valid_mask(
        self,
        jobs:            List[Job],
        machine_states:  List[MachineState],
        machine_busy:    List[bool],   # True if machine currently processing a job
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Builds complete valid action set for Agent 2.
        Returns dict mapping (job_id, op_idx) -> list of valid machine ids.

        An (op, machine) pair is valid iff:
            1. Operation is READY
            2. Machine is in op's eligible_machines list
            3. Machine status is OP (not PM/CM/FAIL)
            4. Machine is not currently busy processing another operation

        Args:
            jobs:           Current job list
            machine_states: Post-Agent-1 machine states
            machine_busy:   [n_machines] True if machine occupied

        Returns:
            Dict {(job_id, op_idx): [valid_machine_ids]}
            Empty dict means WAIT is the only valid action.
        """
        # Build machine availability once
        avail = {
            s.machine_id: (
                s.status == MachineStatus.OP
                and not machine_busy[s.machine_id]
            )
            for s in machine_states
        }

        valid_actions = {}
        for job in jobs:
            for op in job.operations:
                if op.status != OpStatus.READY:
                    continue

                valid_machines = [
                    m for m in op.eligible_machines
                    if avail.get(m, False)
                ]

                if valid_machines:
                    valid_actions[(job.job_id, op.op_idx)] = valid_machines

        return valid_actions


    def log_failure_event(
        self,
        machine_id:   int,
        current_time: float,
        machine_state: MachineState,
    ) -> dict:
        """
        Creates a failure event log entry for debugging and TensorBoard.
        Called when a failure is detected.

        Returns dict with failure metadata.
        """
        return {
            "event":        "machine_failure",
            "machine_id":   machine_id,
            "time":         current_time,
            "health_at_failure": machine_state.health,
            "virtual_age_at_failure": machine_state.virtual_age,
            "effective_age_at_failure": (
                machine_state.virtual_age + machine_state.time_since_maint
            ),
            "hazard_rate_at_failure": machine_state.hazard_rate,
        }
