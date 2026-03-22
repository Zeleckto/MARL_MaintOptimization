"""
environments/transitions/job_dynamics.py
=========================================
Job and operation lifecycle management.

Handles:
    - Job and operation data structures
    - Processing time decrement each timestep
    - Operation completion detection
    - Job completion and tardiness calculation
    - Job arrivals (batch at episode start for Phase 1/2, Poisson for Phase 3)
    - Compatibility matrix: which operations can run on which machines

Key design:
    Operation has a status: PENDING -> READY -> IN_PROGRESS -> DONE
    PENDING:     predecessor operations not yet complete
    READY:       all predecessors done, can be assigned to a machine
    IN_PROGRESS: assigned to a machine, processing time counting down
    DONE:        completed

    Agent 2 can only assign READY operations.
    Action masking enforces this — no need to check in policy network.

Usage:
    from environments.transitions.job_dynamics import (
        Job, Operation, OpStatus, JobDynamicsEngine
    )
    engine = JobDynamicsEngine(config)
    jobs = engine.generate_job_batch(n_jobs=20, rng=rng)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from utils.distributions import (
    sample_processing_time,
    sample_job_arrivals,
)


# =============================================================================
# OPERATION STATUS CONSTANTS
# =============================================================================

class OpStatus:
    PENDING     = 0   # waiting for predecessors
    READY       = 1   # can be assigned to a machine
    IN_PROGRESS = 2   # currently being processed
    DONE        = 3   # completed


# =============================================================================
# OPERATION DATACLASS
# =============================================================================

@dataclass
class Operation:
    """
    Single operation (j, k) — one processing step of a job.

    Maps to one node in the TGIN tripartite graph (Op node, 10-dim features).
    Feature vector order must match Table 3.9 in report and graph_builder.py.
    """
    job_id:       int           # j — which job this belongs to
    op_idx:       int           # k — position in job's operation sequence (0-indexed)
    status:       int = OpStatus.PENDING

    # Eligible machines for this operation (subset of M)
    eligible_machines: List[int] = field(default_factory=list)

    # Processing times per eligible machine (hours) — nominal values
    nominal_proc_times: Dict[int, float] = field(default_factory=dict)

    # Actual processing time once assigned (sampled from LogNormal in Phase 2+)
    actual_proc_time: float = 0.0

    # Assignment tracking
    assigned_machine: Optional[int] = None  # machine id when IN_PROGRESS
    remaining_time:   float = 0.0           # shifts remaining until completion
    start_time:       float = 0.0           # timestep when processing started
    completion_time:  float = 0.0           # timestep when completed

    def to_feature_vector(
        self,
        job_due_date:    float,
        current_time:    float,
        n_total_machines: int,
        n_total_ops:     int,
    ) -> np.ndarray:
        """
        Returns 10-dim feature vector for TGIN Op node.
        Must match Table 3.9 in report exactly.

        Features:
            [0]  status (normalised: 0=pending, 0.33=ready, 0.67=in_prog, 1=done)
            [1]  min processing time across eligible machines (normalised by 50h)
            [2]  avg processing time across eligible machines (normalised by 50h)
            [3]  eligibility ratio: |eligible_machines| / n_total_machines
            [4]  position in job (op_idx / n_total_ops)
            [5]  time to job due date (normalised by 200 steps)
            [6]  remaining processing time (normalised by 50h) — 0 if not in progress
            [7]  is_ready binary flag
            [8]  is_in_progress binary flag
            [9]  slack estimate (time_to_due - remaining_proc / dt_hours)
        """
        dt_hours = 8.0  # hours per timestep

        # Processing time stats
        if self.nominal_proc_times:
            proc_times = list(self.nominal_proc_times.values())
            min_proc = min(proc_times) / 50.0
            avg_proc = sum(proc_times) / len(proc_times) / 50.0
        else:
            min_proc = avg_proc = 0.0

        # Time to due date
        time_to_due = max(job_due_date - current_time, 0.0) / 200.0

        # Remaining time
        remaining_norm = self.remaining_time * dt_hours / 50.0

        # Slack: shifts until due minus shifts of remaining processing
        slack_shifts = (job_due_date - current_time) - self.remaining_time
        slack_norm = np.clip(slack_shifts / 200.0, -1.0, 1.0)

        return np.array([
            self.status / 3.0,                              # [0]
            np.clip(min_proc, 0.0, 1.0),                   # [1]
            np.clip(avg_proc, 0.0, 1.0),                   # [2]
            len(self.eligible_machines) / max(n_total_machines, 1),  # [3]
            self.op_idx / max(n_total_ops - 1, 1),          # [4]
            np.clip(time_to_due, 0.0, 1.0),                # [5]
            np.clip(remaining_norm, 0.0, 1.0),             # [6]
            float(self.status == OpStatus.READY),           # [7]
            float(self.status == OpStatus.IN_PROGRESS),     # [8]
            np.clip(slack_norm, -1.0, 1.0),                # [9]
        ], dtype=np.float32)


# =============================================================================
# JOB DATACLASS
# =============================================================================

@dataclass
class Job:
    """
    A job j consisting of n_j ordered operations.

    Maps to one node in the TGIN tripartite graph (Job node, 7-dim features).
    """
    job_id:       int
    release_time: float          # r_j: earliest start time (timestep)
    due_date:     float          # d_j: deadline (timestep)
    weight:       float = 1.0   # w_j: priority weight for tardiness

    operations:   List[Operation] = field(default_factory=list)

    # Tracking
    completion_time: Optional[float] = None  # set when all ops done
    tardiness:       float = 0.0

    @property
    def n_ops(self) -> int:
        return len(self.operations)

    @property
    def is_complete(self) -> bool:
        return all(op.status == OpStatus.DONE for op in self.operations)

    @property
    def current_op_idx(self) -> int:
        """Index of first non-DONE operation."""
        for i, op in enumerate(self.operations):
            if op.status != OpStatus.DONE:
                return i
        return self.n_ops  # all done

    @property
    def completion_ratio(self) -> float:
        """Fraction of operations completed."""
        done = sum(1 for op in self.operations if op.status == OpStatus.DONE)
        return done / max(self.n_ops, 1)

    def to_feature_vector(self, current_time: float) -> np.ndarray:
        """
        Returns 7-dim feature vector for TGIN Job node.
        Must match Table 3.9 in report exactly.

        Features:
            [0]  weight w_j (normalised by max weight = 5)
            [1]  time to due date (normalised by 200 steps)
            [2]  slack: time_to_due - remaining_work_estimate (normalised)
            [3]  current op index (normalised by n_ops)
            [4]  completion ratio [0, 1]
            [5]  lateness indicator: 1 if already late, 0 otherwise
            [6]  number of remaining operations (normalised by max_ops)
        """
        time_to_due = max(self.due_date - current_time, 0.0) / 200.0

        # Rough remaining work estimate (sum of min proc times of remaining ops)
        remaining_ops = [
            op for op in self.operations if op.status != OpStatus.DONE
        ]
        remaining_work = sum(
            min(op.nominal_proc_times.values()) / 8.0   # convert hrs to shifts
            if op.nominal_proc_times else 1.0
            for op in remaining_ops
        )
        slack = ((self.due_date - current_time) - remaining_work) / 200.0

        return np.array([
            self.weight / 5.0,                                    # [0]
            np.clip(time_to_due, 0.0, 1.0),                      # [1]
            np.clip(slack, -1.0, 1.0),                           # [2]
            self.current_op_idx / max(self.n_ops, 1),            # [3]
            self.completion_ratio,                                 # [4]
            float(current_time > self.due_date),                  # [5]
            len(remaining_ops) / max(self.n_ops, 1),             # [6]
        ], dtype=np.float32)


# =============================================================================
# JOB DYNAMICS ENGINE
# =============================================================================

class JobDynamicsEngine:
    """
    Manages all job and operation lifecycle transitions.

    Called once per full timestep during physics resolution.
    Sequence within tick():
        1. Decrement remaining_time for all IN_PROGRESS operations
        2. Detect and complete finished operations
        3. Update PENDING -> READY for newly unblocked operations
        4. Compute tardiness for newly completed jobs
        5. Phase 3: sample new job arrivals
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict. Reads:
                stochasticity_level
                jobs.n_ops_per_job_min/max
                jobs.lambda_arr (Phase 3)
                processing.sigma_log
                episode.dt_hours
                machines (for compatibility matrix generation)
        """
        self.stoch_level  = config.get("stochasticity_level", 1)
        self.dt           = config.get("episode", {}).get("dt_hours", 8.0)
        self.n_ops_min    = config.get("jobs", {}).get("n_ops_per_job_min", 2)
        self.n_ops_max    = config.get("jobs", {}).get("n_ops_per_job_max", 5)
        self.lambda_arr   = config.get("jobs", {}).get("lambda_arr", 0.5)
        self.sigma_log    = config.get("processing", {}).get("sigma_log", 0.15)
        self.n_machines   = len(config.get("machines", []))
        self.t_max_train  = config.get("episode", {}).get("t_max_train", 200)


    def generate_job_batch(
        self,
        n_jobs: int,
        rng:    np.random.Generator,
        start_job_id: int = 0,
    ) -> List[Job]:
        """
        Generates a batch of jobs for episode start (Phase 1 and 2).
        All jobs released at t=0 with staggered due dates.

        Args:
            n_jobs:       Number of jobs to generate
            rng:          NumPy Generator
            start_job_id: Starting job id (for Phase 3 dynamic arrivals)

        Returns:
            List of Job objects with operations and compatibility
        """
        jobs = []
        for j in range(n_jobs):
            job_id = start_job_id + j

            # Number of operations for this job
            n_ops = int(rng.integers(self.n_ops_min, self.n_ops_max + 1))

            # Due date: staggered to create varying urgency
            # Earlier jobs get tighter deadlines to create scheduling pressure
            due_date = float(rng.integers(
                low  = max(n_ops * 2, 10),          # minimum: 2 shifts per op
                high = self.t_max_train - 10,        # must complete before episode ends
            ))

            # Priority weight: uniform [1, 3] — higher weight = more urgent
            weight = float(rng.choice([1.0, 2.0, 3.0]))

            # Build operations
            operations = self._generate_operations(job_id, n_ops, rng)

            job = Job(
                job_id=job_id,
                release_time=0.0,
                due_date=due_date,
                weight=weight,
                operations=operations,
            )
            jobs.append(job)

        # Set first operation of each job to READY (no predecessors)
        for job in jobs:
            if job.operations:
                job.operations[0].status = OpStatus.READY

        return jobs


    def _generate_operations(
        self,
        job_id: int,
        n_ops:  int,
        rng:    np.random.Generator,
    ) -> List[Operation]:
        """
        Generates operations for one job with random machine eligibility
        and processing times.

        Each operation is eligible on at least 1 machine (guaranteed).
        Most operations are eligible on 2-3 machines (flexible job shop).
        """
        operations = []
        for k in range(n_ops):
            # Random subset of eligible machines (at least 1, at most all)
            # Probability of each machine being eligible: 0.6
            eligible = [
                m for m in range(self.n_machines)
                if rng.random() < 0.6
            ]
            # Guarantee at least one eligible machine
            if not eligible:
                eligible = [int(rng.integers(0, self.n_machines))]

            # Processing time per eligible machine (hours)
            # Uniform [1h, 16h] = [0.125 shifts, 2 shifts]
            nominal_proc_times = {
                m: float(rng.uniform(1.0, 16.0))
                for m in eligible
            }

            op = Operation(
                job_id=job_id,
                op_idx=k,
                status=OpStatus.PENDING,   # will update to READY below
                eligible_machines=eligible,
                nominal_proc_times=nominal_proc_times,
            )
            operations.append(op)

        return operations


    def tick(
        self,
        jobs:         List[Job],
        current_time: float,
        rng:          np.random.Generator,
    ) -> Tuple[List[Job], List[int], List[int]]:
        """
        Advances all job/operation states by one timestep.

        Called AFTER both agents have acted, during physics resolution.

        Args:
            jobs:         Current list of active jobs
            current_time: Current timestep index
            rng:          NumPy Generator

        Returns:
            (updated_jobs, completed_job_ids, freed_machine_ids)
            completed_job_ids: jobs that finished this step
            freed_machine_ids: machines that became free this step
        """
        completed_job_ids  = []
        freed_machine_ids  = []

        for job in jobs:
            if job.is_complete:
                continue

            for op in job.operations:
                if op.status != OpStatus.IN_PROGRESS:
                    continue

                # Decrement remaining processing time by 1 shift
                op.remaining_time -= 1.0
                op.remaining_time  = max(op.remaining_time, 0.0)

                # Check if operation completed this step
                if op.remaining_time <= 0.0:
                    op.status          = OpStatus.DONE
                    op.completion_time = current_time

                    # Free the machine
                    if op.assigned_machine is not None:
                        freed_machine_ids.append(op.assigned_machine)
                        op.assigned_machine = None

                    # Unlock next operation in job if exists
                    next_idx = op.op_idx + 1
                    if next_idx < job.n_ops:
                        job.operations[next_idx].status = OpStatus.READY

            # Check if entire job just completed
            if job.is_complete and job.completion_time is None:
                job.completion_time = float(current_time)
                job.tardiness = max(0.0, current_time - job.due_date)
                completed_job_ids.append(job.job_id)

        return jobs, completed_job_ids, freed_machine_ids


    def assign_operation(
        self,
        jobs:       List[Job],
        job_id:     int,
        op_idx:     int,
        machine_id: int,
        rng:        np.random.Generator,
    ) -> Tuple[List[Job], float]:
        """
        Assigns a READY operation to a machine.
        Called during Agent 2's half-step when action (j, k, m) is taken.

        Samples actual processing time (LogNormal in Phase 2+, nominal in Phase 1).

        Args:
            jobs:       Current job list
            job_id:     j
            op_idx:     k
            machine_id: m
            rng:        NumPy Generator

        Returns:
            (updated_jobs, actual_proc_time_in_shifts)

        Raises:
            ValueError if operation not READY or machine not eligible
        """
        job = next((j for j in jobs if j.job_id == job_id), None)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        op = job.operations[op_idx]

        if op.status != OpStatus.READY:
            raise ValueError(
                f"Op ({job_id},{op_idx}) not READY (status={op.status})"
            )
        if machine_id not in op.eligible_machines:
            raise ValueError(
                f"Machine {machine_id} not eligible for op ({job_id},{op_idx})"
            )

        # Sample actual processing time
        nominal = op.nominal_proc_times[machine_id]
        if self.stoch_level >= 2:
            actual_hours = sample_processing_time(nominal, self.sigma_log, rng)
        else:
            actual_hours = nominal

        # Convert hours to shifts (dt = 8 hours/shift)
        actual_shifts = actual_hours / self.dt

        # Update operation state
        op.status           = OpStatus.IN_PROGRESS
        op.assigned_machine = machine_id
        op.actual_proc_time = actual_shifts
        op.remaining_time   = actual_shifts
        op.start_time       = float(len([
            o for job_ in jobs
            for o in job_.operations
            if o.status == OpStatus.DONE
        ]))  # rough proxy; exact start_time tracking via current_time in env

        return jobs, actual_shifts


    def sample_arrivals(
        self,
        current_time:  float,
        existing_jobs: List[Job],
        rng:           np.random.Generator,
    ) -> List[Job]:
        """
        Phase 3 only: samples new job arrivals this timestep.
        Phase 1 and 2: returns empty list.

        Args:
            current_time:  Current timestep
            existing_jobs: Existing job list (for id generation)
            rng:           NumPy Generator

        Returns:
            List of newly arrived Job objects (may be empty)
        """
        if self.stoch_level < 3:
            return []

        n_new = sample_job_arrivals(self.lambda_arr, rng)
        if n_new == 0:
            return []

        start_id = max((j.job_id for j in existing_jobs), default=-1) + 1
        new_jobs  = self.generate_job_batch(n_new, rng, start_job_id=start_id)

        # Set release time to current timestep
        for job in new_jobs:
            job.release_time = current_time
            # Tighter due dates for mid-episode arrivals
            job.due_date = current_time + float(rng.integers(
                low=job.n_ops * 2, high=min(50, self.t_max_train - int(current_time))
            ))

        return new_jobs


    def compute_weighted_tardiness(self, jobs: List[Job]) -> float:
        """
        Computes total weighted tardiness across all jobs.
        Used in reward computation (scheduling_reward.py).

        Returns:
            sum_j w_j * max(0, C_j - d_j)
        """
        return sum(
            job.weight * job.tardiness
            for job in jobs
            if job.completion_time is not None
        )


    def get_ready_ops(self, jobs: List[Job]) -> List[Tuple[int, int]]:
        """
        Returns list of (job_id, op_idx) for all READY operations.
        Used by Agent 2 to build the valid action set.
        """
        ready = []
        for job in jobs:
            for op in job.operations:
                if op.status == OpStatus.READY:
                    ready.append((job.job_id, op.op_idx))
        return ready


    def get_active_jobs(self, jobs: List[Job]) -> List[Job]:
        """Returns jobs that are released and not yet complete."""
        return [j for j in jobs if not j.is_complete]
