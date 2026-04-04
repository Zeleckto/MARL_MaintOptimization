"""
environments/transitions/job_dynamics.py
=========================================
Job and operation lifecycle with structured operation types.

KEY REDESIGN (Phase 1):
    OLD: each operation has random eligible_machines (60% each)
    NEW: each operation has a typed op_type drawn from config.
         Eligibility is DETERMINISTIC from (op_type -> machine_types -> machine_ids).
         This gives exactly 43% average flexibility (Brandimarte mk01 range).

    Operation types (7):
        Milling   -> Type A,B machines  (60%)
        Drilling  -> Type A,D           (60%)
        Turning   -> Type B,C           (40%)
        Grinding  -> Type C only        (20%)
        Boring    -> Type A,B           (60%)
        Finishing -> Type B,C           (40%)
        Pressing  -> Type D only        (20%)

    Processing times per (op_type, machine_type) from config.
    Due dates = t_arrival + due_date_factor * sum(min_proc_times).

    Phase 3 arrivals: gated behind stochasticity_level >= 3.
    Poisson(lambda_arr) new jobs per shift when active.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# OPERATION STATUS
# ---------------------------------------------------------------------------
class OpStatus:
    PENDING     = 0   # predecessor ops not complete
    READY       = 1   # all predecessors done, can be assigned
    IN_PROGRESS = 2   # being processed on a machine
    DONE        = 3   # completed


# ---------------------------------------------------------------------------
# OPERATION DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class Operation:
    """
    One processing step within a job.
    TGIN Op node features are 10-dim (see to_feature_vector).
    """
    job_id:    int
    op_idx:    int
    op_type:   str   = "Milling"    # operation type name from config
    status:    int   = OpStatus.PENDING

    # Eligible machines (determined by op_type -> machine_type mapping)
    eligible_machines: List[int] = field(default_factory=list)

    # Processing time per eligible machine (in SHIFTS)
    nominal_proc_times: Dict[int, float] = field(default_factory=dict)

    # Assignment tracking
    assigned_machine: Optional[int] = None
    remaining_time:   float = 0.0
    actual_proc_time: float = 0.0
    start_time:       float = 0.0
    completion_time:  float = 0.0

    def to_feature_vector(
        self,
        job_due_date:     float,
        current_time:     float,
        n_total_machines: int,
        n_total_ops:      int,
    ) -> np.ndarray:
        """
        Returns 10-dim feature vector for TGIN Op node.
        Normalisation ranges chosen to keep all values in [-1, 1].
        """
        T_norm = 150.0   # normalisation for time features

        if self.nominal_proc_times:
            proc_vals = list(self.nominal_proc_times.values())
            min_proc = min(proc_vals) / 8.0   # normalise by 8 shifts
            avg_proc = sum(proc_vals) / len(proc_vals) / 8.0
        else:
            min_proc = avg_proc = 0.0

        time_to_due = max(job_due_date - current_time, 0.0) / T_norm
        remaining_norm = self.remaining_time / 8.0

        slack_shifts = (job_due_date - current_time) - self.remaining_time
        slack_norm = np.clip(slack_shifts / T_norm, -1.0, 1.0)

        return np.array([
            self.status / 3.0,                                          # [0] status
            np.clip(min_proc, 0.0, 1.0),                               # [1] min proc time
            np.clip(avg_proc, 0.0, 1.0),                               # [2] avg proc time
            len(self.eligible_machines) / max(n_total_machines, 1),    # [3] flexibility ratio
            self.op_idx / max(n_total_ops - 1, 1),                     # [4] position in job
            np.clip(time_to_due, 0.0, 1.0),                           # [5] time to due
            np.clip(remaining_norm, 0.0, 1.0),                        # [6] remaining time
            float(self.status == OpStatus.READY),                      # [7] is_ready
            float(self.status == OpStatus.IN_PROGRESS),               # [8] is_in_progress
            np.clip(slack_norm, -1.0, 1.0),                           # [9] slack
        ], dtype=np.float32)


OP_FEATURE_DIM = 10


# ---------------------------------------------------------------------------
# JOB DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class Job:
    """
    A job j consisting of n_j ordered operations.
    TGIN Job node features are 7-dim.
    """
    job_id:       int
    release_time: float
    due_date:     float
    weight:       float = 1.0

    operations:      List[Operation] = field(default_factory=list)
    completion_time: Optional[float] = None
    tardiness:       float = 0.0

    @property
    def n_ops(self) -> int:
        return len(self.operations)

    @property
    def is_complete(self) -> bool:
        return all(op.status == OpStatus.DONE for op in self.operations)

    @property
    def current_op_idx(self) -> int:
        for i, op in enumerate(self.operations):
            if op.status != OpStatus.DONE:
                return i
        return self.n_ops

    @property
    def completion_ratio(self) -> float:
        done = sum(1 for op in self.operations if op.status == OpStatus.DONE)
        return done / max(self.n_ops, 1)

    def to_feature_vector(self, current_time: float) -> np.ndarray:
        """Returns 7-dim feature vector for TGIN Job node."""
        T_norm = 150.0

        time_to_due = max(self.due_date - current_time, 0.0) / T_norm

        remaining_ops = [op for op in self.operations if op.status != OpStatus.DONE]
        remaining_work = sum(
            min(op.nominal_proc_times.values())
            if op.nominal_proc_times else 1.0
            for op in remaining_ops
        )
        slack = ((self.due_date - current_time) - remaining_work) / T_norm

        return np.array([
            self.weight / 3.0,                                    # [0] weight
            np.clip(time_to_due, 0.0, 1.0),                      # [1] time to due
            np.clip(slack, -1.0, 1.0),                           # [2] slack
            self.current_op_idx / max(self.n_ops, 1),            # [3] current op
            self.completion_ratio,                                 # [4] completion ratio
            float(current_time > self.due_date),                  # [5] is late
            len(remaining_ops) / max(self.n_ops, 1),             # [6] remaining ops
        ], dtype=np.float32)


JOB_FEATURE_DIM = 7


# ---------------------------------------------------------------------------
# JOB DYNAMICS ENGINE
# ---------------------------------------------------------------------------
class JobDynamicsEngine:
    """
    Manages all job/operation lifecycle transitions.

    Sequence per tick():
        1. Decrement remaining_time for IN_PROGRESS operations
        2. Detect and complete finished operations
        3. PENDING -> READY for unblocked operations
        4. Compute tardiness for completed jobs
        5. Phase 3: Poisson arrivals
    """

    def __init__(self, config: dict):
        self.stoch_level = config.get("stochasticity_level", 1)
        self.dt          = config.get("episode", {}).get("dt_hours", 8.0)
        self.n_ops_min   = config.get("jobs", {}).get("n_ops_per_job_min", 3)
        self.n_ops_max   = config.get("jobs", {}).get("n_ops_per_job_max", 5)
        self.lambda_arr  = config.get("jobs", {}).get("lambda_arr", 0.25)
        self.sigma_log   = config.get("processing", {}).get("sigma_log", 0.15)
        self.n_machines  = len(config.get("machines", []))
        self.t_max       = config.get("episode", {}).get("t_max_train", 150)
        self.due_factor  = config.get("jobs", {}).get("due_date_factor", 1.5)
        self.weights     = config.get("jobs", {}).get("weight_choices", [1, 2, 3])

        # Build eligibility map from config
        # Maps: op_type_name -> list of machine_ids
        self._op_types, self._eligibility_map, self._proc_time_map = \
            self._build_op_maps(config)

        # Map machine_id -> machine_type
        self._machine_type_map: Dict[int, str] = {
            m["machine_id"]: m.get("machine_type", "A")
            for m in config.get("machines", [])
        }

    def _build_op_maps(self, config):
        """
        Builds:
          op_types: list of op type names
          eligibility_map: {op_type: [machine_ids]}
          proc_time_map: {op_type: {machine_id: (min_shifts, max_shifts)}}
        """
        machines_cfg = config.get("machines", [])
        op_types_cfg = config.get("operation_types", [])

        # Map machine_type -> list of machine_ids
        type_to_ids: Dict[str, List[int]] = {}
        for m in machines_cfg:
            t = m.get("machine_type", "A")
            type_to_ids.setdefault(t, []).append(m["machine_id"])

        op_types = []
        eligibility_map = {}
        proc_time_map = {}

        for op_cfg in op_types_cfg:
            name = op_cfg["op_type"]
            op_types.append(name)

            # Build eligible machine_ids
            eligible_ids = []
            for mt in op_cfg.get("eligible_machine_types", []):
                eligible_ids.extend(type_to_ids.get(mt, []))
            eligibility_map[name] = sorted(set(eligible_ids))

            # Build proc time ranges per machine_id
            pt_cfg = op_cfg.get("proc_time_by_type", {})
            proc_per_machine = {}
            for m in machines_cfg:
                mid = m["machine_id"]
                mt  = m.get("machine_type", "A")
                if mid in eligibility_map[name] and mt in pt_cfg:
                    lo, hi = pt_cfg[mt]
                    proc_per_machine[mid] = (float(lo), float(hi))
            proc_time_map[name] = proc_per_machine

        # Fallback: if no op_types in config, use uniform random (old behaviour)
        if not op_types:
            op_types = ["Generic"]
            all_ids = [m["machine_id"] for m in machines_cfg]
            eligibility_map["Generic"] = all_ids
            proc_time_map["Generic"] = {
                m["machine_id"]: (2.0, 8.0) for m in machines_cfg
            }

        return op_types, eligibility_map, proc_time_map

    def generate_job_batch(
        self,
        n_jobs: int,
        rng: np.random.Generator,
        start_job_id: int = 0,
    ) -> List[Job]:
        """
        Generates a batch of jobs at episode start (Phases 1 and 2).
        All released at t=0 with due dates based on processing load.
        """
        jobs = []
        for j in range(n_jobs):
            job_id = start_job_id + j
            n_ops  = int(rng.integers(self.n_ops_min, self.n_ops_max + 1))
            weight = float(rng.choice(self.weights))

            # Generate operations first to compute due date
            operations = self._generate_operations(job_id, n_ops, rng)

            # Due date = 1.5 * sum of minimum processing times across all ops
            # This creates achievable but tight schedules (~5-8 jobs late)
            total_min_proc = sum(
                min(op.nominal_proc_times.values()) if op.nominal_proc_times else 2.0
                for op in operations
            )
            due_date = float(total_min_proc * self.due_factor) + float(
                rng.integers(0, max(int(self.t_max * 0.3), 5))
            )
            due_date = min(due_date, float(self.t_max - 5))
            due_date = max(due_date, float(n_ops * 2))

            # First op is READY
            if operations:
                operations[0].status = OpStatus.READY

            job = Job(
                job_id       = job_id,
                release_time = 0.0,
                due_date     = due_date,
                weight       = weight,
                operations   = operations,
            )
            jobs.append(job)

        return jobs

    def _generate_operations(
        self, job_id: int, n_ops: int, rng: np.random.Generator
    ) -> List[Operation]:
        """
        Generates n_ops operations for a job.
        Each operation gets an op_type sampled from the pool.
        Ensures at least one eligible machine exists (guaranteed by design).
        """
        operations = []
        op_type_indices = rng.integers(0, len(self._op_types), size=n_ops)

        for k, ti in enumerate(op_type_indices):
            op_type = self._op_types[ti]
            eligible = self._eligibility_map[op_type]

            # Fallback: if somehow no eligible machines, use first op type
            attempts = 0
            while not eligible and attempts < len(self._op_types):
                ti = (ti + 1) % len(self._op_types)
                op_type = self._op_types[ti]
                eligible = self._eligibility_map[op_type]
                attempts += 1

            if not eligible:
                eligible = list(range(self.n_machines))

            # Sample processing times per eligible machine
            pt_ranges = self._proc_time_map.get(op_type, {})
            nominal_proc_times = {}
            for mid in eligible:
                if mid in pt_ranges:
                    lo, hi = pt_ranges[mid]
                    t = float(rng.uniform(lo, hi))
                else:
                    t = float(rng.uniform(2.0, 8.0))
                nominal_proc_times[mid] = t

            op = Operation(
                job_id             = job_id,
                op_idx             = k,
                op_type            = op_type,
                status             = OpStatus.PENDING,
                eligible_machines  = list(eligible),
                nominal_proc_times = nominal_proc_times,
            )
            operations.append(op)

        return operations

    def tick(
        self,
        jobs: List[Job],
        current_time: float,
        rng: np.random.Generator,
    ) -> Tuple[List[Job], List[int], List[int]]:
        """
        Advances all job/operation states by one shift.
        Called after both agents have acted.

        Returns:
            (jobs, completed_job_ids, freed_machine_ids)
        """
        completed_job_ids = []
        freed_machine_ids = []

        for job in jobs:
            if job.is_complete:
                continue

            for op in job.operations:
                if op.status != OpStatus.IN_PROGRESS:
                    continue

                op.remaining_time -= 1.0
                op.remaining_time  = max(op.remaining_time, 0.0)

                if op.remaining_time <= 0.0:
                    op.status          = OpStatus.DONE
                    op.completion_time = current_time

                    if op.assigned_machine is not None:
                        freed_machine_ids.append(op.assigned_machine)
                        op.assigned_machine = None

                    # Unlock next op
                    next_idx = op.op_idx + 1
                    if next_idx < job.n_ops:
                        job.operations[next_idx].status = OpStatus.READY

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
        Returns (jobs, actual_proc_time_in_shifts).
        """
        job = next((j for j in jobs if j.job_id == job_id), None)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        op = job.operations[op_idx]

        if op.status != OpStatus.READY:
            raise ValueError(f"Op ({job_id},{op_idx}) not READY (status={op.status})")
        if machine_id not in op.eligible_machines:
            raise ValueError(f"Machine {machine_id} not eligible for op ({job_id},{op_idx})")

        nominal = op.nominal_proc_times.get(machine_id, 4.0)

        # Phase 2+: add LogNormal noise
        if self.stoch_level >= 2:
            actual = float(rng.lognormal(
                mean=np.log(nominal) - 0.5 * self.sigma_log ** 2,
                sigma=self.sigma_log
            ))
            actual = max(actual, 0.5)  # minimum 0.5 shifts
        else:
            actual = nominal

        op.status           = OpStatus.IN_PROGRESS
        op.assigned_machine = machine_id
        op.actual_proc_time = actual
        op.remaining_time   = actual

        return jobs, actual

    def sample_arrivals(
        self,
        current_time:  float,
        existing_jobs: List[Job],
        rng:           np.random.Generator,
    ) -> List[Job]:
        """
        Phase 3 only: Poisson job arrivals this shift.
        Gated behind stochasticity_level >= 3.
        """
        if self.stoch_level < 3:
            return []

        n_new = rng.poisson(self.lambda_arr)
        if n_new == 0:
            return []

        start_id = max((j.job_id for j in existing_jobs), default=-1) + 1
        new_jobs  = self.generate_job_batch(n_new, rng, start_job_id=start_id)

        remaining = self.t_max - int(current_time)
        for job in new_jobs:
            job.release_time = current_time
            # Tighter due dates for mid-episode arrivals
            total_min_proc = sum(
                min(op.nominal_proc_times.values()) if op.nominal_proc_times else 2.0
                for op in job.operations
            )
            extra = max(int(remaining * 0.3), 5)
            job.due_date = current_time + total_min_proc * self.due_factor + \
                float(rng.integers(0, extra + 1))
            job.due_date = min(job.due_date, float(self.t_max - 2))

        return new_jobs

    def compute_weighted_tardiness(self, jobs: List[Job]) -> float:
        return sum(
            j.weight * j.tardiness
            for j in jobs if j.completion_time is not None
        )

    def get_ready_ops(self, jobs: List[Job]) -> List[Tuple[int, int]]:
        return [
            (job.job_id, op.op_idx)
            for job in jobs
            for op in job.operations
            if op.status == OpStatus.READY
        ]

    def get_active_jobs(self, jobs: List[Job]) -> List[Job]:
        return [j for j in jobs if not j.is_complete]

    def get_eligibility_stats(self) -> dict:
        """
        Returns eligibility statistics for checker validation.
        Target: ~43% average flexibility.
        """
        if not self._op_types:
            return {}

        total_eligible = sum(len(self._eligibility_map[ot]) for ot in self._op_types)
        total_possible = len(self._op_types) * self.n_machines
        avg_flexibility = total_eligible / max(total_possible, 1)

        return {
            "op_types": self._op_types,
            "eligibility_map": {k: v for k, v in self._eligibility_map.items()},
            "avg_flexibility": avg_flexibility,
            "total_eligible": total_eligible,
            "total_possible": total_possible,
        }
